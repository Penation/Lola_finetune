#!/usr/bin/env python
"""
Distributed startup probe for LoLA CALVIN training.

Purpose:
- Reproduce the same multi-node / multi-GPU startup path as train_lola_azure.py
- Emit precise phase logs so we can see whether the hang is in:
  - distributed init
  - dataset metadata / dataset / dataloader creation
  - model load
  - FSDP wrap
  - optimizer creation
  - model.train()
  - wandb.init()
  - distributed barrier
  - first dataloader batch
  - first forward / backward / optimizer step
"""

import argparse
import datetime
import logging
import os
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_lola_azure as train_mod
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig


logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] [Rank {os.environ.get('RANK', '0')}] [probe] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _sync_cuda_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def phase(name: str, fn):
    logger.info(f">>> {name} START")
    start = time.monotonic()
    try:
        result = fn()
        _sync_cuda_if_needed()
    except Exception:
        elapsed = time.monotonic() - start
        logger.exception(f"!!! {name} FAIL after {elapsed:.2f}s")
        raise
    elapsed = time.monotonic() - start
    logger.info(f"<<< {name} DONE in {elapsed:.2f}s")
    return result


def barrier_phase(name: str):
    if not dist.is_available() or not dist.is_initialized():
        logger.info(f"--- {name}: skipped (not distributed)")
        return
    logger.info(f">>> barrier:{name} ENTER")
    start = time.monotonic()
    dist.barrier()
    elapsed = time.monotonic() - start
    logger.info(f"<<< barrier:{name} EXIT in {elapsed:.2f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe LoLA distributed startup path")

    parser.add_argument("--dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--video_backend",
        type=str,
        default=None,
        choices=["pyav", "torchvision", "torchcodec"],
    )

    parser.add_argument("--strategy", type=str, default="fsdp", choices=["ddp", "fsdp"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_every_n_steps", type=int, default=0)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--vlm_path", type=str, required=True)
    parser.add_argument("--train_vlm", action="store_true")
    parser.add_argument("--stage_train_vlm_after_epoch", type=int, default=0)
    parser.add_argument("--save_checkpoint_on_vlm_unfreeze", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="/tmp/lola_probe")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--n_obs_steps", type=int, default=1)

    parser.add_argument("--load_full_history", action="store_true")
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--history_padding_side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--convert_calvin_rpy_to_ortho6d", action="store_true")
    parser.add_argument("--calvin_xyz_only_normalize", action="store_true")

    parser.add_argument("--wandb_project", type=str, default="lola-azure-probe")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

    parser.add_argument("--probe_skip_wandb", action="store_true")
    parser.add_argument("--probe_skip_batch", action="store_true")
    parser.add_argument("--probe_skip_backward", action="store_true")
    return parser


def maybe_init_wandb(trainer: train_mod.LoLATrainer, run_name: str):
    if not trainer.use_wandb:
        logger.info("wandb disabled on this rank")
        return None

    service_wait_s = int(os.environ.get("WANDB__SERVICE_WAIT", os.environ.get("WANDB_SERVICE_WAIT", "120")))
    init_timeout_s = int(os.environ.get("WANDB_INIT_TIMEOUT", "120"))
    start_method = os.environ.get("WANDB_START_METHOD", "thread")

    logger.info(
        f"wandb config: run={run_name}, start_method={start_method}, "
        f"service_wait={service_wait_s}s, init_timeout={init_timeout_s}s"
    )

    return train_mod.wandb.init(
        project=trainer.wandb_project,
        name=run_name,
        entity=trainer.wandb_entity,
        id=trainer.wandb_id,
        resume="allow" if trainer.wandb_id else None,
        settings=train_mod.wandb.Settings(
            start_method=start_method,
            init_timeout=init_timeout_s,
            _service_wait=service_wait_s,
        ),
        config={
            "learning_rate": trainer.learning_rate,
            "max_steps": trainer.max_steps,
            "batch_size": 1,
            "strategy": trainer.strategy,
            "world_size": trainer.world_size,
        },
    )


def main():
    dist_info = phase("setup_distributed", train_mod.setup_distributed)
    args = build_parser().parse_args()

    if args.dataset_repo_id is None and args.dataset_root is None:
        raise ValueError("Either --dataset_repo_id or --dataset_root must be provided.")

    logger.info(
        f"hostname={os.uname().nodename} local_rank={dist_info['local_rank']} "
        f"world_rank={dist_info['world_rank']} world_size={dist_info['world_size']} "
        f"device={dist_info['device']}"
    )

    if args.dataset_root is not None:
        info_path = os.path.join(args.dataset_root, "meta", "info.json")
        if not os.path.isfile(info_path):
            raise FileNotFoundError(f"Dataset root is missing metadata file: {info_path}")

    dataset_metadata = phase(
        "load_dataset_metadata",
        lambda: train_mod.LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root),
    )
    logger.info(
        f"dataset_metadata: repo_id={args.dataset_repo_id} root={args.dataset_root} "
        f"episodes={dataset_metadata.total_episodes} frames={dataset_metadata.total_frames} fps={dataset_metadata.fps}"
    )

    features = phase("dataset_to_policy_features", lambda: dataset_to_policy_features(dataset_metadata.features))
    dataset_stats: dict[str, dict[str, Any]] = dataset_metadata.stats
    batch_transform = None

    calvin_translation_only_mode = args.convert_calvin_rpy_to_ortho6d or args.calvin_xyz_only_normalize
    if calvin_translation_only_mode:
        if train_mod.is_calvin_single_arm_rpy_dataset(dataset_metadata):
            logger.info("detected legacy CALVIN 7D xyz+rpy+gripper dataset")
            features, action_dim = phase(
                "build_calvin_ortho6d_features",
                lambda: train_mod.build_calvin_ortho6d_features(features),
            )
            dataset_stats = phase(
                "build_calvin_partial_normalization_stats",
                lambda: train_mod.build_calvin_partial_normalization_stats(dataset_stats),
            )
            batch_transform = train_mod.CalvinSingleArmBatchTransform(dataset_stats["action"])
        elif train_mod.is_calvin_ortho6d_dataset(dataset_metadata):
            action_dim = features["action"].shape[0]
            logger.info(f"detected CALVIN ortho6d dataset action_dim={action_dim}")
            dataset_stats = phase(
                "build_translation_only_normalization_stats",
                lambda: train_mod.build_translation_only_normalization_stats(dataset_stats, action_dim),
            )
            batch_transform = train_mod.TranslationOnlyActionBatchTransform(dataset_stats["action"])
        else:
            raise ValueError("CALVIN translation-only normalization requested, but dataset layout was not recognized.")
    elif "action" in features:
        action_dim = features["action"].shape[0]
    else:
        action_dim = args.action_dim

    initial_train_vlm = args.train_vlm and args.stage_train_vlm_after_epoch <= 0
    logger.info(f"resolved action_dim={action_dim} initial_train_vlm={initial_train_vlm}")

    config = phase(
        "build_lola_config",
        lambda: LoLAConfig(
            vlm_model_name="Qwen/Qwen3.5-4B",
            vlm_path=args.vlm_path,
            action_dim=action_dim,
            action_chunk_size=args.action_chunk_size,
            pred_chunk_size=args.pred_chunk_size,
            n_obs_steps=args.n_obs_steps,
            input_features={key: ft for key, ft in features.items() if ft.type != FeatureType.ACTION},
            output_features={key: ft for key, ft in features.items() if ft.type == FeatureType.ACTION},
            train_vlm=initial_train_vlm,
            load_full_history=args.load_full_history,
            max_history_length=args.max_history_length,
            history_padding_side=args.history_padding_side,
        ),
    )

    train_dataset = phase(
        "create_lola_dataset",
        lambda: train_mod.create_lola_dataset(
            repo_id=args.dataset_repo_id,
            config=config,
            root=args.dataset_root,
            episodes=args.episodes,
            video_backend=args.video_backend,
            use_lola_dataset=args.load_full_history,
            max_history_length=args.max_history_length,
            history_padding_side=args.history_padding_side,
        ),
    )
    logger.info(f"dataset_len={len(train_dataset)}")
    barrier_phase("after_dataset")

    sampler = None
    shuffle = True
    if dist_info["is_distributed"]:
        sampler = phase(
            "build_distributed_sampler",
            lambda: DistributedSampler(
                train_dataset,
                num_replicas=dist_info["world_size"],
                rank=dist_info["world_rank"],
                shuffle=True,
            ),
        )
        shuffle = False

    train_loader = phase(
        "build_dataloader",
        lambda: DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=args.num_workers,
            collate_fn=train_mod.collate_fn,
            pin_memory=True,
            drop_last=True,
        ),
    )
    logger.info(f"dataloader_len={len(train_loader)} num_workers={args.num_workers}")
    barrier_phase("after_dataloader")

    trainer = phase(
        "build_trainer",
        lambda: train_mod.LoLATrainer(
            config=config,
            dataset_stats=dataset_stats,
            dist_info=dist_info,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            train_vlm=initial_train_vlm,
            strategy=args.strategy,
            gradient_clip_val=args.gradient_clip_val,
            ckpt_dir=args.ckpt_dir,
            save_every_n_steps=args.save_every_n_steps,
            log_every_n_steps=args.log_every_n_steps,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            wandb_entity=args.wandb_entity,
            wandb_id=args.wandb_id,
            batch_transform=batch_transform,
            stage_train_vlm_after_epoch=args.stage_train_vlm_after_epoch,
            save_checkpoint_on_vlm_unfreeze=args.save_checkpoint_on_vlm_unfreeze,
        ),
    )

    if args.disable_wandb or args.probe_skip_wandb:
        trainer.use_wandb = False

    phase("setup_model", trainer.setup_model)
    barrier_phase("after_setup_model")

    phase("setup_optimizer", trainer.setup_optimizer)
    barrier_phase("after_setup_optimizer")

    phase("model.train", trainer.model.train)
    barrier_phase("after_model_train")

    wandb_run = None
    probe_run_name = args.wandb_name or f"probe-{args.strategy}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if trainer.use_wandb:
        wandb_run = phase("wandb.init", lambda: maybe_init_wandb(trainer, probe_run_name))
    else:
        logger.info("wandb probe skipped")
    barrier_phase("after_wandb")

    if not args.probe_skip_batch:
        loader_iter = iter(train_loader)
        batch = phase("fetch_first_batch", lambda: next(loader_iter))
        logger.info(f"first_batch_keys={sorted(batch.keys())}")
        barrier_phase("after_fetch_first_batch")

        phase("optimizer.zero_grad", trainer.optimizer.zero_grad)
        loss, loss_dict = phase("training_step", lambda: trainer.training_step(batch))
        logger.info(f"probe_loss={float(loss.detach().item()):.6f} loss_dict_keys={list(loss_dict.keys())}")
        barrier_phase("after_training_step")

        if not args.probe_skip_backward:
            phase("backward", loss.backward)
            barrier_phase("after_backward")

            if trainer.gradient_clip_val > 0:
                phase(
                    "clip_grad_norm",
                    lambda: torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.gradient_clip_val),
                )

            phase("optimizer.step", trainer.optimizer.step)
            phase("scheduler.step", trainer.scheduler.step)
            barrier_phase("after_optimizer_step")
    else:
        logger.info("batch probe skipped")

    if wandb_run is not None:
        phase("wandb.finish", train_mod.wandb.finish)

    barrier_phase("before_cleanup")
    phase("cleanup_distributed", train_mod.cleanup_distributed)
    logger.info("probe completed successfully")


if __name__ == "__main__":
    main()
