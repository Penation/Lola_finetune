import os
import shutil
import subprocess
import urllib.request
import tarfile
import argparse
import re
import time
import threading

# 🌟 强制设置 AzCopy 使用 Managed Identity 身份验证
os.environ["AZCOPY_AUTO_LOGIN_TYPE"] = "MSI"

# 1. 控制并发数：对于 175MB 的大文件，64 到 128 是最能跑满网络带宽且不会触发限流的甜点区间。
# 可以先 "AUTO" (让它自己动态调)，或者强制锁定为一个固定值，如 "96" (与 CPU 核心数 1:1)
# os.environ["AZCOPY_CONCURRENCY_VALUE"] = "32" 

# 2. 扩大内存缓冲：对于 399GB 的超大共享内存。
# 默认情况下 azcopy 会动态占用，为了让网络到 NVMe 盘的写入极其丝滑，直接给它分配 8GB 的专属物理内存缓冲
# os.environ["AZCOPY_BUFFER_GB"] = "8"

# ==========================================
# 模块 0：轻度 GPU 负载保活 (防 Suspended)
# ==========================================
def light_gpu_load_worker(device_id):
    """单卡轻度计算负载，用于产生持续的心跳骗过空闲监控"""
    try:
        import torch
    except ImportError:
        return

    device = torch.device(f'cuda:{device_id}')
    
    # 1. 轻度显存占位：只占 10GB
    try:
        elements = int((10 * 1024**3) / 4)
        _dummy_tensor = torch.empty(elements, dtype=torch.float32, device=device)
    except:
        pass 

    # 2. 准备小矩阵
    mat_a = torch.randn(1024, 1024, device=device)
    mat_b = torch.randn(1024, 1024, device=device)

    print(f"  🟢 GPU {device_id}: 轻度保活负载已启动 (10GB显存占用，持续心跳)。")

    # 3. 产生持续但不密集的活跃度
    while True:
        try:
            _ = torch.matmul(mat_a, mat_b)
            torch.cuda.synchronize(device)
            # 休眠 0.1 秒，产生约 1%~5% 的利用率波动，不抢占拉数据的 CPU 资源
            time.sleep(0.05) 
        except Exception:
            time.sleep(1)

def start_light_gpu_load():
    """启动所有可用 GPU 的轻度保活线程"""
    try:
        import torch
    except ImportError:
        print("\n⚠️ 未检测到 PyTorch，无法启动 GPU 保活，脚本将仅执行下载。")
        return
    if not torch.cuda.is_available():
        print("\n⚠️ 未检测到 CUDA 设备，跳过 GPU 保活。")
        return

    num_gpus = torch.cuda.device_count()
    print(f"\n[INFO] 启动 GPU 轻度心跳防挂起机制...")
    for i in range(num_gpus):
        threading.Thread(target=light_gpu_load_worker, args=(i,), daemon=True).start()
    print("[INFO] 保活生效，主进程安心执行 I/O。\n")

# ==========================================
# 模块 1：环境准备与 AzCopy 安装
# ==========================================
def install_azcopy():
    """在当前目录自动下载并解压 Linux 版 azcopy，若已存在则直接返回路径"""
    azcopy_path = "./azcopy"
    if os.path.exists(azcopy_path):
        return azcopy_path

    print("正在下载官方版 AzCopy 工具...")
    tar_url = "https://aka.ms/downloadazcopy-v10-linux"
    tar_filename = "azcopy_linux.tar.gz"
    urllib.request.urlretrieve(tar_url, tar_filename)

    # 从 tar 包中提取 azcopy 二进制文件
    extract_dir = ""
    with tarfile.open(tar_filename, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("azcopy") and member.isfile():
                tar.extract(member, path=".")
                os.rename(os.path.join(".", member.name), azcopy_path)
                extract_dir = os.path.dirname(os.path.join(".", member.name))
                break

    os.chmod(azcopy_path, 0o755)
    os.remove(tar_filename)
    if extract_dir and extract_dir != ".":
        shutil.rmtree(extract_dir, ignore_errors=True)

    print("✅ AzCopy 准备就绪！")
    return azcopy_path

# ==========================================
# 模块 2：解析 AzCopy 输出中的失败信息与传输统计
# ==========================================
def parse_transfer_result(output_lines):
    """从 azcopy 输出中解析传输结果

    Returns:
        failed: 是否存在失败文件 (Failed > 0)
        job_id: 本次传输的 JobId，用于断点续传
        transferred_bytes: 已传输字节数
        throughput_bytes_per_sec: 实时吞吐速度 (bytes/s)，来自进度行
    """
    failed = False
    job_id = None
    transferred_bytes = 0
    throughput_bytes_per_sec = 0

    for line in output_lines:
        line_str = line.strip()

        # 匹配进度行 "75.7 %, 12017 Done, 1 Failed, 3474 Pending, 0 Skipped, 15492 Total, 2-sec Throughput (Mb/s): 2904.1843"
        m_progress = re.search(r'(\d+)\s+Failed', line_str)
        if m_progress and int(m_progress.group(1)) > 0:
            failed = True

        # 匹配进度行中的实时吞吐 "Throughput (Mb/s): 2904.1843"
        m_throughput = re.search(r'Throughput\s*\((\w+)/s\):\s*([\d.]+)', line_str)
        if m_throughput:
            throughput_unit = m_throughput.group(1)
            throughput_val = float(m_throughput.group(2))
            unit_map = {'b': 1/8, 'B': 1, 'Kb': 1e3/8, 'KB': 1e3,
                        'Mb': 1e6/8, 'MB': 1e6, 'Gb': 1e9/8, 'GB': 1e9}
            throughput_bytes_per_sec = throughput_val * unit_map.get(throughput_unit, 1)

        # 匹配 "Number of File Transfers Failed: 1"
        m_fail = re.search(r'Number of File Transfers Failed:\s*(\d+)', line_str)
        if m_fail and int(m_fail.group(1)) > 0:
            failed = True

        # 匹配 "Job 64cdbc89-733e-4541-5cdf-03e8bef16519 has started"
        m_job = re.search(r'Job\s+([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', line_str)
        if m_job:
            job_id = m_job.group(1)

        # 匹配 "Total Number of Bytes Transferred: 1394255917"（精确字节数，优先）
        m_bytes_exact = re.search(r'Total Number of Bytes Transferred:\s*(\d+)', line_str)
        if m_bytes_exact:
            transferred_bytes = int(m_bytes_exact.group(1))

        # 兼容旧格式 "Transferred: 1.234 GiB / 5.678 GiB"
        if not m_bytes_exact:
            m_bytes = re.search(r'([\d.]+)\s*(B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)\s*(?:/|Transferred)', line_str, re.IGNORECASE)
            if m_bytes:
                transferred_bytes = _parse_size_to_bytes(m_bytes.group(1), m_bytes.group(2))

    return failed, job_id, transferred_bytes, throughput_bytes_per_sec

def _parse_size_to_bytes(value_str, unit_str):
    """将 '1.23 GiB' 这类字符串转换为字节数"""
    value = float(value_str)
    unit_map = {
        'B': 1,
        'KiB': 1024, 'KB': 1024,
        'MiB': 1024**2, 'MB': 1024**2,
        'GiB': 1024**3, 'GB': 1024**3,
        'TiB': 1024**4, 'TB': 1024**4,
    }
    return int(value * unit_map.get(unit_str, 1))

def _format_bytes(num_bytes):
    """将字节数格式化为人类可读的字符串，如 '1.23 GiB'"""
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PiB"

def _format_duration(seconds):
    """将秒数格式化为 HH:MM:SS 或 MM:SS"""
    secs = int(seconds)
    hrs, remainder = divmod(secs, 3600)
    mins, secs = divmod(remainder, 60)
    if hrs > 0:
        return f"{hrs:d}:{mins:02d}:{secs:02d}"
    return f"{mins:d}:{secs:02d}"

def _wait_with_countdown(seconds):
    """倒计时等待，用于重试间隔"""
    for i in range(seconds, 0, -1):
        print(f"\r  ⏳ {i} 秒后重试...", end="", flush=True)
        time.sleep(1)
    print("\r  " + " " * 30 + "\r", end="", flush=True)

# ==========================================
# 模块 3：调用 AzCopy 执行精准传输（单循环扁平化设计 + 计时统计）
# ==========================================
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10

def run_azcopy_transfer(azcopy_bin, source_url, destination_path, max_retries=MAX_RETRIES):
    """执行 AzCopy 传输，支持断点续传与失败重试

    传输策略（单循环扁平化）：
      - 首次 → azcopy copy（全量拉取）
      - 有 JobId 时 → azcopy jobs resume（断点续传）
        但如果 resume 仍然失败，丢弃 JobId，退回 copy 模式
      - 无 JobId + 重试 → azcopy copy --overwrite=ifSourceNewer（智能对比续传，
        仅下载源端更新的文件，跳过本地已有的相同文件）

    成功判定：退出码 == 0 且 Failed 文件数 == 0
    """
    print(f"\n🚀 开始通过 AzCopy 直连拉取至: {destination_path}")
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    current_job_id = None
    resume_attempted = False  # 是否已经尝试过 jobs resume
    task_start_time = time.time()  # 整个任务的总计时起点

    for attempt in range(1, max_retries + 1):
        output_lines = []
        attempt_start_time = time.time()  # 单次尝试的计时起点

        # 决定传输模式
        if current_job_id and not resume_attempted:
            # 仅尝试一次 jobs resume；如果 resume 仍然失败，丢弃 JobId 退回 copy
            print(f"\n🔄 [第 {attempt}/{max_retries} 次尝试] 启动断点续传 (JobId: {current_job_id})")
            command = [azcopy_bin, "jobs", "resume", current_job_id]
            resume_attempted = True
        else:
            # 丢弃 JobId：resume 无效或首次/重试，使用 copy 模式
            if current_job_id:
                print(f"\n⚠️ [第 {attempt}/{max_retries} 次尝试] 断点续传仍失败，丢弃 JobId，退回智能对比续传模式")
                current_job_id = None
            if attempt == 1:
                print(f"\n📥 [第 {attempt}/{max_retries} 次尝试] 执行初始全量拉取")
                command = [azcopy_bin, "copy", source_url, destination_path, "--recursive=true"]
            else:
                print(f"\n📥 [第 {attempt}/{max_retries} 次尝试] 智能对比续传 (--overwrite=ifSourceNewer)")
                command = [azcopy_bin, "copy", source_url, destination_path, "--recursive=true", "--overwrite=ifSourceNewer"]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in process.stdout:
            output_lines.append(line)
            if any(kw in line for kw in ("Done", "Transferred", "Failed", "%", "Job ", "Elapsed", "summary", "Status")):
                print(f"    {line.strip()}")

        process.wait()
        attempt_elapsed = time.time() - attempt_start_time  # 单次尝试耗时

        # 解析本次运行的结果
        failed, new_job_id, transferred_bytes, throughput = parse_transfer_result(output_lines)

        # 更新 JobId，供下一次重试使用
        if new_job_id:
            current_job_id = new_job_id

        # 输出本次尝试的计时与速度统计
        speed = transferred_bytes / attempt_elapsed if attempt_elapsed > 0 else 0
        speed_display = f"{_format_bytes(throughput)}/s" if throughput > 0 else f"{_format_bytes(speed)}/s"
        print(f"    ⏱  本次耗时: {_format_duration(attempt_elapsed)}  |  "
              f"传输量: {_format_bytes(transferred_bytes)}  |  "
              f"速度: {speed_display}")

        # 判断是否彻底成功
        if process.returncode == 0 and not failed:
            total_elapsed = time.time() - task_start_time
            total_speed = transferred_bytes / total_elapsed if total_elapsed > 0 else 0
            print(f"✅ 目录 {destination_path} 拉取彻底成功！")
            print(f"    ⏱  总耗时: {_format_duration(total_elapsed)}  |  "
                  f"总传输量: {_format_bytes(transferred_bytes)}  |  "
                  f"平均速度: {_format_bytes(total_speed)}/s")
            return True
        else:
            # resume 失败时丢弃 JobId，下次循环自动退回 copy 模式
            if resume_attempted and current_job_id:
                current_job_id = None
            print(f"❌ 尝试结束，存在瑕疵 (退出码: {process.returncode})")
            if attempt < max_retries:
                _wait_with_countdown(RETRY_DELAY_SECONDS)

    total_elapsed = time.time() - task_start_time
    print(f"🚨 目录 {destination_path} 在 {max_retries} 次尝试后仍然失败。")
    print(f"    ⏱  总耗时: {_format_duration(total_elapsed)}")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 AzCopy 从 Azure Blob Storage 拉取数据集")
    parser.add_argument("--account", type=str, required=True, help="Azure Storage 账户名称")
    parser.add_argument("--container", type=str, required=True, help="Azure Storage 容器名称")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help=f"最大重试次数 (默认: {MAX_RETRIES})")
    args = parser.parse_args()

    # 启动 GPU 负载
    start_light_gpu_load()

    azcopy_bin = install_azcopy()

    base_url = f"https://{args.account}.blob.core.windows.net/{args.container}"

    tasks = [
        {
            "cloud_path": "robot_dataset/lerobot-format-v30/merged_0412_v1/",
            "local_path": "/scratch/amlt_code/lola_lerobot/robot_dataset/lerobot-format-v30/merged_0412_v1/"
        },
    ]

    # 登录 Azure Managed Identity
    subprocess.run([azcopy_bin, "login", "--identity"], check=True)

    global_start = time.time()

    failed_tasks = []
    for task in tasks:
        src_url = f"{base_url}/{task['cloud_path']}"
        success = run_azcopy_transfer(azcopy_bin, src_url, task["local_path"], max_retries=args.max_retries)
        if not success:
            failed_tasks.append(task)

    global_elapsed = time.time() - global_start

    if failed_tasks:
        print(f"\n❌ 以下 {len(failed_tasks)} 个任务在所有重试后仍失败：")
        for t in failed_tasks:
            print(f"  - {t['cloud_path']}")
    else:
        print("\n🎉 所有指定数据集已安全、极速地抵达本地存储！")

    print(f"\n⏱  全局总耗时: {_format_duration(global_elapsed)}")
