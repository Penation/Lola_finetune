import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# 线程安全的进度条（使用锁）
class ThreadSafeTqdm:
    def __init__(self, total, desc):
        self.pbar = tqdm(total=total, desc=desc, unit='file', dynamic_ncols=True)
        self.lock = threading.Lock()
    
    def update(self, n=1, postfix=None):
        with self.lock:
            if postfix:
                self.pbar.set_postfix_str(postfix)
            self.pbar.update(n)
    
    def close(self):
        with self.lock:
            self.pbar.close()

def copy_file_task(src_file: Path, dst_file: Path, progress: ThreadSafeTqdm):
    """复制单个文件任务（线程安全进度更新）"""
    try:
        # 确保目标目录存在
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 复制文件（使用缓冲减少内存占用）
        with open(src_file, 'rb') as fsrc, open(dst_file, 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
        
        # 保留文件元数据
        shutil.copystat(src_file, dst_file)
        
        # 更新进度（成功）
        progress.update(postfix=f"✓ {src_file.name[:25]}...")
        return True
    except Exception as e:
        # 更新进度（失败）
        progress.update(postfix=f"✗ {src_file.name[:20]}... {str(e)[:30]}")
        return False

def recursive_copy_with_progress(src_root: str, dst_root: str, max_workers: int = 8):
    """递归复制目录（带进度条）"""
    src_path = Path(src_root).resolve()
    dst_path = Path(dst_root).resolve()
    
    if not src_path.exists():
        raise FileNotFoundError(f"源目录不存在: {src_root}")
    
    # 1. 递归收集所有文件（原生方案）
    all_files = []
    for root, _, files in os.walk(str(src_path)):
        for file in files:
            src_file = Path(root) / file
            # 计算相对路径
            rel_path = src_file.relative_to(src_path)
            dst_file = dst_path / rel_path
            all_files.append((src_file, dst_file))
    
    total_files = len(all_files)
    if total_files == 0:
        print("没有需要复制的文件")
        return
    
    print(f"发现 {total_files} 个文件 | 源: {src_path} → 目标: {dst_path}")
    
    # 2. 创建线程安全的进度条
    progress = ThreadSafeTqdm(total=total_files, desc="递归复制中")
    
    # 3. 多线程复制
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [
            executor.submit(copy_file_task, src, dst, progress)
            for src, dst in all_files
        ]
        
        # 收集结果（不阻塞进度显示）
        for future in as_completed(futures):
            if future.result():
                success_count += 1
    
    # 4. 清理进度条
    progress.close()
    
    # 5. 最终报告
    print(f"\n复制完成! 成功: {success_count}/{total_files} | 失败: {total_files - success_count}")
    if success_count < total_files:
        print(f"警告: {total_files - success_count} 个文件复制失败（详情见进度条中的错误标记）")

if __name__ == "__main__":
    # 配置路径
    directory_A = "/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/merged_0412_v1/"  # blobfuse挂载的源目录
    directory_B = "/scratch/amlt_code/lola_lerobot/robot_dataset/lerobot-format-v30/merged_0412_v1/"         # 本地目标目录
    
    # 调整线程数（根据实际环境优化）
    MAX_WORKERS = 128  # 云环境建议 16-32，普通PC建议 8-16
    
    recursive_copy_with_progress(directory_A, directory_B, max_workers=MAX_WORKERS)
