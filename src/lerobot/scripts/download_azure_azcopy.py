import os
import shutil
import subprocess
import urllib.request
import tarfile
import argparse
import re
import time

# 🌟 强制设置 AzCopy 使用 Managed Identity 身份验证
os.environ["AZCOPY_AUTO_LOGIN_TYPE"] = "MSI"

# ==========================================
# 模块 1：环境准备与 AzCopy 安装
# ==========================================
def install_azcopy():
    azcopy_path = "./azcopy"
    if os.path.exists(azcopy_path):
        return azcopy_path

    print("正在下载官方版 AzCopy 工具...")
    tar_url = "https://aka.ms/downloadazcopy-v10-linux"
    tar_filename = "azcopy_linux.tar.gz"
    urllib.request.urlretrieve(tar_url, tar_filename)

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
# 模块 2：解析 AzCopy 输出中的失败信息
# ==========================================
def parse_transfer_result(output_lines):
    failed = False
    job_id = None

    for line in output_lines:
        line_str = line.strip()

        # 匹配 "Number of Transfers Failed: 1" 或 "Failed: 1"
        m_fail = re.search(r'Failed:\s*(\d+)', line_str)
        if m_fail and int(m_fail.group(1)) > 0:
            failed = True

        # 匹配 JobId: xxxxxxxx-xxxx-...
        m_job = re.search(r'[Jj]ob\s*[Ii][Dd]\s*[:\s]*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', line_str)
        if m_job:
            job_id = m_job.group(1)

    return failed, job_id

def _wait_with_countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"\r  ⏳ {i} 秒后重试...", end="", flush=True)
        time.sleep(1)
    print("\r  " + " " * 30 + "\r", end="", flush=True)

# ==========================================
# 模块 3：调用 AzCopy 执行精准传输（单循环扁平化设计）
# ==========================================
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10

def run_azcopy_transfer(azcopy_bin, source_url, destination_path, max_retries=MAX_RETRIES):
    print(f"\n🚀 开始通过 AzCopy 直连拉取至: {destination_path}")
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    current_job_id = None

    for attempt in range(1, max_retries + 1):
        output_lines = []

        # 根据是否有 JobId 决定是 Copy 还是 Resume
        if current_job_id:
            print(f"\n🔄 [第 {attempt}/{max_retries} 次尝试] 启动极速断点续传 (JobId: {current_job_id})")
            command = [azcopy_bin, "jobs", "resume", current_job_id]
        else:
            if attempt == 1:
                print(f"\n📥 [第 {attempt}/{max_retries} 次尝试] 执行初始全量拉取")
                command = [azcopy_bin, "copy", source_url, destination_path, "--recursive=true"]
            else:
                print(f"\n⚠️ [第 {attempt}/{max_retries} 次尝试] 无 JobId，退回智能对比续传模式")
                command = [azcopy_bin, "copy", source_url, destination_path, "--recursive=true", "--overwrite=ifSourceNewer"]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in process.stdout:
            output_lines.append(line)
            if "Done" in line or "Transferred" in line or "Failed" in line or "%" in line or "JobId" in line:
                print(f"    {line.strip()}")

        process.wait()

        failed, new_job_id = parse_transfer_result(output_lines)

        # 更新 JobId，供下一次重试使用
        if new_job_id:
            current_job_id = new_job_id

        if process.returncode == 0 and not failed:
            print(f"✅ 目录 {destination_path} 拉取彻底成功！")
            return True
        else:
            print(f"❌ 尝试结束，存在瑕疵 (退出码: {process.returncode}, JobId: {current_job_id})")
            if attempt < max_retries:
                _wait_with_countdown(RETRY_DELAY_SECONDS)

    print(f"🚨 目录 {destination_path} 在 {max_retries} 次尝试后仍然失败。")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 AzCopy 从 Azure Blob Storage 拉取数据集")
    parser.add_argument("--account", type=str, required=True, help="Azure Storage 账户名称")
    parser.add_argument("--container", type=str, required=True, help="Azure Storage 容器名称")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help=f"最大重试次数 (默认: {MAX_RETRIES})")
    args = parser.parse_args()

    azcopy_bin = install_azcopy()

    base_url = f"https://{args.account}.blob.core.windows.net/{args.container}"

    tasks = [
        {
            "cloud_path": "robot_dataset/lerobot-format-v30/merged_0412_v1/",
            "local_path": "/scratch/amlt_code/lola_lerobot/robot_dataset/lerobot-format-v30/merged_0412_v1/"
        },
    ]

    subprocess.run([azcopy_bin, "login", "--identity"], check=True)

    failed_tasks = []
    for task in tasks:
        src_url = f"{base_url}/{task['cloud_path']}"
        success = run_azcopy_transfer(azcopy_bin, src_url, task["local_path"], max_retries=args.max_retries)
        if not success:
            failed_tasks.append(task)

    if failed_tasks:
        print(f"\n❌ 以下 {len(failed_tasks)} 个任务在所有重试后仍失败：")
        for t in failed_tasks:
            print(f"  - {t['cloud_path']}")
    else:
        print("\n🎉 所有指定数据集已安全、极速地抵达本地存储！")
