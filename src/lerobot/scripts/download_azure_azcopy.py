import os
import shutil
import subprocess
import urllib.request
import tarfile
import argparse

# 🌟 强制设置 AzCopy 使用 Managed Identity 身份验证
os.environ["AZCOPY_AUTO_LOGIN_TYPE"] = "MSI"

# ==========================================
# 模块 1：环境准备与 AzCopy 安装
# ==========================================
def install_azcopy():
    """在当前目录自动下载并解压 Linux 版 azcopy"""
    azcopy_path = "./azcopy"
    if os.path.exists(azcopy_path):
        return azcopy_path

    print("正在下载官方版 AzCopy 工具...")
    tar_url = "https://aka.ms/downloadazcopy-v10-linux"
    tar_filename = "azcopy_linux.tar.gz"
    
    urllib.request.urlretrieve(tar_url, tar_filename)
    
    print("解压 AzCopy...")
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
# 模块 2：调用 AzCopy 执行精准传输
# ==========================================
def run_azcopy_transfer(azcopy_bin, source_url, destination_path):
    print(f"\n🚀 开始通过 AzCopy 直连拉取至: {destination_path}")
    
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    command = [
        azcopy_bin, "copy", 
        source_url, 
        destination_path, 
        "--recursive=true"
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        if "Done" in line or "Transferred" in line or "Failed" in line or "%" in line:
            print(line.strip())
            
    process.wait()
    
    if process.returncode == 0:
        print(f"✅ 目录 {destination_path} 拉取成功！")
    else:
        print(f"❌ 目录 {destination_path} 拉取失败，请检查 Azure CLI 是否已登录 (az login) 以及当前账号是否有 Blob 数据读取权限。")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用 AzCopy 从 Azure Blob Storage 拉取数据集")
    parser.add_argument("--account", type=str, required=True, help="Azure Storage 账户名称 (Account Name)")
    parser.add_argument("--container", type=str, required=True, help="Azure Storage 容器名称 (Container Name)")
    args = parser.parse_args()

    azcopy_bin = install_azcopy()
    
    # 使用传入的参数构建基础 URL
    base_url = f"https://{args.account}.blob.core.windows.net/{args.container}"

    # 您需要精准提取的三个文件夹的相对路径，以及对应的本地目标路径
    tasks = [
        {
            "cloud_path": "robot_dataset/lerobot-format-v30/merged_0412_v1/",
            "local_path": "/scratch/amlt_code/lola_lerobot/robot_dataset/lerobot-format-v30/merged_0412_v1/"
        },
    ]
    
    # 进行登录操作
    subprocess.run([azcopy_bin, "login", "--identity"], check=True)

    for task in tasks:
        # 直接拼接 URL
        src_url = f"{base_url}/{task['cloud_path']}"
        run_azcopy_transfer(azcopy_bin, src_url, task["local_path"])
        
    print("\n所有指定数据集已安全、极速地抵达本地存储！")