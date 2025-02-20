# -*- coding: utf-8 -*-
import os
import subprocess
import sys

# 定义所有可用的R1模型
AVAILABLE_MODELS = {
    "1": "DeepSeek-R1-Distill-Qwen-1.5B",
    "2": "DeepSeek-R1-Distill-Llama-8B", 
    "3": "DeepSeek-R1-Distill-Qwen-14B",
    "4": "DeepSeek-R1-Distill-Qwen-32B",
    "5": "DeepSeek-R1-Distill-Llama-70B",
    "6": "DeepSeek-R1"
}

def print_available_models():
    """打印所有可用的模型列表"""
    print("\n=== 可用的DeepSeek R1模型 ===")
    for key, model in AVAILABLE_MODELS.items():
        print(f"{key}. {model}")
    print("========================")

def get_model_choice():
    """获取用户选择的模型"""
    while True:
        choice = input("\n请输入要下载的模型编号(1-6): ")
        if choice in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[choice]
        print("无效的选择,请重新输入!")

def download_model(model_name):
    """下载指定的模型"""
    # 创建保存模型的目录
    base_dir = "/root/autodl-tmp"
    model_dir = os.path.join(base_dir, model_name)
    
    # 构建下载命令
    cmd = [
        "modelscope", "download",
        "--model", f"deepseek-ai/{model_name}",
        "--local_dir", model_dir
    ]
    
    print(f"\n开始下载模型 {model_name}")
    print(f"模型将保存到: {model_dir}")
    print("\n下载过程中请耐心等待...\n")
    
    try:
        # 执行下载命令
        subprocess.run(cmd, check=True)
        print(f"\n模型 {model_name} 下载成功!")
        print(f"模型文件位置: {model_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n错误: 模型下载失败")
        print(f"错误信息: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生未知错误: {str(e)}")
        sys.exit(1)

def main():
    """主函数"""
    print("\n欢迎使用 DeepSeek R1 模型下载器!")
    
    while True:
        print_available_models()
        model_name = get_model_choice()
        
        # 确认下载
        confirm = input(f"\n您选择下载的模型是: {model_name}\n确认下载吗? (y/n): ")
        if confirm.lower() == 'y':
            download_model(model_name)
            break
        else:
            print("\n已取消下载,请重新选择")

if __name__ == "__main__":
    main()