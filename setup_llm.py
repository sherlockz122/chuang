# 可选组件: 下载 Qwen2.5-0.5B-Instruct 权重（LLM 定稿通道，~1GB）
# 不下载也能运行演示（前端"LLM 定稿通道"开关关掉，走 MLP 兜底）
# 用法: python setup_llm.py [hf|ms]   (hf=HuggingFace, ms=ModelScope, 默认 ms)
import sys, os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, 'models', 'qwen2.5-0.5b-instruct')

def main():
    src = (sys.argv[1] if len(sys.argv) > 1 else 'ms').lower()
    if os.path.exists(os.path.join(DST, 'model.safetensors')):
        print('已存在 model.safetensors，跳过（如需重下请删除该目录）')
        return
    os.makedirs(DST, exist_ok=True)
    if src == 'hf':
        cmd = [sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub']
        subprocess.run(cmd, check=True)
        from huggingface_hub import snapshot_download
        snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir=DST)
    else:
        cmd = [sys.executable, '-m', 'pip', 'install', '-q', 'modelscope']
        subprocess.run(cmd, check=True)
        from modelscope import snapshot_download
        snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir=DST)
    print(f'权重已就绪: {DST}')

if __name__ == '__main__':
    main()
