"""
DIB-FNO 快速运行脚本
用法:
    python run.py                  # 快速验证模式
    python run.py --mode probe     # 探测 GRIB 文件
    python run.py --mode full      # 完整流程 (GRIB 数据)
    python run.py --mode train     # 仅训练
    python run.py --mode eval      # 仅评估
    python run.py --config quick   # 快速配置
    python run.py --config paper   # 论文级配置
    python run.py --config grib    # GRIB 数据配置
    python run.py --device cuda    # 强制使用 GPU
    python run.py --epochs 100     # 覆盖训练轮数
    python run.py --compile        # 使用 torch.compile 加速
"""

import subprocess
import sys
import os


def main():
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 构建命令
    cmd = [sys.executable, "main.py"] + sys.argv[1:]

    # 如果没有参数，默认快速验证
    if len(sys.argv) == 1:
        cmd.append("--mode")
        cmd.append("quick")

    print(f"运行: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()