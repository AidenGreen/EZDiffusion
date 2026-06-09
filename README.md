# EZDiffusion
一个基于diffuser的简单Diffusion调用和封装框架，用于学习原理和简单实验

使用方法：
1. git同步到本地
2. 使用VS Code打开工程

安装pytorch和其他依赖项

F:
cd F:\Python_Projects\EZDiffusion
conda activate .\_env

python -m pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r _requirements.txt
   
4. 打开data文件夹，运行cifar_download.py下载数据集
5. 打开config.py进行配置
6. 运行train.py进行训练
7. 运行test.py进行采样
