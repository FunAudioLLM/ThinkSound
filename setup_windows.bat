@echo off
echo Creating conda environment...
call conda create -n thinksound python=3.10 -y
call activate base
call conda activate thinksound

echo Installing requirements...
call pip install thinksound
call conda install -y -c conda-forge 'ffmpeg<7'

echo Installing Git LFS...
call git lfs install

echo Cloning pretrained models...
call git clone https://huggingface.co/liuhuadai/ThinkSound ckpts

echo Setup complete! You can now run demo.bat
pause
