git clone https://github.com/google-deepmind/videoprism.git
cd videoprism
pip install .
cd ..
pip install -r scripts/PrismAudio/setup/requirements.txt
pip install tensorflow-cpu==2.15.0
pip install facenet_pytorch==2.6.0 --no-deps

conda install -y -c conda-forge 'ffmpeg<7'
