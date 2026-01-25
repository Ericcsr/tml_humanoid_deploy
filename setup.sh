conda create -n bm python=3.10.18
conda activate bm
pip install mujoco joblib onnxruntime pybullet scipy torch PyYAML opencv-python redis
sudo apt install redis-server
cd ..
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e . --no-deps
cd ../tml_humanoid_deploy
