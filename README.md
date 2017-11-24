# Basic Tensorflow Usage
## CUDA
We need CUDA to run Tensorflow with GPU<br>
Nvidia CUDA website: https://developer.nvidia.com/cuda-downloads<br>
cudnn library download: https://developer.nvidia.com/cudnn
## Anaconda
### anaconda install
Anaconda website: https://www.anaconda.com/<br>
Download Python 3.6 linux version, and run command.
```
sh ANACONDA_FILE
```
Using command ```which python``` or ```which python3``` to check using Anaconda python or not.
### using virtual environment
create virtual environment
```
conda create -n ENV_NAME python=3.6 anaconda
```
start virtual environment
```
source activate ENV_NAME
```
end vurtual environment
```
source deactivate
```
## Install tensorflow
### GPU version
```
conda install tensorflow-gpu
```
## Display tensorflow data
### Tensorboard
Run command, and open url ```YOUR_IP:6006``` to display
```
tensorboard --logdir . --host YOUR_IP --inspect tf_NUM.py
```
