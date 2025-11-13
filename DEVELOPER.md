### Install CUDA on Ubuntu 24.04
The recommended configuration using a RTX 5080 on Ubuntu 24.04
* Kernel: 6.14.0
* Nvidia Driver: `nvidia-driver-580-open`
* Cuda Toolkit 13.0: `cuda-toolkit-13-0`


### Install Python3.11 on Ubuntu 24.04

Python3.11 is the recommended version for IsaacLab, but Python3.12 comes standard with Ubuntu.  These steps help to install Python with venv.
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

### Create a venv and install PyTorch

```
python3.11 -m venv testenv
source testenv/bin/activate
pip3 install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu130
```

### Install IsaacSim

```
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### Install Isaaclab

```
git clone git@github.com:isaac-sim/IsaacLab.git
./isaaclab.sh --install
```

