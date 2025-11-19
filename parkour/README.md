# Isaaclab_Parkour

Isaaclab based Parkour locomotion 

Base model: [Extreme-Parkour](https://extreme-parkour.github.io/)

https://github.com/user-attachments/assets/aa9f7ece-83c1-404f-be50-6ae6a3ba3530


## How to install 

```
cd IsaacLab ## going to IsaacLab
```

```
https://github.com/CAI23sbP/Isaaclab_Parkour.git ## cloning this repo
```

```
cd Isaaclab_Parkour && pip3 install -e .
```

```
cd parkour_tasks && pip3 install -e .
```

## How to train policies

### 1.1. Training Teacher Policy

```
# Update to 12288 envs - twice the default for faster training
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless --num_envs 12288

# To restart a training from a checkpoint
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --resume --load_run "2025-11-13_12-15-20" --num_envs 12288 --headless
```

Training the teacher policy is simplified using the Makefile targets

```
make train-teacher

# To restart from a checkpoint
make train-teacher-resume ARGS="--load_run 2025-11-13_12-15-20"
```

NOTE: Expected duration ~6 hours for 7400 iterations.


### 1.2. Training Student Policy

To run opencv during Student training, otherwise set `debug_vis`=False
```
pip install --upgrade opencv-python
pip uninstall numpy==2.3.0
pip install numpy==1.26.0
```

Revert the following commit in the IsaacLab repo.  This removes the concatenation feature in ObservationManager -- Adds option to define the concatenation dimension in the `ObservationManager` and change counter update in `CommandManager` (#2393)
```
git revert be41bb0d0436acaec36daf210fd80d24ff76ea7e

# and fix the merge conflicts...
```

This revert in IsaabLab fixes the following error during student trainings
```
Error executing job with overrides: []
Traceback (most recent call last):
  File "/home/brian/patina/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 101, in hydra_main
    func(env_cfg, agent_cfg, *args, **kwargs)
  File "/home/brian/patina/krabby-research/parkour/scripts/rsl_rl/train.py", line 146, in main
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/brian/patina/krabby-research/parkour/env/lib/python3.11/site-packages/gymnasium/envs/registration.py", line 734, in make
    env = env_creator(**env_spec_kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/brian/patina/krabby-research/parkour/parkour_isaaclab/envs/parkour_manager_based_rl_env.py", line 32, in __init__
    super().__init__(cfg=cfg)
  File "/home/brian/patina/krabby-research/parkour/parkour_isaaclab/envs/parkour_manager_based_env.py", line 112, in __init__
    self.load_managers()
  File "/home/brian/patina/krabby-research/parkour/parkour_isaaclab/envs/parkour_manager_based_rl_env.py", line 52, in load_managers
    super().load_managers()
  File "/home/brian/patina/krabby-research/parkour/parkour_isaaclab/envs/parkour_manager_based_env.py", line 143, in load_managers
    self.observation_manager = ObservationManager(self.cfg.observations, self)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/brian/patina/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py", line 97, in __init__
    dim_sum = torch.sum(term_dims[:, dim], dim=0)
                        ~~~~~~~~~^^^^^^^^
IndexError: index -1 is out of bounds for dimension 1 with size 0
```

Train the student 

```
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless
```


Training the student policy is simplified using the Makefile targets

```
make train-student

# To restart from a checkpoint
make train-student-resume ARGS="--load_run 2025-11-13_12-15-20"
```


NOTE: Expected duration ~24 hours for 38,000 iterations.


## How to Visualize the Training Parameters using TensorBoard

### Install TensorBoard with pip
```
pip install tensorboard
```

### Run TensorBoard using a Training Run
```
tensorboard --logdir=./logs/rsl_rl/unitree_go2_parkour_teacher/2025-11-13_21-26-55

# In another terminal window or navigate to 
open http://localhost:6006/
```


## How to play your policy 

### 2.1. Pretrained Teacher Policy 

Download Teacher Policy by this [link](https://drive.google.com/file/d/1JtGzwkBixDHUWD_npz2Codc82tsaec_w/view?usp=sharing)


### 2.2. Playing Teacher Policy 

```
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 16
```

[Screencast from 2025년 08월 16일 12시 43분 38초.webm](https://github.com/user-attachments/assets/ff1f58db-2439-449c-b596-5a047c526f1f)


### 2.3. Evaluation Teacher Policy

```
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0 
```

### 3.1 Pretrained Student Policy 

Download Student Policy by this [link](https://drive.google.com/file/d/1qter_3JZgbBcpUnTmTrexKnle7sUpDVe/view?usp=sharing)

### 3.2. Playing Student Policy 

```
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16
```

https://github.com/user-attachments/assets/82a5cecb-ffbf-4a46-8504-79188a147c40


### 3.3. Evaluation Student Policy

```
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 
```

e.g.
```
Mean reward: 20.00$\pm$7.71
Mean episode length: 929.70$\pm$278.06
Mean number of waypoints: 0.92$\pm$0.19
Mean edge violation: 0.21$\pm$0.49
```


## How to deploy in IsaacLab

[Screencast from 2025년 08월 20일 18시 55분 01초.webm](https://github.com/user-attachments/assets/4fb1ba4b-1780-49b0-a739-bff0b95d9b66)

### 4.1. Deployment Teacher Policy 

```
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 
```


### 4.2. Deployment Student Policy 

```
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 
```

## Testing your modules

```
cd parkour_test/ ## You can test your modules in here
```

## Visualize Control (ParkourViewportCameraController)

```
press 1 or 2: Going to environment

press 8: camera forward    

press 4: camera leftward   

press 6: camera rightward   

press 5: camera backward

press 0: Use free camera (can use mouse)

press 1: Not use free camera (default)
```


## How to Deploy sim2sim or sim2real

it is a future work, i will open this repo as soon as possible

* [x] sim2sim: isaaclab to mujoco

* [ ] sim2real: isaaclab to real world

see this [repo](https://github.com/CAI23sbP/go2_parkour_deploy)


### TODO list

* [x] Opening code for training Teacher model  

* [x] Opening code for training Distillation 

* [x] Opening code for deploying policy in IsaacLab by demo: code refer [site](https://isaac-sim.github.io/IsaacLab/main/source/overview/showroom.html)  

* [x] Opening code for deploying policy by sim2sim (mujoco)

* [ ] Opening code for deploying policy in real world 

## Citation

If you use this code for your research, you **must** cite the following paper:

```
@article{cheng2023parkour,
title={Extreme Parkour with Legged Robots},
author={Cheng, Xuxin and Shi, Kexin and Agarwal, Ananye and Pathak, Deepak},
journal={arXiv preprint arXiv:2309.14341},
year={2023}
}
```

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

```
Copyright (c) 2025, Sangbaek Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software …

The use of this software in academic or scientific publications requires
explicit citation of the following repository:

https://github.com/CAI23sbP/Isaaclab_Parkour
```

## contact us 

```
sbp0783@hanyang.ac.kr
```
