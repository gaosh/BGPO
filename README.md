# Bregman Gradient Policy Optimization
Authors: Feihu Huang*, Shangqian Gao* and Huang Heng (* indicates equal contribuation)

PyTorch Implementation of [Bregman Gradient Policy Optimization](https://openreview.net/pdf?id=ZU-zFnTum1N) (ICLR 2022).

# Requirements
pytorch 1.1.0 or higher\
[garage 2021.03](https://github.com/rlworkgroup/garage/tree/release-2021.03)\
[mujuco](http://www.mujoco.org/)  
[gym](https://github.com/openai/gym)  
If you do not install mujuco, then only CartPole environment is available.

# Usage
To run BGPO on Pendulum-v2
```
python BGPO_test.py --env Pendulum --type Diag
```
To run VR-BGPO on Pendulum-v2
```
python VR_BPO_test.py --env Pendulum --type Diag
```
To run different environments change --env to one of the followings: "Pendulum", "DPendulum", "Reacher", "Walker", "Swim" or "HalfCheetah". If you want to use our algorithms on other enviroments, you need to implement it by yourself, but it should be pretty straightforward.

This codebase is based on the implementation of our [MBPG](https://github.com/gaosh/MBPG) (Momentum-Based Policy Gradient Methods) work.

# Citation
If you find this repository is helpful to your work, considering to cite our BGPO and MBPG papers.
```
@InProceedings{huang2020accelerated,
  author    = {Huang, Feihu and Gao, Shangqian and Pei, Jian and Huang, Heng},
  title     = {Momentum-Based Policy Gradient Methods},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  year      = {2020},}
```
```
@inproceedings{huang2022bregman,
  author    = {Feihu Huang and Shangqian Gao and Heng Huang},
  title     = {Bregman Gradient Policy Optimization},
  booktitle = {International Conference on Learning Representations},
  year      = {2022},
  url       = {https://openreview.net/forum?id=ZU-zFnTum1N}}
```
