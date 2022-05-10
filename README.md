# Domain Adaptation with Multi Task Features for Bird’s Eye View Segmentation

# Team 10 - **Vanshaj Chowdhary (vanshajc), Punit Bhatt (punitb), Adithya Sampath (adithyas)**

# Webpage: [https://16824-team10.notion.site/689d6c2bdad341939b6461f55d7887d2](https://www.notion.so/Domain-Adaptation-with-Multi-Task-Features-for-Bird-s-Eye-View-Segmentation-689d6c2bdad341939b6461f55d7887d2)

# Motivation

Bird’s Eye View (BEV) segmentation has become a popular research focus especially for self-driving and drone applications. The top-down grid-based representation is both convenient and efficient to use for motion planning and prediction components for navigation systems. Techniques like inverse perspective mapping have been applied to already existing semantic segmentation networks to warp a single view camera to a top-down view. However, these yield undesirable artifacts around vehicles and other objects [1]. While there has been significant progress in semantic segmentation, BEV segmentation encounters unique challenges that make the task particularly difficult. Namely, there is a significant lack of real-world bird’s eye view segmentation labeled ground truth. Typical segmentation models are trained in a supervised learning manner, where images are manually annotated with the appropriate class at a pixel level. Compared to semantic segmentation which has dozens of popular datasets and benchmarks, getting ground truth BEV segmentation labeled data, especially for an autonomous driving use case is very expensive and challenging. Instead, the community has relied on simulators to generate synthetic datasets and attempt to transfer from simulation to real-world scenarios using domain adaptation. 

We propose a solution that also relies on simulation to real-world transfer, but also transfer tasks in addition. Specifically, we force our model to learn features for semantic segmentation and BEV segmentation and try to learn a mapping from front-view to BEV in the simulation domain. Then, we apply such mapping to the real world domain where there are plenty of already semantic segmentation models achieving state-of-the-art results. In conclusion, the goal of this project is to generate Bird’s Eye View segmentation from a single front camera without any real labeled ground truth BEV labels by utilizing domain adaptation and task transfer learning.


# Dataset

Follow the steps in the below repositories to set up nuScenes and Carla datasets.

- [NuScenes Dataset](https://github.com/CMU-punit-bhatt/datasets)
- [Carla Dataset](https://github.com/carla-simulator/carla)

# Training N1:

1. Update `train_front.yaml` config with path to nuScenes and Carla front camera datasets.
2. Run the train script below to train N1

```
python train_n1.py
```

# Training N2:

1. Update `train_bev.yaml` config with path to Carla front camera and BEV datasets.
2. Run the train script below to train N2

```
python train_n2.py
```

# Training G:

1. Update `train_transfer.yaml` config with path to Carla front camera and BEV datasets.
2. Run the train script below to train G

```
python train_G.py
```

# Evaluation

1. Update `eval_adaptive.yaml` config with path to Carla front camera and BEV datasets.
2. Run the below command to evaluate the adaptive net

```
python eval_adaptive.py
```

# Credits:

- Code inspired from the [unofficial pytorch implementation](https://github.com/adricarda/AT-DT-Pytorch-implementation)

# Citation

This repository contains the unoofficial pytorch implementation of AT/DT, proposed in the paper "Learning Across Tasks and Domains", ICCV 2019.

```
@inproceedings{ramirez2019,
  title     = {Learning Across and Domains},
  author    = {Zama Ramirez, Pierluigi and
                Tonioni, Alessio and
                Salti, Samuele and
                Di Stefano, Luigi},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year = {2019}
}
```
