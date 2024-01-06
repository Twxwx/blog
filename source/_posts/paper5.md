---
title: Exploiting Unlabeled Data with Vision and Language Models for Object Detection
date: 2023-12-16 17:28:03
categories: 
    - 深度学习
tags: 
    - open vocabulary
---

# Contributions
1. Fusing CLIP scores and objectness scores of the two-stage proposal generator
2. removing redundant proposals by repeated application of the localization head (2nd stage) in the proposal generator
3. We leverage V&L models for improving object detection frameworks by generating pseudo labels on unlabeled data
4. A simple but effective strategy to improve the localization quality of pseudo labels scored with the V&L model CLIP

![Fig 1](/img/paper1/fig1.png)

# Problems
a major challenge of V&L models is the rather low object localization quality

# Method
The goal of our work is to mine unlabeled images with vision & language (V&L) models to generate semantically rich pseudo labels (PLs) in the form of bounding boxes so that object detectors can better leverage unlabeled data

![Fig 2](/img/paper1/fig2.png)