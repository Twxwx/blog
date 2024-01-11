---
title: CORA Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching
date: 2024-01-06 23:11:06
categories:
tags:
---

# Obstacles
1. the distribution mismatch that happens when applying a VL-model trained on whole images to region recognition tasks.
2. the difficulty of localizing objects of unseen classes.

# Contributions
1. we propose CORA, a DETR-style framework that adapts CLIP for Open-vocabulary detection by Region prompting and Anchor pre-matching. 
2. Region prompting mitigates the whole-to-region distribution gap by prompting the region features of the CLIP-based region classifier. 
3. Anchor pre-matching helps learning generalizable object localization by a class-aware matching mechanism.