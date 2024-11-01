# 🤣👉Sarva👈
<p align="center">
  <img src="./figs/introduction1.png"> 
</p>

This repository is the official codebase of "**Sarva: Multimodal Sarcasm Detection via Multimodal Large Language Models Base on Bootstrapping Sarcastic Reasoning and Reentry Augmentation**"
</br>
In the following, we will guide you how to use this repository step by step.
## 🤣👌Introduction
To the best of our knowledge, we are the first to explore the sentiment reasoning capability of Large Vision-Language Models (LVMs) for the task of multimodal sarcasm detection, addressing the common challenges faced by current multimodal sarcasm detection models through MLLMs.
<p align="center">
  <img src="./figs/introduction2.png" width="700 height="700">
</p>

## 🤓 Overview
<p align="center">
  <img src="./figs/framework.png" width="700 height="700">
</p>

## 😅 Sentiment Instruction and Augmented Samples
<p align="center">
  <img src="./figs/reason_way.png" width="400" height="300">
  <img src="./figs/augmentation.png" width="400" height="300">
</p>

To better aid MLLMs in performing complex emotional reasoning for understanding sarcastic semantics, we have designed three novel guidance strategies: **Chain of Emotional Contradiction (CoEC)**, **Graph of Emotional Contradiction (GoEC)**, and **Bagging of Emotional Contradiction (BoEC)**. These strategies are designed to prompt MLLMs to detect sarcasm in humans by incorporating both sequential and non-sequential prompting approaches. To further analyze the samples enhanced by the Reentry Augmentation module, we conduct a qualitative analysis of the generated augmented samples. Specifically, we select four examples to present both their original and augmented forms.

## 🤡 Quickstart

### Datasets Preparation
MMSD dataset--[link](https://github.com/soujanyaporia/MUStARD)	 </br>
MMSD2.0 dataset--[link](https://github.com/JoeYing1019/MMSD2.0.git) </br>
MultiBully dataset--[link](https://github.com/ROC-HCI/UR-FUNNY?tab=readme-ov-file) </br>

