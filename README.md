# 🧠 DeepInsights: Advanced NLP & Vision Projects

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![DeepSeek](https://img.shields.io/badge/DeepSeek-R1_8B-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-green)
![ViT](https://img.shields.io/badge/ViT-Vision_Transformer-purple)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-CarRacing-green)

## 📋 Overview

This repository showcases cutting-edge implementations of advanced machine learning techniques in Natural Language Processing, Computer Vision, and Reinforcement Learning. These projects leverage state-of-the-art architectures to solve complex problems in medical and autonomous systems domains.

## 🚀 Projects

### 1️⃣ NLP Project: Medical Question Answering

Fine-tuning of DeepSeek R1 (8B parameters) for medical reasoning:

- **LoRA Fine-tuning**: Parameter-efficient adaptation of the 8B parameter model for medical question answering with chain-of-thought reasoning
- **RAG Implementation**: Retrieval-Augmented Generation system using FAISS vector store for enhanced factual accuracy
- **Dataset**: Trained on specialized medical datasets with expert reasoning patterns
- **Performance**: Achieves significant improvements in reasoning quality and factual accuracy compared to base model

### 2️⃣ Computer Vision Project: COVID-19 Detection

Vision Transformer (ViT) implementation for COVID-19 detection from ultrasound images:

- **Architecture**: Implementation of Vision Transformer with patch-based image processing
- **Transfer Learning**: Fine-tuning pre-trained ViT models for medical imaging
- **Data Processing**: Advanced video frame extraction and augmentation techniques
- **Evaluation**: Comprehensive visualization and analysis of model predictions

### 3️⃣ Reinforcement Learning Project: Autonomous Car Racing

Comparative study of DQN and A2C algorithms in the CarRacing-v3 environment:

- **Deep Q-Network (DQN)**: Implementation of value-based RL with experience replay and target networks
- **Advantage Actor-Critic (A2C)**: Policy gradient approach for continuous control of racing vehicles
- **Environment**: Gymnasium's CarRacing-v3 with frame stacking and preprocessing
- **Analysis**: Comprehensive comparison of performance, training stability, and computational requirements

## 🛠️ Technologies

- **Deep Learning**: PyTorch, Transformers, TensorFlow
- **Efficient Fine-tuning**: Unsloth, LoRA (Low-Rank Adaptation)
- **Vector Search**: FAISS
- **RAG Pipeline**: LangChain
- **Experiment Tracking**: Weights & Biases
- **Vision Models**: Timm library
- **Reinforcement Learning**: Gymnasium, OpenAI environments

## 📊 Results

All projects demonstrate the power of advanced architectures when applied to specialized domains:

- The NLP model shows enhanced reasoning capabilities for complex medical questions
- The Vision Transformer achieves high accuracy in COVID-19 detection from ultrasound images
- The RL implementations provide insights into the trade-offs between value-based and policy-based approaches for autonomous racing

## 🔍 Getting Started

Each project directory contains detailed Jupyter notebooks and documentation:

- `NLP/final_version_fine-tuned.ipynb`: Complete LoRA fine-tuning pipeline
- `NLP/RAG_DEEPSEEKR1_8B.ipynb`: RAG implementation with FAISS
- `CV/Vit-Covid-Prediction.ipynb`: Vision Transformer implementation for COVID detection
- `RL/a2c.ipynb`: Implementation of the Advantage Actor-Critic algorithm
- `RL/dqn.ipynb`: Implementation of the Deep Q-Network algorithm
- `RL/main.tex`: Research paper comparing DQN and A2C performance

## 📚 Requirements

See requirements.txt for a complete list of dependencies.

## 🔗 References

- [DeepSeek R1 Model](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8b)
- [Vision Transformers Paper](https://arxiv.org/abs/2010.11929)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Deep Q-Network](https://www.nature.com/articles/nature14236)
- [Advantage Actor-Critic](https://arxiv.org/abs/1602.01783)
