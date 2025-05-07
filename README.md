# 🧠 DeepMind: Advanced NLP & Vision Projects

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![DeepSeek](https://img.shields.io/badge/DeepSeek-R1_8B-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-green)
![ViT](https://img.shields.io/badge/ViT-Vision_Transformer-purple)

## 📋 Overview

This repository showcases cutting-edge implementations of advanced machine learning techniques in Natural Language Processing and Computer Vision. Both projects leverage state-of-the-art transformer architectures to solve complex problems in the medical domain.

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

## 🛠️ Technologies

- **Deep Learning**: PyTorch, Transformers
- **Efficient Fine-tuning**: Unsloth, LoRA (Low-Rank Adaptation)
- **Vector Search**: FAISS
- **RAG Pipeline**: LangChain
- **Experiment Tracking**: Weights & Biases
- **Vision Models**: Timm library

## 📊 Results

Both projects demonstrate the power of transformer architectures when applied to specialized domains:

- The NLP model shows enhanced reasoning capabilities for complex medical questions
- The Vision Transformer achieves high accuracy in COVID-19 detection from ultrasound images

## 🔍 Getting Started

Each project directory contains detailed Jupyter notebooks with step-by-step implementations:

- `NLP/final_version_fine-tuned.ipynb`: Complete LoRA fine-tuning pipeline
- `NLP/RAG_DEEPSEEKR1_8B.ipynb`: RAG implementation with FAISS
- `CV/Vit-Covid-Prediction.ipynb`: Vision Transformer implementation for COVID detection

## 📚 Requirements

See requirements.txt for a complete list of dependencies.

## 🔗 References

- [DeepSeek R1 Model](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8b)
- [Vision Transformers Paper](https://arxiv.org/abs/2010.11929)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
