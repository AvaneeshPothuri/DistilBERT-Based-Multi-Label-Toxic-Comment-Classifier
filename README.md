# DistilBERT-Based-Multi-Label-Toxic-Comment-Classifier

Detecting multiple types of toxicity in online comments using state-of-the-art NLP and experiment tracking.

---

## Project Overview

This project fine-tunes the DistilBERT transformer model to perform **multi-label classification** on the [Kaggle Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) dataset. The goal is to automatically identify six types of toxicity in online comments, supporting safer digital communities.

---

## Problem Statement

Online platforms face challenges moderating harmful content, where comments can be toxic in multiple, overlapping ways. This project builds a robust, multi-label classifier to identify several forms of toxicity in a single comment.

---

## Dataset

- **Source:** Kaggle Jigsaw Toxic Comment Classification Challenge
- **Subset Used:** 5,000 randomly sampled comments (for rapid prototyping)
- **Labels:**  
  - `toxic`
  - `severe_toxic`
  - `obscene`
  - `threat`
  - `insult`
  - `identity_hate`

---

## Approach

### 1. Data Preparation
- Loaded and inspected the dataset with pandas.
- Selected a subset (5,000 samples) for efficient experimentation.
- Tokenized comment texts using DistilBERT’s tokenizer.

### 2. Model Fine-Tuning
- Used `distilbert-base-uncased` from Hugging Face Transformers.
- Adapted for multi-label classification (`num_labels=6`, sigmoid output).
- Trained for 1 epoch with batch size 4 (optimized for Colab T4 GPU).
- Loss function: Binary Cross-Entropy with Logits (BCEWithLogitsLoss).

### 3. Experiment Tracking
- Integrated [Weights & Biases (wandb.ai)](https://wandb.ai) for real-time monitoring, metric visualization, and reproducibility.

### 4. Evaluation
- Assessed model with mean ROC-AUC and validation loss.

---

## Results

- **Validation ROC-AUC:** `0.9486`
- **Validation Loss:** `0.0537`
- **Epochs:** 1 (prototype run)

These results demonstrate strong generalization and effective multi-label classification, even with a small data subset and limited training.
