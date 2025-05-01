# Mental Health Status Classification Using BERT and Deep Learning

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

A deep learning NLP system to classify textual statements into **7 mental health categories** (anxiety, normal, depression, suicidal, stress, bipolar, personality disorder) using BERT embeddings. Achieves **82% accuracy** and addresses class imbalance via SMOTE-inspired techniques.

![Word Cloud Example](https://via.placeholder.com/600x200?text=Word+Clouds+for+Mental+Health+Conditions)

## 📌 Overview
- **Problem**: Mental health disorders affect 970 million people globally, yet early detection remains challenging.  
- **Solution**: A BERT-based model to analyze textual statements for early intervention.  
- **Impact**: Detects linguistic cues (e.g., suicidal ideation markers) and integrates with telehealth platforms.  
- **Key Innovation**: Hybrid pipeline combining BERT embeddings, manual feature engineering, and LDA topic modeling.

## 🚀 Key Features
- **BERT Embeddings**: Contextual feature extraction using `bert-base-uncased`.  
- **Class Imbalance Handling**: SMOTE-inspired duplication and class-weighted loss.  
- **Interpretability**: LDA topic modeling to decode condition-specific themes.  
- **Scalability**: Designed for deployment via APIs or mobile apps.

## 📊 Results
| Model               | Accuracy | F1-Score (Suicidal) |
|---------------------|----------|---------------------|
| TF-IDF + SVM        | 72%      | 0.58                |
| LSTM                | 78%      | 0.63                |
| **BERT (Ours)**     | **82%**  | **0.67**            |

**Classification Report (Top Classes)**:
| Class                | Precision | Recall | F1-Score |
|----------------------|-----------|--------|----------|
| Normal               | 0.96      | 0.94   | 0.95     |
| Anxiety              | 0.89      | 0.92   | 0.91     |
| Personality Disorder | 0.81      | 0.78   | 0.79     |

