# Natural Language Processing with Hugging Face Transformers  
<p align="center">Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">  
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

---

## 👤 Author
**Name:** Arifian Saputra

---

## 📌 Project Overview

This repository contains a collection of practical exercises using Hugging Face Transformers to perform various NLP tasks such as:

- Sentiment Analysis
- Zero-shot Topic Classification
- Text Generation
- Masked Language Modeling
- Named Entity Recognition (NER)
- Question Answering

Each task includes both example runs and personal custom inputs, followed by corresponding analysis.

---

## ✅ Example & To-Do Analyses

### 🔹 Example 1 - Sentiment Analysis
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I am doing this on a regular basis, baking a cake in the morning!")
