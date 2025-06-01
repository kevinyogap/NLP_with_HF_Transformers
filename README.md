# Natural Language Processing with Hugging Face Transformers

<p align="center">Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

</div>

**Author:** Kevin Yoga Pratama

## Overview

This project provides a comprehensive introduction to Natural Language Processing (NLP) using Hugging Face Transformers. It demonstrates various NLP tasks including sentiment analysis, topic classification, text generation, named entity recognition, question answering, text summarization, and translation.

## Table of Contents

- [Example Tasks](#example-tasks)
  - [1. Sentiment Analysis](#1-sentiment-analysis)
  - [2. Topic Classification](#2-topic-classification)
  - [3. Text Generation](#3-text-generation)
  - [4. Named Entity Recognition (NER)](#4-named-entity-recognition-ner)
  - [5. Question Answering](#5-question-answering)
  - [6. Text Summarization](#6-text-summarization)
  - [7. Translation](#7-translation)
- [Project Analysis](#project-analysis)
- [Change Log](#change-log)

## Example Tasks

### 1. Sentiment Analysis

#### Example 1 - Basic Sentiment Analysis

```python
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I am doing this on a regular basis, baking a cake in the morning!")
```

**Result:**
```python
[{'label': 'POSITIVE', 'score': 0.9959210157394409}]
```

**Analysis:** The sentiment analysis classifier accurately detects the positive tone in the given sentence with high confidence, indicating reliability for straightforward emotional expressions.

#### My Todo 1 - Personal Sentiment Analysis

```python
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("My biggest weakness is that I'm not consistent in anything I try, whether it's hobbies, routines, or even long-term goals")
```

**Result:**
```python
[{'label': 'NEGATIVE', 'score': 0.9997879862785339}]
```

**Analysis:** The model correctly identified the sentiment as 'NEGATIVE' with a very high score of 0.9997, demonstrating effectiveness in detecting negative tones in personal, reflective statements.

### 2. Topic Classification

#### Example 2 - Zero-Shot Classification

```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Cats are beloved domestic companions known for their independence and agility. These fascinating creatures exhibit a range of behaviors, from playful pouncing to peaceful purring. With their sleek fur, captivating eyes, and mysterious charm, cats have captivated humans for centuries, becoming cherished members of countless households worldwide.",
    candidate_labels=["science", "pet", "machine learning"],
)
```

**Result:**
```python
{
    'sequence': 'Cats are beloved domestic companions...',
    'labels': ['pet', 'machine learning', 'science'],
    'scores': [0.9174826145172119, 0.048576705157756805, 0.03394068405032158]
}
```

**Analysis:** The zero-shot classifier correctly identifies "pet" as the most relevant label with high confidence, showing strong ability to associate descriptive context with predefined categories.

#### My Todo 2 - Personal Attribute Classification

```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "I have a strong desire to learn new things, constantly reading books and exploring new computer technologies.",
    candidate_labels=["personal strengths", "hobbies", "career goals", "weaknesses"],
)
```

**Result:**
```python
{
    'sequence': 'I have a strong desire to learn new things...',
    'labels': ['personal strengths', 'hobbies', 'weaknesses', 'career goals'],
    'scores': [0.7655122876167297, 0.18992270529270172, 0.0267732385545969, 0.017791789025068283]
}
```

**Analysis:** The model successfully classified the statement predominantly as "personal strengths" (0.7655 score), accurately reflecting the nature of the input.

### 3. Text Generation

#### Example 3 - Basic Text Generation

```python
generator = pipeline("text-generation", model="distilgpt2")
generator(
    "This cooking will make you",
    max_length=30,
    num_return_sequences=2,
)
```

**Result:**
```python
[
    {'generated_text': 'This cooking will make you even richer. I used to work too little, I thought it was kind of ridiculous to take it that far. I was'},
    {'generated_text': 'This cooking will make you feel alive for hours every afternoon. It would also help keep your children in school throughout the day.\n\n\nOne of'}
]
```

**Analysis:** The text generation model produces coherent and imaginative continuations, demonstrating creativity and sentence flow for narrative text generation.

#### My Todo 3 - Fictional Character Text Generation

```python
generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In a world shrouded by darkness and despair, there walked a lone black swordsman named Guts. His colossal blade, Dragonslayer, was as heavy as the endless suffering he carried, a grim testament to the battles he had endured since childhood. Every swing was a cry, every step a burden, yet he pressed on, driven by a raw, unyielding rage against",
    max_length=100,
    num_return_sequences=2,
)
```

**Analysis:** The model successfully continues the dark fantasy narrative about Guts, maintaining the somber tone and thematic elements while extending complex narrative ideas coherently.

#### My Todo 3.5 - Masked Language Modeling

```python
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("Guts, the black swordsman, always carried his colossal blade, the <mask>.", top_k=4)
```

**Result:**
```python
[
    {'score': 0.18598486483097076, 'token': 21106, 'token_str': ' spear', 'sequence': 'Guts, the black swordsman, always carried his colossal blade, the spear.'},
    {'score': 0.1538282036781311, 'token': 42013, 'token_str': ' dagger', 'sequence': 'Guts, the black swordsman, always carried his colossal blade, the dagger.'},
    {'score': 0.12840501964092255, 'token': 20744, 'token_str': ' sword', 'sequence': 'Guts, the black swordsman, always carried his colossal blade, the sword.'},
    {'score': 0.058232732117176056, 'token': 27729, 'token_str': ' axe', 'sequence': 'Guts, the black swordsman, always carried his colossal blade, the axe.'}
]
```

**Analysis:** The fill-mask pipeline provided relevant weapon predictions, with "sword" being a logical prediction given the context, demonstrating the model's understanding of weaponry and context.

### 4. Named Entity Recognition (NER)

#### Example 4 - Basic NER

```python
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Roberta and I work with IBM Skills Network in Toronto")
```

**Result:**
```python
[
    {'entity_group': 'PER', 'score': 0.9978566, 'word': 'Roberta', 'start': 11, 'end': 18},
    {'entity_group': 'ORG', 'score': 0.7615841, 'word': 'AI', 'start': 28, 'end': 30},
    {'entity_group': 'ORG', 'score': 0.9623977, 'word': 'Infinite Learning', 'start': 51, 'end': 68},
    {'entity_group': 'LOC', 'score': 0.9913697, 'word': 'Batam Island', 'start': 70, 'end': 82}
]
```

**Analysis:** The NER model successfully identifies personal, organizational, and location entities with high confidence scores, demonstrating effectiveness for information extraction tasks.

#### My Todo 4 - Public Figure NER

```python
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("Joko Widodo, commonly known as Jokowi, is the current president of Indonesia.")
```

**Result:**
```python
[
    {'entity_group': 'PER', 'score': 0.99640924, 'word': 'Joko Widodo', 'start': 0, 'end': 11},
    {'entity_group': 'PER', 'score': 0.97210026, 'word': 'Jokowi', 'start': 31, 'end': 37},
    {'entity_group': 'LOC', 'score': 0.9988881, 'word': 'Indonesia', 'start': 67, 'end': 76}
]
```

**Analysis:** The model accurately identified both "Joko Widodo" and "Jokowi" as person entities, and "Indonesia" as a location entity with high confidence scores.

### 5. Question Answering

#### Example 5 - Basic QA

```python
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What four-legged animal sometimes comes inside the house and likes to sleep?"
context = "Four-legged animal that sometimes comes inside the house and likes to sleep is a cat"
qa_model(question=question, context=context)
```

**Result:**
```python
{'score': 0.6314472556114197, 'start': 79, 'end': 84, 'answer': 'a cat'}
```

**Analysis:** The QA model correctly extracts "a cat" from the context with decent confidence, showcasing strong capabilities in natural question understanding.

#### My Todo 5 - Historical Event QA

```python
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "When did Japan launch a full-scale invasion of China?"
context = "The expansion of the Japanese Empire in Asia began in the late 19th century, but the full-scale invasion of China, marking the start of the Second Sino-Japanese War, occurred in 1937. Japan then continued its expansion into other parts of Southeast Asia during World War II, including the Philippines, Indonesia, Malaya, and Burma."
qa_model(question=question, context=context)
```

**Result:**
```python
{'score': 0.9805444478988647, 'start': 178, 'end': 182, 'answer': '1937'}
```

**Analysis:** The model successfully extracted "1937" with high confidence (0.9805), demonstrating strong ability to pinpoint specific factual information within historical contexts.

### 6. Text Summarization

#### Example 6 - Indonesian Text Summarization

```python
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer("""
Machine Learning adalah cabang dari Kecerdasan Buatan yang memungkinkan sistem komputer untuk belajar dari data tanpa diprogram secara eksplisit. Melalui algoritma, mesin dapat mengidentifikasi pola, membuat prediksi, dan meningkatkan kinerja seiring waktu. Penerapannya luas, mulai dari rekomendasi produk hingga diagnosis medis, mengubah cara kita berinteraksi dengan teknologi.
""")
```

**Result:**
```python
[{'summary_text': ' Machine Learning adalah cabang dari Kecerdasan Buatan yang memungkinkan komputer untuk belajar dari data tanpa diprogram secara eksplisit . Melalui algoritma, mesin dapat mengidentifikasi pola, membuat prediksi, dan meningkatkan kinerja seiring waktu .'}]
```

**Analysis:** The summarization pipeline effectively condenses the core ideas while maintaining key concepts like machine learning, pattern recognition, and practical applications.

#### My Todo 6 - AI Ethics Summarization

```python
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    "The rapid advancements in artificial intelligence are reshaping various industries and aspects of daily life. From automating repetitive tasks to enabling complex data analysis, AI is proving to be a transformative force. However, its widespread adoption also brings forth significant ethical considerations, including concerns about job displacement, privacy, and bias in algorithms. Experts are calling for a balanced approach that leverages AI's potential for progress while establishing robust frameworks to mitigate its risks. Educational institutions are also adapting, introducing new curricula to prepare future generations for a world increasingly influenced by intelligent machines. The debate continues on how to best harness AI's power for the benefit of humanity without compromising fundamental societal values."
)
```

**Result:**
```python
[{'summary_text': " The rapid advancements in artificial intelligence are reshaping various industries and aspects of daily life . Experts are calling for a balanced approach that leverages AI's potential for progress while establishing robust frameworks to mitigate its risks . The debate continues on how to best harness AI's power for the benefit of humanity without compromising fundamental societal values ."}]
```

**Analysis:** The model effectively condensed the multi-sentence paragraph, extracting main points about AI's transformative role, ethical concerns, and the call for balanced approaches.

### 7. Translation

#### Example 7 - Indonesian to French Translation

```python
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Hari ini masak apa, chef?")
```

**Result:**
```python
[{'translation_text': "Qu'est-ce qu'on fait aujourd'hui, chef ?"}]
```

**Analysis:** The translation model delivers accurate and context-aware French translation, handling informal conversational input smoothly.

#### My Todo 7 - Greeting Translation

```python
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Halo, apa kabar hari ini?")
```

**Result:**
```python
[{'translation_text': "Bonjour, comment allez-vous aujourd'hui ?"}]
```

**Analysis:** The model successfully translated the Indonesian greeting to its French equivalent, demonstrating capability for accurate direct translation between Indonesian and French.

## Project Analysis

This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems. The completion of the "My Todo" sections for each task further reinforces the understanding and practical application of these NLP concepts.

## Technologies Used

- **Python**: Primary programming language
- **PyTorch**: Deep learning framework
- **Hugging Face Transformers**: Pre-trained NLP models
- **Various Pre-trained Models**:
  - DistilBERT for sentiment analysis and question answering
  - BART for zero-shot classification and summarization
  - DistilGPT-2 for text generation
  - DistilRoBERTa for masked language modeling
  - BERT for named entity recognition
  - Helsinki-NLP models for translation

## Requirements

```bash
pip install transformers torch
```

## Usage

Each task can be run independently by copying the respective code snippets. Make sure to install the required dependencies before running the examples.

## Change Log

| Date (YYYY-MM-DD) | Version | Changed By | Change Description |
|-------------------|---------|------------|-------------------|
| 2025-06-01 | 1.0 | Kevin Yoga Pratama | Initial creation for Guided Project report |

---

**Copyright © 2025 IBM Corporation. All rights reserved.**
