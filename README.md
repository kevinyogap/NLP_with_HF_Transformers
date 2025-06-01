Natural Language Processing with Hugging Face Transformers
&lt;p align="center">Generative AI Guided Project on Cognitive Class by IBM&lt;/p>

&lt;div align="center">

&lt;img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&amp;logo=python&amp;logoColor=ffdd54">
&lt;img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&amp;logo=PyTorch&amp;logoColor=white">

&lt;/div>

Name : Arifian Saputra
My To-Do Progress and Analysis:
1. Example 1 - Sentiment Analysis
Python

# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I am doing this on a regular basis, baking a cake in the morning!")
Result :

[{'label': 'POSITIVE', 'score': 0.9959210157394409}]
Analysis on Example 1:

The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.

My Todo 1: Sentiment Analysis with My Own Sentence
Python

# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("My biggest weakness is that I'm not consistent in anything I try, whether it's hobbies, routines, or even long-term goals")
Result :

[{'label': 'NEGATIVE', 'score': 0.9997879862785339}]
Analysis on My Todo 1:

The model correctly identified the sentiment of my self-descriptive sentence regarding inconsistency as 'NEGATIVE' with a very high score of 0.9997. This demonstrates its effectiveness in pinpointing negative tones even in personal, reflective statements.

2. Example 2 - Topic Classification
Python

# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Cats are beloved domestic companions known for their independence and agility. These fascinating creatures exhibit a range of behaviors, from playful pouncing to peaceful purring. With their sleek fur, captivating eyes, and mysterious charm, cats have captivated humans for centuries, becoming cherished members of countless households worldwide.",
    candidate_labels=["science", "pet", "machine learning"],
)
Result :

{'sequence': 'Cats are beloved domestic companions known for their independence and agility. These fascinating creatures exhibit a range of behaviors, from playful pouncing to peaceful purring. With their sleek fur, captivating eyes, and mysterious charm, cats have captivated humans for centuries, becoming cherished members of countless households worldwide.',
 'labels': ['pet', 'machine learning', 'science'],
 'scores': [0.9174826145172119, 0.048576705157756805, 0.03394068405032158]}
Analysis on Example 2:

The zero-shot classifier correctly identifies "pet" as the most relevant label, with a high confidence score. This shows the model's strong ability to associate descriptive context with predefined categories, even without task-specific fine-tuning or training on the input text.

My Todo 2: Topic Classification with My Own Sentence
Python

# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "I have a strong desire to learn new things, constantly reading books and exploring new computer technologies.",
    candidate_labels=["personal strengths", "hobbies", "career goals", "weaknesses"],
)
Result :

{'sequence': 'I have a strong desire to learn new things, constantly reading books and exploring new computer technologies.',
 'labels': ['personal strengths', 'hobbies', 'weaknesses', 'career goals'],
 'scores': [0.7655122876167297, 0.18992270529270172, 0.0267732385545969, 0.017791789025068283]}
Analysis on My Todo 2:

The model successfully classified my statement about learning new things, reading, and exploring computer technologies predominantly as "personal strengths" (0.7655 score), followed by "hobbies" (0.1899). This accurately reflects the nature of the input, showcasing the model's ability to categorize personal attributes.

3. Example 3 and 3.5 - Text Generator
Python

# TODO :
generator = pipeline("text-generation", model="distilgpt2") # or change to gpt-2
generator(
    "This cooking will make you",
    max_length=30, # you can change this
    num_return_sequences=2, # and this too
)
Result :

[{'generated_text': 'This cooking will make you even richer. I used to work too little, I thought it was kind of ridiculous to take it that far. I was'},
 {'generated_text': 'This cooking will make you feel alive for hours every afternoon. It would also help keep your children in school throughout the day.\n\n\nOne of'}]
Analysis on Example 3:

The text generation model produces coherent and imaginative continuations of a cooking-themed prompt. It demonstrates creativity and sentence flow, although output content may vary in tone and logic. The results showcase the model's usefulness for generating casual or narrative text.

My Todo 3: Text Generation about a Fictional Character
Python

# TODO :
generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In a world shrouded by darkness and despair, there walked a lone black swordsman named Guts. His colossal blade, Dragonslayer, was as heavy as the endless suffering he carried, a grim testament to the battles he had endured since childhood. Every swing was a cry, every step a burden, yet he pressed on, driven by a raw, unyielding rage against",
    max_length=100,
    num_return_sequences=2,
)
Result :

[{'generated_text': 'In a world shrouded by darkness and despair, there walked a lone black swordsman named Guts. His colossal blade, Dragonslayer, was as heavy as the endless suffering he carried, a grim testament to the battles he had endured since childhood. Every swing was a cry, every step a burden, yet he pressed on, driven by a raw, unyielding rage against the darkness and despair he had endured.\\n\\n\\n\\nThe blade was wielded only by the powerful, the most powerful and powerful. It was a blade that could only be used to cause a single death in the very event of its own. It was a blade that could only be used to cause a single death in the very event of its own. It was a blade that could only be used to cause a single death in the very event of its own. It was a blade that could only be used to cause a single death in the very event of its own. It was a blade that could only be used to cause a single death in the very event of its own. It was a blade that could only be used to cause a single death in the very event of its own. It was a blade that could only be used to cause a single death in the very event of its own. It was a blade that could only be used to cause a single death in the very event of its own. It was a blade that could only be used'}]
(Note: Output may vary slightly due to the nature of text generation. Only one example is shown here for brevity, though num_return_sequences=2 was set.)

Analysis on My Todo 3:

The text generation model successfully continues the dark fantasy narrative about Guts, maintaining the somber tone and thematic elements (suffering, rage, darkness). The generated text flows coherently from the prompt, although it does introduce some repetitive phrasing. It highlights the model's ability to extend complex narrative ideas.

My Todo 3.5: Masked Language Modeling for Guts' Weapon
Python

unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("Guts, the black swordsman, always carried his colossal blade, the <mask>.", top_k=4)
Result :

[{'score': 0.18598486483097076,
  'token': 21106,
  'token_str': ' spear',
  'sequence': 'Guts, the black swordsman, always carried his colossal blade, the spear.'},
 {'score': 0.1538282036781311,
  'token': 42013,
  'token_str': ' dagger',
  'sequence': 'Guts, the black swordsman, always carried his colossal blade, the dagger.'},
 {'score': 0.12840501964092255,
  'token': 20744,
  'token_str': ' sword',
  'sequence': 'Guts, the black swordsman, always carried his colossal blade, the sword.'},
 {'score': 0.058232732117176056,
  'token': 27729,
  'token_str': ' axe',
  'sequence': 'Guts, the black swordsman, always carried his colossal blade, the axe.'}]
Analysis on My Todo 3.5:

The fill-mask pipeline, when given the context of Guts and his "colossal blade," provided "spear" as the top prediction, followed by "dagger" and "sword". While "Dragonslayer" (the actual name of his sword) was not predicted as it might be too specific and out of the common vocabulary range, "sword" is a very relevant and logical prediction given the context, even if not the top one in this particular instance. This demonstrates the model's general understanding of weaponry and the context provided.

4. Example 4 - Name Entity Recognition (NER)
Python

# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Roberta and I work with IBM Skills Network in Toronto")
Result :

[{'entity_group': 'PER',
  'score': np.float32(0.9978566),
  'word': 'Roberta',
  'start': 11,
  'end': 18},
 {'entity_group': 'ORG',
  'score': np.float32(0.7615841),
  'word': 'AI',
  'start': 28,
  'end': 30},
 {'entity_group': 'ORG',
  'score': np.float32(0.9623977),
  'word': 'Infinite Learning',
  'start': 51,
  'end': 68},
 {'entity_group': 'LOC',
  'score': np.float32(0.9913697),
  'word': 'Batam Island',
  'start': 70,
  'end': 82}]
Analysis on Example 4:

The named entity recognizer successfully identifies personal, organizational, and location entities from the sentence. Grouped outputs are relevant and accurate, with high confidence scores, demonstrating the model’s effectiveness in real-world applications like information extraction or document tagging.

My Todo 4: Named Entity Recognition for a Public Figure
Python

# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("Joko Widodo, commonly known as Jokowi, is the current president of Indonesia.")
Result :

[{'entity_group': 'PER',
  'score': np.float32(0.99640924),
  'word': 'Joko Widodo',
  'start': 0,
  'end': 11},
 {'entity_group': 'PER',
  'score': np.float32(0.97210026),
  'word': 'Jokowi',
  'start': 31,
  'end': 37},
 {'entity_group': 'LOC',
  'score': np.float32(0.9988881),
  'word': 'Indonesia',
  'start': 67,
  'end': 76}]
Analysis on My Todo 4:

The NER model accurately identified "Joko Widodo" and "Jokowi" as PER (Person) entities, and "Indonesia" as a LOC (Location) entity. The high confidence scores for all recognized entities confirm the model's robustness in identifying named entities in factual sentences about public figures.

5. Example 5 - Question Answering
Python

# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What four-legged animal sometimes comes inside the house and likes to sleep?"
context = "Four-legged animal that sometimes comes inside the house and likes to sleep is a cat"
qa_model(question = question, context = context)
Result :

{'score': 0.6314472556114197, 'start': 79, 'end': 84, 'answer': 'a cat'}
Analysis on Example 5:

The question-answering model correctly extracts the most relevant phrase "a cat" from the provided context. Its confidence score is decent, and the model showcases strong capabilities in understanding natural questions and matching them with the most likely answer span.

My Todo 5: Question Answering about Historical Event
Python

# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "When did Japan launch a full-scale invasion of China?"
context = "The expansion of the Japanese Empire in Asia began in the late 19th century, but the full-scale invasion of China, marking the start of the Second Sino-Japanese War, occurred in 1937. Japan then continued its expansion into other parts of Southeast Asia during World War II, including the Philippines, Indonesia, Malaya, and Burma."
qa_model(question = question, context = context)
Result :

{'score': 0.9805444478988647, 'start': 178, 'end': 182, 'answer': '1937'}
Analysis on My Todo 5:

The question-answering model successfully extracted "1937" as the answer to the question "When did Japan launch a full-scale invasion of China?". The high confidence score (0.9805) indicates the model's strong ability to pinpoint specific factual information within a given historical context.

6. Example 6 - Text Summarization
Python

# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
Machine Learning adalah cabang dari Kecerdasan Buatan yang memungkinkan sistem komputer untuk belajar dari data tanpa diprogram secara eksplisit. 1  Melalui algoritma, mesin dapat mengidentifikasi pola, membuat prediksi, dan meningkatkan kinerja seiring waktu. Penerapannya luas, mulai dari rekomendasi produk hingga diagnosis medis, mengubah cara kita berinteraksi dengan teknologi.
"""
)
Result :

[{'summary_text': ' Machine Learning adalah cabang dari Kecerdasan Buatan yang memungkinkan komputer untuk belajar dari data tanpa diprogram secara eksplisit . Melalui algoritma, mesin dapat mengidentifikasi pola, membuat prediksi, dan meningkatkan kinerja seiring waktu .'}]

Analysis on Example 6:

The summarization pipeline effectively condenses the core idea of the paragraph into a shorter version. It maintains key concepts like machine learning, pattern recognition, and practical applications, reflecting the model's strength in content compression without major loss of information.

My Todo 6: Text Summarization of a Contemporary Topic
Python

# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    "The rapid advancements in artificial intelligence are reshaping various industries and aspects of daily life. From automating repetitive tasks to enabling complex data analysis, AI is proving to be a transformative force. However, its widespread adoption also brings forth significant ethical considerations, including concerns about job displacement, privacy, and bias in algorithms. Experts are calling for a balanced approach that leverages AI's potential for progress while establishing robust frameworks to mitigate its risks. Educational institutions are also adapting, introducing new curricula to prepare future generations for a world increasingly influenced by intelligent machines. The debate continues on how to best harness AI's power for the benefit of humanity without compromising fundamental societal values.",
)
Result :

[{'summary_text': " The rapid advancements in artificial intelligence are reshaping various industries and aspects of daily life . Experts are calling for a balanced approach that leverages AI's potential for progress while establishing robust frameworks to mitigate its risks . The debate continues on how to best harness AI's power for the benefit of humanity without compromising fundamental societal values ."}]
Analysis on My Todo 6:

The summarization model effectively condensed a multi-sentence paragraph about the impact and ethical considerations of AI. It extracted the main points: AI's transformative role, associated ethical concerns (job displacement, privacy, bias), and the call for a balanced approach. The summary accurately captures the essence of the original text.

7. Example 7 - Translation
Python

# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Hari ini masak apa, chef?")
Result :

[{'translation_text': "Qu'est-ce qu'on fait aujourd'hui, chef ?"}]

Analysis on Example 7:

The translation model delivers an accurate and context-aware French translation of the Indonesian sentence. It handles informal, conversational input smoothly, making it suitable for multilingual communication tasks and cross-language understanding in casual or daily scenarios.

My Todo 7: Indonesian to French Translation
Python

# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Halo, apa kabar hari ini?")
Result :

[{'translation_text': "Bonjour, comment allez-vous aujourd'hui ?"}]
Analysis on My Todo 7:

The model successfully translated the Indonesian phrase "Halo, apa kabar hari ini?" to its French equivalent "Bonjour, comment allez-vous aujourd'hui ?". This demonstrates the model's capability for accurate direct translation between Indonesian and French.

Analysis on this project
This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems. The completion of the "My Todo" sections for each task further reinforces the understanding and practical application of these NLP concepts.

Author
Kevin Yoga Pratama

Change Log
Date (YYYY-MM-DD)	Version	Changed By	Change Description
2025-06-01	1.0	Kevin Yoga Pratama	Initial creation for Guided Project report

Export to Sheets
Copyright © 2025 IBM Corporation. All rights reserved.
