# Differences Across 10 Fine‑Tuning Programs

This document summarizes how the 10 programs differ in dataset, task, model, and training setup.

---

## Program overview

Each program is described using the same structure:

- **Dataset**
- **Task**
- **Number of labels**
- **Model**
- **Training size / Evaluation size**
- **Batch size**
- **Learning rate**
- **Epochs**
- **Text field(s)**
- **Special preprocessing**

---

## 0_original_yelp_5star

- **Dataset:** `Yelp/yelp_review_full`
- **Task:** 5‑star rating prediction
- **Labels:** 5
- **Model:** `bert-base-cased`
- **Train / Eval:** 1000 / 1000
- **Batch size:** 8
- **Learning rate:** 2e‑5
- **Epochs:** 3
- **Text field(s):** `text`
- **Special preprocessing:** none

---

## 1_imdb_binary

- **Dataset:** `imdb`
- **Task:** Binary sentiment (negative / positive)
- **Labels:** 2
- **Model:** `bert-base-uncased`
- **Train / Eval:** 800 / 800
- **Batch size:** 16
- **Learning rate:** 1e‑5
- **Epochs:** 2
- **Text field(s):** `text`
- **Special preprocessing:** none

---

## 2_ag_news_topic

- **Dataset:** `ag_news`
- **Task:** News topic classification (World, Sports, Business, Sci/Tech)
- **Labels:** 4
- **Model:** `distilbert-base-uncased`
- **Train / Eval:** 2000 / 1000
- **Batch size:** 32
- **Learning rate:** 3e‑5
- **Epochs:** 3
- **Text field(s):** `text`
- **Special preprocessing:** none

---

## 3_amazon_polarity

- **Dataset:** `amazon_polarity`
- **Task:** Binary sentiment (negative / positive)
- **Labels:** 2
- **Model:** `roberta-base`
- **Train / Eval:** 1500 / 1000
- **Batch size:** 16
- **Learning rate:** 2e‑5
- **Epochs:** 2
- **Text field(s):** `title` + `content`
- **Special preprocessing:** concatenate title and content

---

## 4_emotion_6class

- **Dataset:** `dair-ai/emotion`
- **Task:** Emotion classification (sadness, joy, love, anger, fear, surprise)
- **Labels:** 6
- **Model:** `bert-base-uncased`
- **Train / Eval:** 1200 / 600
- **Batch size:** 16
- **Learning rate:** 2e‑5
- **Epochs:** 3
- **Text field(s):** `text`
- **Special preprocessing:** none

---

## 5_sms_spam_binary

- **Dataset:** `sms_spam`
- **Task:** Spam detection (ham / spam)
- **Labels:** 2
- **Model:** `distilbert-base-uncased`
- **Train / Eval:** 600 / 300
- **Batch size:** 32
- **Learning rate:** 5e‑5
- **Epochs:** 4
- **Text field(s):** `sms`
- **Special preprocessing:** none

---

## 6_yelp_polarity_binary

- **Dataset:** `yelp_review_full` (filtered)
- **Task:** Binary polarity (negative / positive)
- **Labels:** 2
- **Model:** `bert-base-cased`
- **Train / Eval:** 1500 / 1000
- **Batch size:** 32
- **Learning rate:** 1e‑5
- **Epochs:** 2
- **Text field(s):** `text`
- **Special preprocessing:** filter out 3‑star reviews; map 0–1 stars → 0, 3–4 stars → 1

---

## 7_sst2_binary

- **Dataset:** `sst2`
- **Task:** Binary sentiment on short sentences
- **Labels:** 2
- **Model:** `bert-base-uncased`
- **Train / Eval:** 1200 / 600
- **Batch size:** 16
- **Learning rate:** 2e‑5
- **Epochs:** 3
- **Text field(s):** `sentence`
- **Special preprocessing:** none

---

## 8_dbpedia_14class

- **Dataset:** `dbpedia_14`
- **Task:** Topic classification (14 topics)
- **Labels:** 14
- **Model:** `distilbert-base-uncased`
- **Train / Eval:** 3000 / 1500
- **Batch size:** 32
- **Learning rate:** 3e‑5
- **Epochs:** 2
- **Text field(s):** `content`
- **Special preprocessing:** none

---

## 9_yahoo_answers_10class

- **Dataset:** `yahoo_answers_topics`
- **Task:** Q&A topic classification (10 topics)
- **Labels:** 10
- **Model:** `roberta-base`
- **Train / Eval:** 2500 / 1200
- **Batch size:** 16
- **Learning rate:** 2e‑5
- **Epochs:** 2
- **Text field(s):** `question_title` + `question_content`
- **Special preprocessing:** concatenate title and content

---

## Conceptual differences

### 1. Task type

- **Binary sentiment**
  - `1_imdb_binary`
  - `3_amazon_polarity`
  - `5_sms_spam_binary`
  - `6_yelp_polarity_binary`
  - `7_sst2_binary`

- **Multi‑class rating**
  - `0_original_yelp_5star` (5 classes: 1–5 stars)

- **Multi‑class topic**
  - `2_ag_news_topic` (4 topics)
  - `8_dbpedia_14class` (14 topics)
  - `9_yahoo_answers_10class` (10 topics)

- **Emotion classification**
  - `4_emotion_6class` (6 emotions)

---

### 2. Model family

- **BERT cased**
  - `0_original_yelp_5star`
  - `6_yelp_polarity_binary`

- **BERT uncased**
  - `1_imdb_binary`
  - `4_emotion_6class`
  - `7_sst2_binary`

- **DistilBERT (smaller, faster)**
  - `2_ag_news_topic`
  - `5_sms_spam_binary`
  - `8_dbpedia_14class`

- **RoBERTa**
  - `3_amazon_polarity`
  - `9_yahoo_answers_10class`

---

### 3. Data size & difficulty

- **Smallest dataset**
  - `5_sms_spam_binary` (600 train / 300 eval, very short SMS messages)

- **Largest dataset**
  - `8_dbpedia_14class` (3000 train / 1500 eval, 14 topics)

- **Medium‑sized datasets**
  - Most others (roughly 800–2500 training examples)

---

### 4. Batch size & learning rate

- **Small batch (8)**
  - `0_original_yelp_5star`

- **Medium batch (16)**
  - `1_imdb_binary`
  - `3_amazon_polarity`
  - `4_emotion_6class`
  - `7_sst2_binary`
  - `9_yahoo_answers_10class`

- **Large batch (32)**
  - `2_ag_news_topic`
  - `5_sms_spam_binary`
  - `6_yelp_polarity_binary`
  - `8_dbpedia_14class`

- **Lowest learning rate (1e‑5)**
  - `1_imdb_binary`
  - `6_yelp_polarity_binary`

- **Highest learning rate (5e‑5)**
  - `5_sms_spam_binary` (small data, short texts)

---

### 5. Preprocessing quirks

- **Concatenating multiple text fields**
  - `3_amazon_polarity`: `title` + `content`
  - `9_yahoo_answers_10class`: `question_title` + `question_content`

- **Filtering & label mapping**
  - `6_yelp_polarity_binary`:
    - Removes 3‑star reviews
    - Maps 0–1 stars → 0 (negative)
    - Maps 3–4 stars → 1 (positive)

- **All other programs**
  - Use a single text column directly (`text`, `sms`, `sentence`, or `content`).

---

## How to use this

- If you want **fast experiments**, start from a DistilBERT program with a small dataset (e.g., `5_sms_spam_binary` or `2_ag_news_topic`).
- If you want **higher accuracy** and have more compute, start from a BERT or RoBERTa program with more data (e.g., `1_imdb_binary`, `3_amazon_polarity`, or `9_yahoo_answers_10class`).
- If you want to experiment with **many classes**, use `8_dbpedia_14class` or `9_yahoo_answers_10class`.
- If you want a **simple binary sentiment baseline**, use `1_imdb_binary` or `7_sst2_binary`.