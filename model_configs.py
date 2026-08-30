"""Configuration for all 10 fine-tuning models."""

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"


@dataclass
class ModelConfig:
    id: str
    name: str
    description: str
    dataset: str
    model_name: str
    num_labels: int
    label_names: list[str]
    text_fields: list[str]
    text_field_labels: list[str]
    train_size: int
    eval_size: int
    test_size: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    epochs: int
    max_length: int = 256
    patience: int = 2
    eval_split: str = "test"
    legacy_checkpoint: str | None = None
    preprocess: str | None = None  # "yelp_polarity", "concat"

    @property
    def checkpoint_dir(self) -> Path:
        return MODELS_DIR / self.id

    @property
    def is_trained(self) -> bool:
        if self.checkpoint_dir.exists() and (self.checkpoint_dir / "config.json").exists():
            return True
        if self.legacy_checkpoint and (ROOT_DIR / self.legacy_checkpoint).exists():
            return True
        return False


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "yelp_5star": ModelConfig(
        id="yelp_5star",
        name="Yelp 5-Star Rating",
        description="Predict 1–5 star ratings from Yelp reviews",
        dataset="Yelp/yelp_review_full",
        model_name="bert-base-cased",
        num_labels=5,
        label_names=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
        text_fields=["text"],
        text_field_labels=["Review Text"],
        train_size=4000,
        eval_size=1000,
        test_size=1000,
        batch_size=4,
        eval_batch_size=8,
        learning_rate=2e-5,
        epochs=4,
        max_length=256,
        legacy_checkpoint="best_bert_yelp.pt",
    ),
    "imdb": ModelConfig(
        id="imdb",
        name="IMDB Sentiment",
        description="Binary sentiment classification on movie reviews",
        dataset="stanfordnlp/imdb",
        model_name="bert-base-uncased",
        num_labels=2,
        label_names=["Negative", "Positive"],
        text_fields=["text"],
        text_field_labels=["Review Text"],
        train_size=800,
        eval_size=800,
        test_size=500,
        batch_size=16,
        eval_batch_size=16,
        learning_rate=1e-5,
        epochs=2,
    ),
    "ag_news": ModelConfig(
        id="ag_news",
        name="AG News Topics",
        description="Classify news into World, Sports, Business, or Sci/Tech",
        dataset="fancyzhx/ag_news",
        model_name="distilbert-base-uncased",
        num_labels=4,
        label_names=["World", "Sports", "Business", "Sci/Tech"],
        text_fields=["text"],
        text_field_labels=["News Article"],
        train_size=2000,
        eval_size=1000,
        test_size=500,
        batch_size=32,
        eval_batch_size=32,
        learning_rate=3e-5,
        epochs=3,
    ),
    "amazon_polarity": ModelConfig(
        id="amazon_polarity",
        name="Amazon Polarity",
        description="Binary sentiment on Amazon product reviews",
        dataset="SetFit/amazon_polarity",
        model_name="roberta-base",
        num_labels=2,
        label_names=["Negative", "Positive"],
        text_fields=["title", "text"],
        text_field_labels=["Product Title", "Review Content"],
        train_size=1500,
        eval_size=1000,
        test_size=500,
        batch_size=16,
        eval_batch_size=16,
        learning_rate=2e-5,
        epochs=2,
        preprocess="concat",
    ),
    "emotion": ModelConfig(
        id="emotion",
        name="Emotion Classification",
        description="Detect sadness, joy, love, anger, fear, or surprise",
        dataset="dair-ai/emotion",
        model_name="bert-base-uncased",
        num_labels=6,
        label_names=["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"],
        text_fields=["text"],
        text_field_labels=["Text"],
        train_size=1200,
        eval_size=600,
        test_size=400,
        batch_size=16,
        eval_batch_size=16,
        learning_rate=2e-5,
        epochs=3,
    ),
    "sms_spam": ModelConfig(
        id="sms_spam",
        name="SMS Spam Detection",
        description="Classify SMS messages as ham or spam",
        dataset="ucirvine/sms_spam",
        model_name="distilbert-base-uncased",
        num_labels=2,
        label_names=["Ham", "Spam"],
        text_fields=["sms"],
        text_field_labels=["SMS Message"],
        train_size=600,
        eval_size=300,
        test_size=200,
        batch_size=32,
        eval_batch_size=32,
        learning_rate=5e-5,
        epochs=4,
    ),
    "yelp_polarity": ModelConfig(
        id="yelp_polarity",
        name="Yelp Polarity",
        description="Binary positive/negative from Yelp reviews (3-star excluded)",
        dataset="Yelp/yelp_review_full",
        model_name="bert-base-cased",
        num_labels=2,
        label_names=["Negative", "Positive"],
        text_fields=["text"],
        text_field_labels=["Review Text"],
        train_size=1500,
        eval_size=1000,
        test_size=500,
        batch_size=32,
        eval_batch_size=32,
        learning_rate=1e-5,
        epochs=2,
        preprocess="yelp_polarity",
    ),
    "sst2": ModelConfig(
        id="sst2",
        name="SST-2 Sentiment",
        description="Binary sentiment on short sentences",
        dataset="stanfordnlp/sst2",
        model_name="bert-base-uncased",
        num_labels=2,
        label_names=["Negative", "Positive"],
        text_fields=["sentence"],
        text_field_labels=["Sentence"],
        train_size=1200,
        eval_size=600,
        test_size=400,
        batch_size=16,
        eval_batch_size=16,
        learning_rate=2e-5,
        epochs=3,
        eval_split="validation",
    ),
    "dbpedia": ModelConfig(
        id="dbpedia",
        name="DBpedia 14-Class",
        description="Topic classification across 14 Wikipedia categories",
        dataset="fancyzhx/dbpedia_14",
        model_name="distilbert-base-uncased",
        num_labels=14,
        label_names=[
            "Company", "Educational Institution", "Artist", "Athlete",
            "Office Holder", "Mean of Transportation", "Building",
            "Natural Place", "Village", "Animal", "Plant", "Album",
            "Film", "Written Work",
        ],
        text_fields=["content"],
        text_field_labels=["Article Content"],
        train_size=3000,
        eval_size=1500,
        test_size=500,
        batch_size=32,
        eval_batch_size=32,
        learning_rate=3e-5,
        epochs=2,
    ),
    "yahoo_answers": ModelConfig(
        id="yahoo_answers",
        name="Yahoo Answers Topics",
        description="Classify Q&A into 10 topic categories",
        dataset="community-datasets/yahoo_answers_topics",
        model_name="roberta-base",
        num_labels=10,
        label_names=[
            "Society & Culture", "Science & Mathematics", "Health",
            "Education & Reference", "Computers & Internet", "Sports",
            "Business & Finance", "Entertainment & Music",
            "Family & Relationships", "Politics & Government",
        ],
        text_fields=["question_title", "question_content"],
        text_field_labels=["Question Title", "Question Content"],
        train_size=2500,
        eval_size=1200,
        test_size=500,
        batch_size=16,
        eval_batch_size=16,
        learning_rate=2e-5,
        epochs=2,
        preprocess="concat",
    ),
}


def get_model_config(model_id: str) -> ModelConfig:
    if model_id not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_id}'. "
            f"Available: {', '.join(MODEL_CONFIGS)}"
        )
    return MODEL_CONFIGS[model_id]


def list_models() -> list[ModelConfig]:
    return list(MODEL_CONFIGS.values())
