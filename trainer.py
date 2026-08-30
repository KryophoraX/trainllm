"""Shared fine-tuning trainer for all models."""

import json
from pathlib import Path
from typing import Callable

import torch
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from model_configs import MODELS_DIR, ModelConfig, get_model_config


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _apply_yelp_polarity(dataset: Dataset) -> Dataset:
    def to_polarity(example):
        label = example["label"]
        if label <= 1:
            return {"labels": 0}
        if label >= 3:
            return {"labels": 1}
        return {"labels": -1}

    dataset = dataset.map(to_polarity)
    dataset = dataset.filter(lambda x: x["labels"] != -1)
    cols_to_drop = [c for c in dataset.column_names if c not in ("text", "labels")]
    return dataset.remove_columns(cols_to_drop)


def _standardize_labels(dataset: Dataset) -> Dataset:
    if "labels" in dataset.column_names:
        return dataset
    if "label" in dataset.column_names:
        return dataset.rename_column("label", "labels")
    if "topic" in dataset.column_names:
        return dataset.rename_column("topic", "labels")
    return dataset


def prepare_datasets(config: ModelConfig):
    dataset = load_dataset(config.dataset)

    if config.preprocess == "yelp_polarity":
        train_pool = _apply_yelp_polarity(dataset["train"])
        test_pool = _apply_yelp_polarity(dataset["test"])
    else:
        train_pool = _standardize_labels(dataset["train"])
        test_pool = (
            _standardize_labels(dataset[config.eval_split])
            if config.eval_split in dataset
            else None
        )

    if test_pool is None:
        total = config.train_size + config.eval_size + config.test_size
        pool = train_pool.shuffle(seed=42).select(range(total))
        train_dataset = pool.select(range(config.train_size))
        eval_dataset = pool.select(
            range(config.train_size, config.train_size + config.eval_size)
        )
        test_dataset = pool.select(
            range(config.train_size + config.eval_size, total)
        )
    else:
        total_train_val = config.train_size + config.eval_size
        train_val = train_pool.shuffle(seed=42).select(range(total_train_val))
        split = train_val.train_test_split(
            test_size=config.eval_size,
            seed=42,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        test_dataset = test_pool.shuffle(seed=42).select(range(config.test_size))

    return train_dataset, eval_dataset, test_dataset


def _build_texts(config: ModelConfig, examples: dict) -> list[str]:
    if config.preprocess == "concat":
        return [
            " ".join(str(examples[field][i]) for field in config.text_fields)
            for i in range(len(examples[config.text_fields[0]]))
        ]
    return examples[config.text_fields[0]]


def tokenize_datasets(
    config: ModelConfig,
    train_dataset,
    eval_dataset,
    test_dataset,
    tokenizer,
):
    def tokenize_function(examples):
        texts = _build_texts(config, examples)
        return tokenizer(
            texts,
            truncation=True,
            max_length=config.max_length,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    columns_to_remove = [
        col
        for col in train_dataset.column_names
        if col not in ("input_ids", "attention_mask", "labels")
    ]
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    eval_dataset = eval_dataset.remove_columns(columns_to_remove)
    test_dataset = test_dataset.remove_columns(columns_to_remove)

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, eval_dataset, test_dataset


def evaluate(model, data_loader, device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            running_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model_id: str,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Train a model and return final metrics."""
    config = get_model_config(model_id)
    device = get_device()

    def log(message: str):
        print(message)
        if progress_callback:
            progress_callback(message)

    log(f"Training: {config.name}")
    log(f"Device: {device}")

    train_dataset, eval_dataset, test_dataset = prepare_datasets(config)

    log(
        f"Dataset sizes — train: {len(train_dataset)}, "
        f"val: {len(eval_dataset)}, test: {len(test_dataset)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset, eval_dataset, test_dataset = tokenize_datasets(
        config, train_dataset, eval_dataset, test_dataset, tokenizer
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = len(train_loader) * config.epochs
    warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_eval_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(config.epochs):
        log(f"\n{'=' * 60}")
        log(f"Epoch {epoch + 1}/{config.epochs}")
        log("=" * 60)

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_number, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            total_train += labels.size(0)
            correct_train += (predictions == labels).sum().item()

            if batch_number % 10 == 0:
                log(
                    f"  Batch {batch_number}/{len(train_loader)} "
                    f"— loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        log(f"Train — loss: {train_loss:.4f}, accuracy: {train_acc * 100:.2f}%")

        eval_loss, eval_acc = evaluate(model, eval_loader, device)
        log(f"Val   — loss: {eval_loss:.4f}, accuracy: {eval_acc * 100:.2f}%")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log("  ✓ Validation improved — saving best checkpoint")
        else:
            patience_counter += 1
            log(f"  No improvement ({patience_counter}/{config.patience})")

        if patience_counter >= config.patience:
            log("\nEarly stopping triggered.")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    test_loss, test_acc = evaluate(model, test_loader, device)
    log(f"\nTest — loss: {test_loss:.4f}, accuracy: {test_acc * 100:.2f}%")

    save_dir = config.checkpoint_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    metrics = {
        "model_id": model_id,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_val_loss": best_eval_loss,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"\nModel saved to {save_dir}")
    log("Training complete!")

    return metrics
