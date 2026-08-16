import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from torch.optim import AdamW
import time
# --------------------------------------------------
# Prepare the dataset
# --------------------------------------------------

dataset = load_dataset("Yelp/yelp_review_full")

# Inspect one training example
print(dataset["train"][100])


# Create smaller datasets for testing
small_train_dataset = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(1000))
)

small_eval_dataset = (
    dataset["test"]
    .shuffle(seed=42)
    .select(range(1000))
)


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True
    )


small_train_dataset = small_train_dataset.map(
    tokenize_function,
    batched=True
)

small_eval_dataset = small_eval_dataset.map(
    tokenize_function,
    batched=True
)


# --------------------------------------------------
# Load pretrained BERT model
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels=5
)


# --------------------------------------------------
# Create PyTorch DataLoaders
# --------------------------------------------------

# Remove raw text because the model only uses tokenized values and labels
small_train_dataset = small_train_dataset.remove_columns(["text"])
small_eval_dataset = small_eval_dataset.remove_columns(["text"])

# Convert Hugging Face Dataset values into PyTorch tensors
small_train_dataset.set_format("torch")
small_eval_dataset.set_format("torch")

# Dynamically pad each batch to the longest review in that batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    small_train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)

eval_loader = DataLoader(
    small_eval_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator
)


# --------------------------------------------------
# Optimizer and device setup
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

num_epochs = 3


# --------------------------------------------------
# Training and evaluation loop
# --------------------------------------------------

for epoch in range(num_epochs):
    # ----- Training -----
    model.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_number, batch in enumerate(train_loader, start=1):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(logits, dim=-1)
        labels = batch["labels"]

        total_train += labels.size(0)
        correct_train += (predictions == labels).sum().item()

        # Print progress every 10 batches
        if batch_number % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_number}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Average Loss: {epoch_loss:.4f}, "
        f"Training Accuracy: {train_accuracy * 100:.2f}%"
    )

    # ----- Evaluation -----
    model.eval()

    correct_eval = 0
    total_eval = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            labels = batch["labels"]

            total_eval += labels.size(0)
            correct_eval += (predictions == labels).sum().item()

    eval_accuracy = correct_eval / total_eval

    print(f"Validation Accuracy: {eval_accuracy * 100:.2f}%")