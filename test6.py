import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading SMS Spam dataset...")

dataset = load_dataset("ucirvine/sms_spam")

print(dataset)
print(dataset["train"].column_names)


# --------------------------------------------------
# Create train/evaluation split
# --------------------------------------------------

data = dataset["train"].shuffle(seed=42)

split = data.train_test_split(
    test_size=0.2,
    seed=42
)

# Safe dataset size
small_train = split["train"].select(
    range(min(600, len(split["train"])))
)

small_eval = split["test"].select(
    range(min(300, len(split["test"])))
)

print(f"Training examples: {len(small_train)}")
print(f"Validation examples: {len(small_eval)}")


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


def tokenize_function(examples):
    return tokenizer(
        examples["sms"],
        truncation=True,
        max_length=64,
    )


small_train = small_train.map(
    tokenize_function,
    batched=True
)

small_eval = small_eval.map(
    tokenize_function,
    batched=True
)


# --------------------------------------------------
# Convert labels
# --------------------------------------------------

def convert_label(example):

    if isinstance(example["label"], str):

        if example["label"].lower() == "spam":
            example["label"] = 1
        else:
            example["label"] = 0

    return example


small_train = small_train.map(convert_label)
small_eval = small_eval.map(convert_label)


# --------------------------------------------------
# Load DistilBERT
# --------------------------------------------------

print("Loading DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# --------------------------------------------------
# Remove original text column
# --------------------------------------------------

small_train = small_train.remove_columns(["sms"])
small_eval = small_eval.remove_columns(["sms"])


# --------------------------------------------------
# PyTorch format
# --------------------------------------------------

small_train.set_format("torch")
small_eval.set_format("torch")


# --------------------------------------------------
# Data collator
# --------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# --------------------------------------------------
# Data loaders
# --------------------------------------------------

train_loader = DataLoader(
    small_train,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator,
)

eval_loader = DataLoader(
    small_eval,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator,
)


# --------------------------------------------------
# Select device
# --------------------------------------------------

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print(f"Using device: {device}")


# --------------------------------------------------
# Move model to device
# --------------------------------------------------

model.to(device)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=5e-5
)

num_epochs = 4


# --------------------------------------------------
# Training
# --------------------------------------------------

print()
print("Starting training...")
print("=" * 60)


for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0


    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for batch_number, batch in enumerate(
        train_loader,
        start=1
    ):

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }


        optimizer.zero_grad()


        outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits


        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        # --------------------------------------------------
        # Training accuracy
        # --------------------------------------------------

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        labels = batch["labels"]

        total_train += labels.size(0)

        correct_train += (
            predictions == labels
        ).sum().item()


        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if batch_number % 10 == 0:

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_number}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )


    # --------------------------------------------------
    # Training statistics
    # --------------------------------------------------

    epoch_loss = (
        running_loss / len(train_loader)
    )

    train_acc = (
        correct_train / total_train
    )


    print()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}]"
    )

    print(
        f"Average Training Loss: "
        f"{epoch_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_acc * 100:.2f}%"
    )


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    correct_eval = 0
    total_eval = 0

    validation_loss = 0.0


    with torch.no_grad():

        for batch in eval_loader:

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }


            outputs = model(**batch)


            validation_loss += (
                outputs.loss.item()
            )


            predictions = torch.argmax(
                outputs.logits,
                dim=-1
            )

            labels = batch["labels"]


            total_eval += labels.size(0)

            correct_eval += (
                predictions == labels
            ).sum().item()


    # --------------------------------------------------
    # Validation statistics
    # --------------------------------------------------

    eval_loss = (
        validation_loss / len(eval_loader)
    )

    eval_acc = (
        correct_eval / total_eval
    )


    print(
        f"Validation Loss: "
        f"{eval_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{eval_acc * 100:.2f}%"
    )

    print("=" * 60)


# --------------------------------------------------
# Finished
# --------------------------------------------------

print()
print("Training complete!")

print(
    f"Final Validation Accuracy: "
    f"{eval_acc * 100:.2f}%"
)