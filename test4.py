
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Amazon Polarity dataset...")

dataset = load_dataset("fancyzhx/amazon_polarity")

print("Dataset loaded successfully.")


# ============================================================
# CREATE TRAINING DATA
# ============================================================

data = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(5000))
)

split_data = data.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = split_data["train"]
eval_dataset = split_data["test"]


# ============================================================
# CREATE TEST DATA
# ============================================================

test_dataset = (
    dataset["test"]
    .shuffle(seed=42)
    .select(range(1000))
)


print(f"\nTraining: {len(train_dataset)}")
print(f"Validation: {len(eval_dataset)}")
print(f"Testing: {len(test_dataset)}")


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "roberta-base"
)


# ============================================================
# TOKENIZATION FUNCTION
# ============================================================

def tokenize_function(examples):

    texts = [
        f"{title} {content}"
        for title, content
        in zip(
            examples["title"],
            examples["content"]
        )
    ]

    return tokenizer(
        texts,
        truncation=True,
        max_length=256
    )


# ============================================================
# TOKENIZE DATASETS
# ============================================================

print("Tokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# ============================================================
# REMOVE ORIGINAL TEXT COLUMNS
# ============================================================

train_dataset = train_dataset.remove_columns(
    ["title", "content"]
)

eval_dataset = eval_dataset.remove_columns(
    ["title", "content"]
)

test_dataset = test_dataset.remove_columns(
    ["title", "content"]
)


# ============================================================
# RENAME LABEL
# ============================================================

train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)

eval_dataset = eval_dataset.rename_column(
    "label",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "label",
    "labels"
)


# ============================================================
# PYTORCH FORMAT
# ============================================================

train_dataset.set_format("torch")
eval_dataset.set_format("torch")
test_dataset.set_format("torch")


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt"
)


# ============================================================
# DATALOADERS
# ============================================================

batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator
)


print("\nDataLoaders created.")


# ============================================================
# MODEL
# ============================================================

print("\nLoading RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

model.to(device)

print("Model loaded.")


# ============================================================
# TRAINING SETTINGS
# ============================================================

num_epochs = 3

learning_rate = 2e-5

optimizer = AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

num_training_steps = (
    len(train_loader) * num_epochs
)

warmup_steps = int(
    0.1 * num_training_steps
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)


# ============================================================
# EARLY STOPPING
# ============================================================

best_eval_loss = float("inf")

patience = 2

patience_counter = 0

best_model_path = "best_roberta_amazon.pt"


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(num_epochs):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0

    correct_train = 0

    total_train = 0


    for batch_number, batch in enumerate(
        train_loader,
        start=1
    ):

        # Move batch to device
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(**batch)

        loss = outputs.loss

        logits = outputs.logits


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()

        scheduler.step()


        # Track loss
        running_loss += loss.item()


        # Predictions
        predictions = torch.argmax(
            logits,
            dim=-1
        )

        labels = batch["labels"]


        total_train += labels.size(0)

        correct_train += (
            predictions == labels
        ).sum().item()


        # Progress
        if batch_number % 50 == 0:

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_number}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )


    # ========================================================
    # TRAINING METRICS
    # ========================================================

    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        correct_train /
        total_train
    )


    print(
        f"\nEpoch [{epoch + 1}/{num_epochs}]"
    )

    print(
        f"Average Training Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    running_eval_loss = 0.0

    correct_eval = 0

    total_eval = 0


    with torch.no_grad():

        for batch in eval_loader:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }


            outputs = model(**batch)

            loss = outputs.loss

            logits = outputs.logits


            running_eval_loss += loss.item()


            predictions = torch.argmax(
                logits,
                dim=-1
            )

            labels = batch["labels"]


            total_eval += labels.size(0)

            correct_eval += (
                predictions == labels
            ).sum().item()


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    eval_loss = (
        running_eval_loss /
        len(eval_loader)
    )

    eval_accuracy = (
        correct_eval /
        total_eval
    )


    print(
        f"Validation Loss: "
        f"{eval_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{eval_accuracy * 100:.2f}%"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if eval_loss < best_eval_loss:

        best_eval_loss = eval_loss

        patience_counter = 0


        torch.save(
            model.state_dict(),
            best_model_path
        )


        print("Best model saved.")


    else:

        patience_counter += 1


        print(
            f"No improvement. "
            f"Patience: "
            f"{patience_counter}/{patience}"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if patience_counter >= patience:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)

model.to(device)

model.eval()


# ============================================================
# FINAL TEST
# ============================================================

print("\nEvaluating on final test set...")


running_test_loss = 0.0

correct_test = 0

total_test = 0


with torch.no_grad():

    for batch in test_loader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }


        outputs = model(**batch)

        loss = outputs.loss

        logits = outputs.logits


        running_test_loss += loss.item()


        predictions = torch.argmax(
            logits,
            dim=-1
        )

        labels = batch["labels"]


        total_test += labels.size(0)

        correct_test += (
            predictions == labels
        ).sum().item()


# ============================================================
# TEST METRICS
# ============================================================

test_loss = (
    running_test_loss /
    len(test_loader)
)

test_accuracy = (
    correct_test /
    total_test
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 50)

print("FINAL TEST RESULTS")

print("=" * 50)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("=" * 50)

print("\nTraining complete!")

