
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

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

dataset = load_dataset("amazon_polarity")

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

test_dataset = (
    dataset["test"]
    .shuffle(seed=42)
    .select(range(1000))
)

print(f"Training: {len(train_dataset)}")
print(f"Validation: {len(eval_dataset)}")
print(f"Testing: {len(test_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(
    "roberta-base"
)

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

train_dataset = train_dataset.remove_columns(
    ["title", "content"]
)

eval_dataset = eval_dataset.remove_columns(
    ["title", "content"]
)

test_dataset = test_dataset.remove_columns(
    ["title", "content"]
)

train_dataset.set_format("torch")
eval_dataset.set_format("torch")
test_dataset.set_format("torch")

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

model.to(device)

num_epochs = 3
learning_rate = 2e-5

optimizer = AdamW(
    model.parameters(),
    lr=learning_rate
)

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

best_eval_loss = float("inf")
patience = 2
patience_counter = 0

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_number, batch in enumerate(
        train_loader,
        start=1
    ):

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
        scheduler.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        labels = batch["labels"]

        total_train += labels.size(0)

        correct_train += (
            predictions == labels
        ).sum().item()

        if batch_number % 50 == 0:

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_number}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

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

    if eval_loss < best_eval_loss:

        best_eval_loss = eval_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            "best_roberta_amazon.pt"
        )

        print("Best model saved.")

    else:

        patience_counter += 1

        print(
            f"No improvement. "
            f"Patience: {patience_counter}/{patience}"
        )

    if patience_counter >= patience:

        print(
            "Early stopping triggered."
        )

        break

print("\nLoading best saved model...")

model.load_state_dict(
    torch.load(
        "best_roberta_amazon.pt",
        map_location=device
    )
)

model.to(device)
model.eval()

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

test_loss = (
    running_test_loss /
    len(test_loader)
)

test_accuracy = (
    correct_test /
    total_test
)

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

print("\nTraining complete!")
