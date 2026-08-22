import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)



dataset = load_dataset("dair-ai/emotion")

small_train = dataset["train"].shuffle(seed=42).select(range(1200))
small_eval = dataset["test"].shuffle(seed=42).select(range(600))



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


small_train = small_train.map(tokenize_function, batched=True)
small_eval = small_eval.map(tokenize_function, batched=True)




model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6
)




small_train = small_train.remove_columns(["text"])
small_eval = small_eval.remove_columns(["text"])

small_train.set_format("torch")
small_eval.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    small_train,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator,
)

eval_loader = DataLoader(
    small_eval,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator,
)




if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3




for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_number, batch in enumerate(train_loader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

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

        if batch_number % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_number}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Average Loss: {epoch_loss:.4f}, "
        f"Training Accuracy: {train_acc * 100:.2f}%"
    )

   
    model.eval()
    correct_eval = 0
    total_eval = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]

            total_eval += labels.size(0)
            correct_eval += (predictions == labels).sum().item()

    eval_acc = correct_eval / total_eval
    print(f"Validation Accuracy: {eval_acc * 100:.2f}%")