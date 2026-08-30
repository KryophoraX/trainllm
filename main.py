import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW


# ==================================================
# 1. DEVICE SETUP
# ==================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


print(f"Using device: {device}")


# ==================================================
# 2. LOAD DATASET
# ==================================================

dataset = load_dataset("Yelp/yelp_review_full")

print("\nExample review:")
print(dataset["train"][100])


# ==================================================
# 3. CREATE TRAIN AND VALIDATION DATA
# ==================================================

# Select 5000 examples
# These will be split into:
# 4000 training examples
# 1000 validation examples

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


# ==================================================
# 4. CREATE SEPARATE FINAL TEST DATA
# ==================================================

# This data is completely separate
# and is only used after training

test_dataset = (
    dataset["test"]
    .shuffle(seed=42)
    .select(range(1000))
)


print("\nDataset Sizes:")

print(f"Training: {len(train_dataset)}")

print(f"Validation: {len(eval_dataset)}")

print(f"Testing: {len(test_dataset)}")


# ==================================================
# 5. LOAD TOKENIZER
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-cased"
)


# ==================================================
# 6. TOKENIZE DATA
# ==================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
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


# ==================================================
# 7. REMOVE RAW TEXT
# ==================================================

train_dataset = train_dataset.remove_columns(
    ["text"]
)


eval_dataset = eval_dataset.remove_columns(
    ["text"]
)


test_dataset = test_dataset.remove_columns(
    ["text"]
)


# ==================================================
# 8. SET PYTORCH FORMAT
# ==================================================

train_dataset.set_format("torch")

eval_dataset.set_format("torch")

test_dataset.set_format("torch")


# ==================================================
# 9. DATA COLLATOR
# ==================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ==================================================
# 10. CREATE DATALOADERS
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=data_collator
)


eval_loader = DataLoader(
    eval_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator
)


test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator
)


# ==================================================
# 11. LOAD PRETRAINED BERT MODEL
# ==================================================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels=5
)


model.to(device)


# ==================================================
# 12. TRAINING SETTINGS
# ==================================================

num_epochs = 4

learning_rate = 2e-5


# ==================================================
# 13. OPTIMIZER
# ==================================================

optimizer = AdamW(
    model.parameters(),
    lr=learning_rate
)


# ==================================================
# 14. LEARNING RATE SCHEDULER
# ==================================================

num_training_steps = (
    len(train_loader)
    * num_epochs
)


warmup_steps = int(
    0.1 * num_training_steps
)


scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)


# ==================================================
# 15. EARLY STOPPING SETTINGS
# ==================================================

best_eval_loss = float("inf")

patience = 2

patience_counter = 0


# ==================================================
# 16. TRAINING LOOP
# ==================================================

for epoch in range(num_epochs):

    print("\n" + "=" * 60)

    print(
        f"STARTING EPOCH "
        f"{epoch + 1}/{num_epochs}"
    )

    print("=" * 60)


    # ----------------------------------------------
    # TRAINING MODE
    # ----------------------------------------------

    model.train()


    running_loss = 0.0

    correct_train = 0

    total_train = 0


    # ----------------------------------------------
    # TRAINING BATCHES
    # ----------------------------------------------

    for batch_number, batch in enumerate(
        train_loader,
        start=1
    ):

        # Move data to device

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }


        # Clear old gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(**batch)


        loss = outputs.loss

        logits = outputs.logits


        # Backpropagation

        loss.backward()


        # Update model

        optimizer.step()


        # Update learning rate

        scheduler.step()


        # ------------------------------------------
        # SAVE TRAINING LOSS
        # ------------------------------------------

        running_loss += loss.item()


        # ------------------------------------------
        # CALCULATE TRAINING ACCURACY
        # ------------------------------------------

        predictions = torch.argmax(
            logits,
            dim=-1
        )


        labels = batch["labels"]


        total_train += labels.size(0)


        correct_train += (
            predictions == labels
        ).sum().item()


        # ------------------------------------------
        # PRINT PROGRESS
        # ------------------------------------------

        if batch_number % 10 == 0:

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_number}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )


    # ==================================================
    # 17. TRAINING RESULTS
    # ==================================================

    train_loss = (
        running_loss
        / len(train_loader)
    )


    train_accuracy = (
        correct_train
        / total_train
    )


    print("\nTraining Results:")


    print(
        f"Average Training Loss: "
        f"{train_loss:.4f}"
    )


    print(
        f"Training Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )


    # ==================================================
    # 18. VALIDATION
    # ==================================================

    model.eval()


    running_eval_loss = 0.0

    correct_eval = 0

    total_eval = 0


    with torch.no_grad():

        for batch in eval_loader:


            # Move data to device

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }


            # Forward pass

            outputs = model(**batch)


            loss = outputs.loss

            logits = outputs.logits


            # Add validation loss

            running_eval_loss += loss.item()


            # Get predictions

            predictions = torch.argmax(
                logits,
                dim=-1
            )


            labels = batch["labels"]


            total_eval += labels.size(0)


            correct_eval += (
                predictions == labels
            ).sum().item()


    # ==================================================
    # 19. VALIDATION RESULTS
    # ==================================================

    eval_loss = (
        running_eval_loss
        / len(eval_loader)
    )


    eval_accuracy = (
        correct_eval
        / total_eval
    )


    print("\nValidation Results:")


    print(
        f"Validation Loss: "
        f"{eval_loss:.4f}"
    )


    print(
        f"Validation Accuracy: "
        f"{eval_accuracy * 100:.2f}%"
    )


    # ==================================================
    # 20. SAVE BEST MODEL
    # ==================================================

    if eval_loss < best_eval_loss:


        best_eval_loss = eval_loss


        patience_counter = 0


        print(
            "\nValidation loss improved!"
        )


        torch.save(
            model.state_dict(),
            "best_bert_yelp.pt"
        )


        print(
            "Best model saved!"
        )


    else:


        patience_counter += 1


        print(
            "\nNo improvement in validation loss."
        )


        print(
            f"Patience: "
            f"{patience_counter}/{patience}"
        )


    # ==================================================
    # 21. EARLY STOPPING
    # ==================================================

    if patience_counter >= patience:


        print(
            "\nEarly stopping triggered."
        )


        break


# ==================================================
# 22. LOAD BEST MODEL
# ==================================================

print("\n" + "=" * 60)

print(
    "Loading best saved model..."
)

print("=" * 60)


model.load_state_dict(
    torch.load(
        "best_bert_yelp.pt",
        map_location=device
    )
)


model.to(device)

model.eval()


# ==================================================
# 23. FINAL TEST EVALUATION
# ==================================================

print(
    "\nEvaluating on final test set..."
)


running_test_loss = 0.0

correct_test = 0

total_test = 0


with torch.no_grad():

    for batch in test_loader:


        # Move data to device

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }


        # Forward pass

        outputs = model(**batch)


        loss = outputs.loss

        logits = outputs.logits


        # Add test loss

        running_test_loss += loss.item()


        # Get predictions

        predictions = torch.argmax(
            logits,
            dim=-1
        )


        labels = batch["labels"]


        total_test += labels.size(0)


        correct_test += (
            predictions == labels
        ).sum().item()


# ==================================================
# 24. FINAL TEST RESULTS
# ==================================================

test_loss = (
    running_test_loss
    / len(test_loader)
)


test_accuracy = (
    correct_test
    / total_test
)


print("\n" + "=" * 60)

print(
    "FINAL TEST RESULTS"
)

print("=" * 60)


print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)


print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


print(
    "\nTraining complete!"
)