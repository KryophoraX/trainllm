This program fine-tunes a pretrained **BERT** language model to predict a Yelp review’s rating class: labels `0`–`4`, representing the five Yelp star categories. 



dataset = load_dataset("Yelp/yelp_review_full")


This loads Yelp reviews into two main splits:

- `dataset["train"]`: examples used to teach the model
- `dataset["test"]`: examples reserved for evaluating it


print(dataset["train"][100])


Prints one example. 



## Smaller datasets

```python
small_train_dataset = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(1000))
)
```

This creates a 1,000-review training subset.

- `.shuffle(seed=42)` randomizes the original order.
- The seed makes the random order repeatable.
- `.select(range(1000))` keeps 1,000 reviews.

You create an independent 1,000-review evaluation subset the same way from `dataset["test"]`. Because those are separate reviews, evaluation better estimates whether the model can generalize beyond the reviews it trained on.

## Tokenization

BERT does not read normal Python strings. It expects integer token IDs.


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


This loads the exact tokenizer designed for the pretrained `bert-base-cased` model.


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True
    )

For each review, the tokenizer:

- Splits text into BERT-compatible tokens.
- Truncates overly long reviews to BERT’s maximum supported sequence length.



## BERT model


This loads pretrained BERT plus a new final classification layer with five output values.



## DataLoader setup

python
small_train_dataset = small_train_dataset.remove_columns(["text"])
small_eval_dataset = small_eval_dataset.remove_columns(["text"])


After tokenization, the raw `text` column is no longer needed by BERT. 



python
small_train_dataset.set_format("torch")
small_eval_dataset.set_format("torch")


This turns dataset values into PyTorch tensors, which PyTorch models can use directly.


train_loader = DataLoader(
    small_train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)


This produces shuffled training batches of eight reviews.


eval_loader = DataLoader(
    small_eval_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator
)


This produces evaluation batches of eight reviews.



optimizer = AdamW(model.parameters(), lr=2e-5)

`AdamW` changes the model’s weights to reduce prediction error.

- `model.parameters()` gives the optimizer all trainable BERT and classifier-layer parameters.
- `lr=2e-5` is the learning rate: a small update size commonly used for BERT fine-tuning.


num_epochs = 3


The model trains on the full 1,000-review training subset three times.



Because the labels are included, the sequence-classification model computes cross-entropy loss automatically.


loss = outputs.loss
logits = outputs.logits


- `loss`: a single number measuring prediction error for the batch
- `logits`: five raw prediction scores per review


loss.backward()
optimizer.step()


These two lines perform learning:

1. `loss.backward()` calculates how each trainable parameter contributed to the error.
2. `optimizer.step()` adjusts parameters to make similar predictions more accurate next time.

```python
predictions = torch.argmax(logits, dim=-1)
```

This chooses the index of the largest score for every review. For example:

```python
logits = [0.2, -0.4, 1.8, 0.5, 0.1]
prediction = 2
```

```python
correct_train += (predictions == labels).sum().item()
total_train += labels.size(0)
```

These count correctly predicted reviews and total processed reviews, enabling accuracy calculation:

```python
train_accuracy = correct_train / total_train
```

```python
if batch_number % 10 == 0:
```

This prints a progress update every 10 batches. It does not change training; it simply reassures you that the code is actively running.

## Evaluation loop

```python
model.eval()
```

This switches BERT to evaluation mode, including disabling training-specific behavior such as dropout.

```python
with torch.no_grad():
```

This tells PyTorch not to calculate or retain gradients. You are not learning during evaluation, so disabling gradient tracking reduces memory use and computation. `model.eval()` and `torch.no_grad()` do different jobs, which is why the program uses both. [stackoverflow](https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)

Inside the evaluation loop:

```python
outputs = model(**batch)
predictions = torch.argmax(outputs.logits, dim=-1)
```

The model predicts rating classes but does **not** call `loss.backward()` or `optimizer.step()`, so weights remain unchanged.

Finally:

```python
eval_accuracy = correct_eval / total_eval
print(f"Validation Accuracy: {eval_accuracy * 100:.2f}%")
```

This reports the percentage of unseen Yelp test reviews classified into the correct rating category.

For example:

```text
Epoch [1/3], Average Loss: 1.5321, Training Accuracy: 32.10%
Validation Accuracy: 35.40%
```

This means that after the first pass through 1,000 training reviews, the model predicted approximately 32.1% of training labels and 35.4% of held-out evaluation labels correctly.