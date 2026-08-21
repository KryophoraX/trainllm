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


logits = [0.2, -0.4, 1.8, 0.5, 0.1]
prediction = 2


`
## Evaluation loop

`
model.eval()


This switches BERT to evaluation mode, including disabling training-specific behavior such as dropout.

with torch.no_grad():

This tells PyTorch not to calculate or retain gradients. The model is not learning during evaluation, so disabling gradient tracking reduces memory use and computation. 

Inside the evaluation loop:


outputs = model(**batch)
predictions = torch.argmax(outputs.logits, dim=-1)


The model predicts rating classes but does **not** call `loss.backward()` or `optimizer.step()`, so weights remain unchanged.



