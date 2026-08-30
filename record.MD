Epoch [3/3], Average Loss: 0.7836, Training Accuracy: 68.80%
Validation Accuracy: 48.00%

First pass through of main.py

Epoch [4/4], Average Loss: 0.3273, Training Accuracy: 90.60%
Validation Accuracy: 60.10%
Test Accuracy:59.50%

Second pass through of py

Small list of what I changed:

Separate datasets: Split the data into training, validation, and final test sets instead of using the test set for validation.
Validation loss: Added validation loss tracking, not just accuracy.
Early stopping: Changed early stopping to use validation loss.
Best model saving: Saves the model whenever validation loss improves and reloads it after training.
Gradient accumulation: Simulates a larger batch size of about 16 while keeping the actual batch size at 4.
Longer input length: Changed max_length from 128 → 256.
Final testing: Added a separate evaluation on unseen test data.
Fixed scheduler steps: Adjusted the scheduler to work correctly with gradient accumulation.
Removed unnecessary import: Removed from sched import scheduler.