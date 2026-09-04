# Model Training Progress Log

## Overview

This document tracks the development and performance of the model across multiple training tests. Each test represents a different iteration of the training process, with improvements made to dataset handling, validation, optimization, and evaluation.

## Initial Training — First Pass

### Epoch 3/3

- **Average Training Loss:** 0.7836
- **Training Accuracy:** 68.80%
- **Validation Accuracy:** 48.00%

### Epoch 4/4 — Second Pass

- **Average Training Loss:** 0.3273
- **Training Accuracy:** 90.60%
- **Validation Accuracy:** 60.10%
- **Test Accuracy:** 59.50%

The initial results showed significant overfitting, with training accuracy substantially higher than validation and test accuracy. This led to several changes to improve the training and evaluation process.

## Changes Made

1. **Separate Dataset Splits**  
   The dataset was divided into three separate groups:

   - Training set — used to train the model.
   - Validation set — used to monitor performance during training.
   - Final test set — kept completely unseen until training was finished.

2. **Validation Loss Tracking**  
   Validation loss was added alongside validation accuracy. Tracking loss provides a more detailed measurement of whether the model is improving and allows the best-performing model to be selected more reliably.

3. **Early Stopping**  
   Early stopping was changed to monitor validation loss rather than only accuracy. If validation loss stops improving for a specified number of epochs, training is stopped to reduce overfitting and unnecessary computation.

4. **Best Model Saving**  
   The model is saved whenever validation loss reaches a new minimum. After training finishes, the best saved model is reloaded before final testing.

5. **Gradient Accumulation**  
   Gradient accumulation was implemented to simulate a larger batch size.

   - Actual batch size: 4
   - Effective batch size: approximately 16

6. **Longer Input Length**  
   The maximum input length was increased from 128 to 256 tokens. This allows the model to process more of each text sample and retain additional context.

7. **Final Testing**  
   A separate final evaluation stage was added. The final test set remains unseen during training and validation, providing a better estimate of model performance on new data.

8. **Scheduler Correction**  
   The learning-rate scheduler was adjusted so that its number of training steps correctly accounts for gradient accumulation.

9. **Removed Unnecessary Import**  
   The unnecessary import below was removed:

   ```python
   from sched import scheduler
   ```

## Experiment Results

| Test | Training Accuracy | Validation Accuracy | Validation Loss | Test Accuracy | Test Loss |
|:---|---:|---:|---:|---:|---:|
| Test 2 | 91.40% | 89.80% | 0.2875 | 90.10% | 0.3162 |
| Test 3 | 92.70% | 91.60% | 0.2641 | 90.90% | 0.3019 |
| Test 4 | 96.80% | 95.40% | 0.1186 | 95.20% | 0.1324 |
| Test 5 | 93.10% | 90.90% | 0.2318 | 90.50% | 0.2554 |
| Test 6 | 99.83% | 98.33% | 0.0713 | 91.50% | 0.2600 |
| Test 7 | 94.20% | 92.70% | 0.1764 | 90.17% | 0.2187 |
| Test 8 | 95.60% | 98.60% | 0.0877 | 92.40% | 0.1942 |
| Test 9 | 99.65% | 97.90% | 0.0648 | 93.80% | 0.1516 |
| Test 10 | 80.40% | 91.20% | 0.2093 | 90.30% | 0.2745 |

## Key Results

### Best Test Accuracy

- **Test:** Test 4
- **Test Accuracy:** 95.20%
- **Test Loss:** 0.1324

### Best Validation Accuracy

- **Test:** Test 8
- **Validation Accuracy:** 98.60%
- **Validation Loss:** 0.0877

### Highest Training Accuracy

- **Test:** Test 6
- **Training Accuracy:** 99.83%
- **Validation Accuracy:** 98.33%
- **Validation Loss:** 0.0713
- **Test Accuracy:** 91.50%

### Overall Test Accuracy Range

The completed tests achieved test accuracies ranging from **90.10% to 95.20%**. Every listed test achieved a test accuracy above 90%, with Test 4 producing the strongest final result at 95.20%.

## Observations

The results show a significant improvement compared with the initial training runs. The first pass produced only 59.50% test accuracy, while the later experiments consistently achieved test accuracies above 90%.

The high training accuracies in Tests 6 and 9 demonstrate that the model can fit the training data extremely well. However, the difference between training accuracy and test accuracy in Test 6 suggests that some degree of overfitting remains.

The addition of validation loss tracking, early stopping, best-model checkpointing, gradient accumulation, and a separate final test set provides a more reliable and controlled training process.

## Current Best Results

| Metric | Best Result | Test |
|:---|---:|:---|
| Training Accuracy | 99.83% | Test 6 |
| Validation Accuracy | 98.60% | Test 8 |
| Validation Loss | 0.0713 | Test 6 |
| Test Accuracy | 95.20% | Test 4 |
| Test Loss | 0.1324 | Test 4 |

## Conclusion

The model development process produced substantial improvements. The initial test accuracy of 59.50% increased to a best recorded test accuracy of 95.20%, and every completed test in the final experiment set achieved more than 90% test accuracy.

The current training pipeline also provides stronger safeguards against overfitting and data leakage through separate dataset splits, validation-loss monitoring, early stopping, best-model checkpointing, and final evaluation on an unseen test set.