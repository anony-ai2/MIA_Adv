# README

## Project Overview

This code implements a membership inference attack (MIA) classifier based on feature vectors and neural networks. The main workflow includes:

1. Feature Extraction

   * Read pairs of original text and model predictions from the victim model’s output files (`true_file`/`false_file`) and their corresponding ground-truth files.
   * Extract semantic embeddings via CodeBERT and compute similarity.
   * Compute perplexity using CodeGPT.
   * Calculate the maximum and minimum values of similarity and perplexity.

2. Saving Features

   * Save all samples’ feature vectors and labels into `feature.npz`.

3. Classifier Training and Evaluation

   * Load features from `feature.npz` and split the dataset.
   * Train a multi-layer perceptron (MLP) and report metrics such as accuracy and AUC on the test set.

## Dependencies

```bash
pip install torch transformers numpy scikit-learn
```

## Quick Start

### 1. Data Preparation

Assume you have the following four files, in JSONL or plain text format:

* `train_codes.txt`
* `test_codes.txt`
* `train_labels.json`  (each line is JSON containing the field `{"gt": "..."}`)
* `test_labels.json`

Place them in the same directory, e.g., `/data/MIAdataset/`.

### 2. Feature Extraction and Saving

On the first run, the script will detect that the `.npz` specified by `--feature_path` does not exist and will automatically extract and save features:

```bash
python new_classifier_submit.py \
  --input_dir "/data/MIAdataset" \
  --true_file "train_codes.txt" \
  --false_file "test_codes.txt" \
  --true_gt_file "train_labels.json" \
  --false_gt_file "test_labels.json" \
  --feature_path "feature.npz" \
  --n_samples_per_class 40 \
  --global_random_seed 42 \
  --random_state 123 \
  --random_state_test 456 \
  --dropout 0.1 \
  --batch_size 4 \
  --lr 1e-3 \
  --num_epochs 25 \
  --hidden_dims 512 512 512
```

* `--feature_path`: Path to the feature file; it will be generated on first run.
* `--n_samples_per_class`: Total number of samples to draw per class (member / non-member).
* Adjust other hyperparameters as needed.

### 3. Training and Evaluation

Once the feature file has been generated, the script will:

1. Split the dataset stratified at a 50/50 train/test ratio.
2. Train the MLP (`CustomMLP`) with each hidden layer followed by `ReLU + Dropout`.
3. Output the final test Loss/Accuracy, and for binary classification print AUC, TPR, FPR.

## Script Arguments

| Argument                | Type    | Req. | Default | Description                                                          |
| ----------------------- | ------- | ---- | ------- | -------------------------------------------------------------------- |
| `--input_dir`           | `str`   | Yes  | —       | Directory containing input texts and ground-truth files              |
| `--true_file`           | `str`   | Yes  | —       | Model output file for member (positive) samples                      |
| `--false_file`          | `str`   | Yes  | —       | Model output file for non-member (negative) samples                  |
| `--true_gt_file`        | `str`   | Yes  | —       | Ground-truth JSON file for member samples                            |
| `--false_gt_file`       | `str`   | Yes  | —       | Ground-truth JSON file for non-member samples                        |
| `--feature_path`        | `str`   | Yes  | —       | Path to the feature `.npz` file                                      |
| `--feature_path_test`   | `str`   | No   | `None`  | If specified, features from this file will be re-split and evaluated |
| `--exclude`             | `str`   | No   | `None`  | Indices of feature columns to exclude (Python list format)           |
| `--n_samples_per_class` | `int`   | Yes  | —       | Total number of samples to draw per class                            |
| `--global_random_seed`  | `int`   | Yes  | —       | Global random seed                                                   |
| `--random_state`        | `int`   | Yes  | —       | Random seed for stratified splitting of the feature file             |
| `--random_state_test`   | `int`   | No   | `None`  | Random seed for additional test split                                |
| `--dropout`             | `float` | Yes  | —       | Dropout rate for MLP hidden layers                                   |
| `--batch_size`          | `int`   | Yes  | —       | Training batch size                                                  |
| `--lr`                  | `float` | Yes  | —       | Learning rate                                                        |
| `--num_epochs`          | `int`   | Yes  | —       | Number of training epochs                                            |
| `--hidden_dims`         | `int`…  | Yes  | —       | Hidden layer sizes, e.g., `--hidden_dims 512 512 512`                |

## Sample Output

```text
Saved features to feature.npz: X.shape=(240, 14), y.shape=(240,)

Final Test Loss: 0.1054, Test Accuracy: 90.00%
=== Binary classification metrics ===
Confusion matrix: TN=46, FP=4, FN=6, TP=44
TPR (Recall)=88.00%   FPR=8.00%
AUC: 0.9623
```
