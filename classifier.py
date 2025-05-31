import argparse
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve, auc, confusion_matrix, recall_score, log_loss
from sklearn.preprocessing import StandardScaler


if not os.path.exists("feature.npz"):
    # Initialize the pre-trained embedding model (e.g., CodeBERT)
    MODEL_NAME = "microsoft/codebert-base"
    model = AutoModel.from_pretrained(MODEL_NAME, cache_dir='/root/.cache/huggingface/hub/')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir='/root/.cache/huggingface/hub/')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # move model to GPU

    codegpt_tokenizer = GPT2Tokenizer.from_pretrained("microsoft/CodeGPT-small-py", cache_dir='/root/.cache/huggingface/hub/')
    codegpt_model = GPT2LMHeadModel.from_pretrained("microsoft/CodeGPT-small-py", cache_dir='/root/.cache/huggingface/hub/').to(device)

    codegpt_model.eval()


class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.5):
        """
        hidden_dims: a list of the width of every hidden layers
        e.g. [128, 64, 32] represent three hidden layers, with widths of 128, 64 and 32
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        # 最后一层映射到输出
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_mlp(model: nn.Module,
              X_train: torch.Tensor,
              y_train: torch.Tensor,
              X_val: torch.Tensor = None,
              y_val: torch.Tensor = None,
              device: torch.device = None,
              batch_size: int = 2,
              lr: float = 1e-3,
              num_epochs: int = 50):
    """
    basic training function, with optional evaluating dataset output.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # === instantiate LR scheduler ===
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(loader.dataset)

        msg = f"Epoch {epoch}/{num_epochs}  train_loss={avg_loss:.4f}"
        if X_val is not None and y_val is not None:
            val_loss, val_acc = evaluate_mlp(model, X_val, y_val, device)
            msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4%}"
        # print(msg)
        # === update learning rate fater every epoch===
        scheduler.step()


def evaluate_mlp(model: nn.Module,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 device: torch.device = None):
    """
    calculate average cross entropy loss and accuracy
    """
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        loss = F.nll_loss(probs.log(), y).item()
        preds = probs.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return loss, acc


def report_binary_metrics(model: nn.Module,
                          X: torch.Tensor,
                          y: torch.Tensor,
                          device: torch.device = None):
    """
    (only binary classify)print confusion matrix, TPR, FPR, AUC.
    """
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        probs = F.softmax(model(X), dim=1)[:, 1].cpu().numpy()
        y_true = y.cpu().numpy()

    # confusion matrix T N | F P
    y_pred = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)


    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        # 1) check input
        if torch.isnan(X).any():
            # assume X is a Tensor with shape=(N, D)
            nan_mask = torch.isnan(X)                  # boolean matrices of the same dimension, with True at NaN
            nan_indices = torch.nonzero(nan_mask)      # return the (row, col) coordinates of all NaN elements

            print(f"find {nan_indices.size(0)} NaN(s):")
            for row, col in nan_indices:
                print(f"  •  {row.item()} row，{col.item()} column: ({X[row, col].item()}) item")
            raise ValueError("input X contains NaN!")
        logits = model(X)
        # 2) check model output
        if torch.isnan(logits).any():
            raise ValueError("model output logits contains NaN!")
        probs = F.softmax(logits, dim=1)[:, 1]
        # 3) check probabilities after softmax
        if torch.isnan(probs).any():
            raise ValueError("probabilities after softmax contains NaN!")
        probs = probs.cpu().numpy()
        y_true = y.cpu().numpy()


    # ROC / AUC
    fpr_arr, tpr_arr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr_arr, tpr_arr)

    # for RQ4:
    # print(f"AUC: {roc_auc:.4f}")
    # return roc_auc

    print("=== Binary classification metrics ===")
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"TPR (Recall): {tpr:.4f}   FPR: {fpr:.4f}")
    print(f"AUC: {roc_auc:.4f}")

    return roc_auc

def extract_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)  # calculate with GPU
    return outputs.squeeze(0).cpu().numpy()  # convert to CPU then convert to NumPy


def compute_similarity(x, y):
    # print(x, y)
    sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    v_x = extract_embeddings(x)
    v_y = extract_embeddings(y)
    return sim(v_x, v_y)


def calculate_perplexity(text, model, tokenizer):
    """
    exp(loss)
    """
    if text == "":
        return 100.0
    max_position_embeddings = model.config.max_position_embeddings
    inputs = tokenizer.encode(text)
    if len(text) > max_position_embeddings:
        inputs = inputs[-max_position_embeddings:]
    input_ids = torch.tensor(inputs).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        # 2) check input_ids
        if input_ids.numel() == 0:
            raise ValueError("The input has been truncated to an empty sequence! Please check the max_length or the text itself.")
        if torch.isnan(input_ids).any():
            raise ValueError("input_ids contains Nan!")

        # 3) forward calculation
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # HF new version uses output.loss and outputs.logits
        logits = outputs.logits

        # 4) checkout loss / logits
        if torch.isnan(loss) or torch.isinf(loss):
            return 1e6
            print(f"‼️ loss exception：{loss.item()}")
            # print logits distribution
            print(f"logits min/max: {logits.min().item()}/{logits.max().item()}")
            raise ValueError("NaN/Inf appear when calculate loss!")

        # 5) calculate perplexity
        ppl = torch.exp(loss)
        if torch.isinf(ppl) or torch.isnan(ppl):
            print(f"‼️ perplexity exception：exp({loss.item()}) = {ppl.item()}")
            raise ValueError("exp(loss) gets NaN/Inf！")

        return ppl.cpu().item()

    outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss).cpu().numpy()


def compute_features(y, y_pred, y_preds_perturbed):
    """Compute features for MIA classification."""
    sim = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    v_y = extract_embeddings(y)
    v_y_pred = extract_embeddings(y_pred)
    v_y_preds_perturbed = [extract_embeddings(y_p) for y_p in y_preds_perturbed]

    similarities = [sim(v_y, v_y_p) for v_y_p in v_y_preds_perturbed]

    base_sim = compute_similarity(y, y_pred)
    perturbed_sims = [compute_similarity(y, y_p) for y_p in y_preds_perturbed]
    # perturbed_sims = [compute_similarity(y_pred, y_p) for y_p in y_preds_perturbed]
    all_sims = [base_sim] + perturbed_sims
    sim_stats = [
        np.mean(all_sims),    # average
        np.std(all_sims)     # standard deviation
    ]
    base_ppl = calculate_perplexity(y_pred, codegpt_model, codegpt_tokenizer)
    perturbed_ppls = [calculate_perplexity(py, codegpt_model, codegpt_tokenizer) for py in y_preds_perturbed]
    # all_ppls = [base_ppl] + perturbed_ppls
    all_ppls = perturbed_ppls
        
    all_ppls = [(ppl - base_ppl) / base_ppl for ppl in all_ppls]

    ppl_stats = [
        np.mean(all_ppls),    # average
        np.std(all_ppls)     # standard deviation
    ]

    feature_vector = (
        all_sims +      
        sim_stats 
        +     
        all_ppls +
        ppl_stats
    )

    return feature_vector


# 分层精确划分函数
def stratified_split(X, y, test_size=0.5, n_samples_per_class=None, random_state=19260817):
    """
    Improved Layered Partition Function
        
    Parameters:
    X: Feature Matrix
    Y: Label vector
    Test_2: Test set ratio (default 0.5)
    N_samples_per_class: The total number of samples for each category (default is to use the minimum number of categories)
    Random_date: Random Seed
        
    return:
    X_train, X_test, y_train, y_test
    """
    
    # 设置随机种子
    np.random.seed(random_state)
    
    # 分离两类样本索引
    class1_idx = np.where(y == 1)[0]
    class0_idx = np.where(y == 0)[0]

    # for _ in range(0, 3):
    #     print(X[class0_idx[_]])
    # for _ in range(0, 3):
    #     print(X[class1_idx[_]])
    
    # Determine the final number of samples for each category, and the default value for n_samples should be 40% of len (class1_idx)
    if n_samples_per_class is None:
        # n_samples = min(len(class1_idx), len(class0_idx))
        # n_samples = math.ceil(len(class0_idx) * 0.2 * 2)
        n_samples = 40
    else:
        n_samples = min(n_samples_per_class, len(class1_idx), len(class0_idx))
    
    # print(len(class0_idx), n_samples)

    # Randomly select a specified number of samples (without replacement)
    selected_class1 = np.random.choice(class1_idx, n_samples, replace=False)
    selected_class0 = np.random.choice(class0_idx, n_samples, replace=False)
    
    # Calculate the number of training samples for each class
    train_per_class = int(n_samples * (1 - test_size))
    
    # Generate random permutation
    perm_class1 = np.random.permutation(n_samples)
    perm_class0 = np.random.permutation(n_samples)
    
    # Partition index
    train_idx = np.concatenate([
        selected_class1[perm_class1[:train_per_class]],
        selected_class0[perm_class0[:train_per_class]]
    ])
    
    test_idx = np.concatenate([
        selected_class1[perm_class1[train_per_class:]],
        selected_class0[perm_class0[train_per_class:]]
    ])
    
    # shuffle
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    # Strictly verify sample balance
    def check_balance(indices, stage):
        counts = np.bincount(y[indices])
        if len(counts) < 2 or counts[0] != counts[1]:
            raise ValueError(f"{stage} imbalace: label 0 counts {counts[0]}, label 1 counts {counts[1]}")
    
    check_balance(train_idx, "train")
    check_balance(test_idx, "test")
    
    # print(len(X[train_idx]), len(X[test_idx]), len(y[train_idx]), len(y[test_idx]))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def load_txt(file_path):
    """Load data from a JSON lines file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

# Load data from JSON files
def load_json(file_path):
    """Load data from a JSON lines file."""
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]


# -------------------------
# 1) Original feature calculation and preservation
# -------------------------


group_size = 12
def pair_data(true_data, true_gt_data):
    paired_data = []
    grouped_lines = [true_data[i:i + group_size] for i in range(0, len(true_data), group_size)]
    
    if len(true_gt_data) != len(grouped_lines):
        raise ValueError(f"Mismatched lengths: f{len(true_gt_data)} and f{len(grouped_lines)}")

    for input, output in zip(true_gt_data, grouped_lines):
        paired_data.append((input['gt'], output[0], output[1:]))
    return paired_data


def set_seed(seed=42):
    random.seed(seed)                      # Python inline random
    np.random.seed(seed)                   # NumPy random
    torch.manual_seed(seed)                # PyTorch random on CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   # PyTorch random on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_features_and_labels(true_file, false_file, true_gt_file, false_gt_file,
                             feature_path="features.npz"):
    # （load_txt/load_json/pair_data/compute_features）
    true_data = load_txt(true_file)
    false_data = load_txt(false_file)
    true_gt_data = load_json(true_gt_file)[::12]
    false_gt_data = load_json(false_gt_file)[::12]

    member_data   = pair_data(true_data,  true_gt_data)
    non_member_data = pair_data(false_data, false_gt_data)

    data = []
    labels = []
    for x, y, y_perturbed in member_data:
        data.append(compute_features(x, y, y_perturbed))
        labels.append(1)
    for x, y, y_perturbed in non_member_data:
        data.append(compute_features(x, y, y_perturbed))
        labels.append(0)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # save to single .npz file
    np.savez(feature_path, X=data, y=labels)
    # print(f"Saved features to {feature_path}: X.shape={data.shape}, y.shape={labels.shape}")


# -------------------------
# 2) Load features and train/report
# -------------------------


def run_training_on_saved_features(feature_path="features.npz", feature_path_test=None, exclude=None, random_state=None, random_state_2=None, n_samples_per_class=None, args=None):
    # load
    arr = np.load(feature_path)
    X = arr["X"]

    # print example
    # for i in range(0, 3):
    #     print(X[0])

    if exclude != None:
        # for RQ2-1:
        if len(exclude) == 1:
            indices = list(range(0, 27))
            exclude.sort(reverse=True)
            for i in exclude:
                indices.pop(i)
            X = X[:, indices]

        # for RQ2-2 in humaneval
        if len(exclude) > 1:
            ex = set(exclude)
            # ex.update({14, 15, 16})

            block1 = [i for i in range(12) if i not in ex]
            block2 = [i for i in range(14, 25) if i not in ex]

            means1 = X[:, block1].mean(axis=1)
            stds1  = X[:, block1].std(axis=1)
            means2 = X[:, block2].mean(axis=1)
            vars2  = X[:, block2].var(axis=1)

            X[:, 12] = means1
            X[:, 13] = stds1
            X[:, 25] = means2
            X[:, 26] = vars2

        keep = [i for i in range(X.shape[1]) if i not in ex]

        X = X[:, keep]


    y = arr["y"]
    # print(f"Loaded X.shape={X.shape}, y.shape={y.shape}")


    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.5, n_samples_per_class=n_samples_per_class, random_state=random_state)

    if feature_path_test != None:
        arr_test = np.load(feature_path_test)
        X_test = arr_test["X"]

        if exclude:
            # Also exclude corresponding columns from the test set
            X_test = X_test[:, indices]

        y_test = arr_test["y"]
        _, X_test, __, y_test = stratified_split(X_test, y_test, test_size=0.5, n_samples_per_class=len(X_test), random_state=random_state_2)

    # ==== Standardization of features ====
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    # (If you want to reuse the scaler in the future, you can save it using joblib.dump (scaler,'scaler. pkl ')


    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t  = torch.from_numpy(X_test)
    y_test_t  = torch.from_numpy(y_test)

    # hyper-parameters
    input_dim   = X_train.shape[1]
    hidden_dim  = 1024
    num_classes = len(np.unique(y))
    # dropout     = 0.1
    dropout     = args.dropout
    # batch_size  = 4
    batch_size  = args.batch_size
    # lr = 1e-3
    lr = args.lr
    # num_epochs  = 25
    num_epochs  = args.num_epochs

    print(args.hidden_dims)
    model = CustomMLP(
        input_dim=input_dim,
        # hidden_dims=[512, 512, 512],
        hidden_dims=args.hidden_dims,
        num_classes=num_classes,
        dropout=dropout
    )

    # Training (including validation loss printing; here we directly use the test set for simple "validation")
    train_mlp(
        model,
        X_train_t, y_train_t,
        X_val=X_test_t, y_val=y_test_t,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs
    )

    # Finally report on the test set
    test_loss, test_acc = evaluate_mlp(model, X_test_t, y_test_t)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4%}")

    # if is binary classfy, prints TPR/FPR/AUC
    if num_classes == 2:
        return report_binary_metrics(model, X_test_t, y_test_t)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        help="The input directory path.")
    parser.add_argument("--true_file",
                        default=None,
                        type=str,
                        help="The file contains input code from the training dataset of victim mode.")
    parser.add_argument("--false_file",
                        default=None,
                        type=str,
                        help="The file contains input code from the testing dataset of victim mode.")
    parser.add_argument("--true_gt_file",
                        default=None,
                        type=str,
                        help="The file contains ground truth from the training dataset of victim mode.")
    parser.add_argument("--false_gt_file",
                        default=None,
                        type=str,
                        help="The file contains ground truth from the testing dataset of victim mode.")

    parser.add_argument("--feature_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The file contains feature vectors. If the file is not exist, it will be filled with feature vectors of samples of true_file and false_file.")
    parser.add_argument("--feature_path_test",
                        default=None,
                        type=str,
                        help="The file contains feature vectors which will be used to evaluate. If it is None, half of samples in feature_path will be used to evaluate.")
    parser.add_argument("--exclude",
                        default=None,
                        type=str,
                        help="The indices of feature vectors which will be ignored.")
    parser.add_argument("--n_samples_per_class",
                        default=None,
                        type=int,
                        required=True,
                        help="The size of the training and testing dataset of classifier.")
    parser.add_argument("--global_random_seed",
                        default=None,
                        type=int,
                        required=True,
                        help="The global random seed.")
    parser.add_argument("--random_state",
                        default=None,
                        type=int,
                        required=True,
                        help="The random seed of splitting data in feature_path.")
    parser.add_argument("--random_state_test",
                        default=None,
                        type=int,
                        help="The random seed of splitting data in feature_path_test.")

    parser.add_argument("--dropout",
                        default=None,
                        type=float,
                        required=True,
                        help="The dropout of classifier.")
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True,
                        help="The batch size of classifier.")
    parser.add_argument("--lr",
                        default=None,
                        type=float,
                        required=True,
                        help="The learning rate of classifier.")
    parser.add_argument("--num_epochs",
                        default=None,
                        type=int,
                        required=True,
                        help="The epochs of classifier.")
    parser.add_argument("--hidden_dims",
                        default=None,
                        nargs="+",
                        type=int,
                        required=True,
                        help="The hidden dims of classifier.")


    args = parser.parse_args()
    set_seed(args.global_random_seed)
    if not os.path.exists(args.feature_path):
        save_features_and_labels(
            os.path.join(args.input_dir, args.true_file), os.path.join(args.input_dir, args.false_file),
            os.path.join(args.input_dir, args.true_gt_file), os.path.join(args.input_dir, args.false_gt_file),
            feature_path=args.feature_path
        )
    
    run_training_on_saved_features(args.feature_path, args.feature_path_test, args.exclude, args.random_state, args.random_state_test, args.n_samples_per_class, args)


if __name__ == '__main__':
    main()