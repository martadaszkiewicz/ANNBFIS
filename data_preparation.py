import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold, train_test_split


# raw_dataset = pd.read_csv("raw_data.csv")
# # SLR score is an ratio of STV and LTV indices - so it is highly correlated

def specify_output(data: pd.DataFrame, thr: None, ref_column: str):
    reference_cols = ["Percentile","Apgar_1","Apgar5","Ph"]
    
    if ref_column not in reference_cols:
        raise ValueError(f"Provided reference column '{ref_column}' is not valid. Available columns are: {', '.join(reference_cols)}")
    
    output_vector = []
    for record in data[ref_column]:
        if record <= thr:
            output_vector.append(1)
        else:
            output_vector.append(-1)
    data = data.drop(["SLR"], axis=1)   # dropping correlated column
    data = data.drop(reference_cols, axis=1)
    data['target'] = output_vector
    
    return data

def add_fuzzy_scores(data: pd.DataFrame, fuzzy_path: str):
    # funcion should be applied after specifying the output:
    with open(fuzzy_path, 'r') as file:
        fuzzy_scores = file.readlines()
    fuzzy_scores = np.array([float(line.strip()) for line in fuzzy_scores]).reshape(-1,1)

    data['fuzzy_scores'] = fuzzy_scores
    # swaping the columns to get target at the last column (only for clarity)
    col_names = list(data.columns)
    target_index = col_names.index('target')
    fuzzy_scores_index = col_names.index('fuzzy_scores')
    col_names[target_index], col_names[fuzzy_scores_index] = col_names[fuzzy_scores_index], col_names[target_index]

    data = data[col_names]

    return data

def zscore_normalize(data: np.ndarray):
    features = data[:, :-2]
    normalized_features = zscore(features, axis=0)
    
    normalized_data = np.hstack((normalized_features, data[:, -2:]))
    
    return normalized_data

def kfold_cv(data: np.ndarray):
    X = data[:,:-1]
    y = data[:,-1]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)

    # organise the split:
    fold_sets = {}

    for i, (train_index, test_index) in enumerate(skf.split(X,y)):
        X_train = X[train_index]
        y_train = y[train_index]
        current_train_set = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)

        X_test = X[test_index]
        y_test = y[test_index]
        current_test_set = np.concatenate((X_test,y_test.reshape(-1,1)),axis=1)

        fold_sets[i] = (current_train_set, current_test_set)
    
    return fold_sets

def train_val_split(train_set: np.ndarray):
    X = train_set[:, :-1]
    y = train_set[:, -1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2024)

    train_set = np.hstack((X_train, y_train.reshape(-1, 1)))
    val_set = np.hstack((X_val, y_val.reshape(-1, 1)))

    return train_set, val_set

def apply_fuzzy_score(sets: tuple, delta: float=None, k: float = None, target_ratio: float = None):
    train_set, val_set, test_set = sets

    print('\nratio before (normal : abnormal):')
    print(f'{np.sum(train_set[:,-1] == -1)/len(train_set[:,-1]):.4} : {np.sum(train_set[:,-1] == 1)/len(train_set[:,-1]):.4}')

    # correcting the label for cases with high degree of certainty
    if delta != None:
        print(f'Given delta value: {delta}')
        # if i want to assess only abnormal:
        condition_1 = train_set[:,-2] >= delta
        train_set[condition_1,-1] = 1

    # removing the records with low degree of certainty
    if k != None:
        print(f'Given k value: {k}')
        # assess only the normal state (negative fuzzy score):
        condition = (train_set[:, -2] > 0) | (train_set[:, -2] < -k)
        train_set = train_set[condition]

    # duplicating the records with abnormal state to given ratio 
    if target_ratio != None:
        print(f'Given target ratio: {target_ratio}')
        # duplicated samples have to satisfy set level of certainty: 
        condition = (train_set[:,-2] >= 0.5) & (train_set[:,-1] > 0)
        duplicated_rows = train_set[condition]      # creating array with possible rows to replication

        while np.sum(train_set[:,-1] == 1)/len(train_set[:,-1]) < target_ratio:
            # iteratively and randomly adding one record to get closer to the given ratio:
            train_set = np.vstack([train_set,duplicated_rows[np.random.randint(0, len(duplicated_rows[:,0])),:]])

    print('ratio after (normal : abnormal):')
    print(f'{np.sum(train_set[:,-1] == -1)/len(train_set[:,-1]):.4} : {np.sum(train_set[:,-1] == 1)/len(train_set[:,-1]):.4}')

    # removing column with fuzzy scores before the training:
    train_set = np.concatenate((train_set[:, :train_set.shape[1]-2], train_set[:, train_set.shape[1]-1:]), axis=1)
    val_set = np.concatenate((val_set[:, :val_set.shape[1]-2], val_set[:, val_set.shape[1]-1:]), axis=1)
    test_set = np.concatenate((test_set[:, :test_set.shape[1]-2], test_set[:, test_set.shape[1]-1:]), axis=1)

    return train_set, val_set, test_set

def calculate_metrics(real_labels: np.ndarray, output_labels: np.ndarray):
    real_labels = real_labels.reshape(-1,1)
    output_labels = output_labels.reshape(-1,1)

    condition_normal_state = output_labels < 0
    output_labels[condition_normal_state] = -1
    condition_abnormal_state = output_labels >= 0
    output_labels[condition_abnormal_state] = 1

    TP = np.sum((real_labels == 1) & (output_labels == 1))
    TN = np.sum((real_labels == -1) & (output_labels == -1))
    FP = np.sum((real_labels == -1) & (output_labels == 1))
    FN = np.sum((real_labels == 1) & (output_labels == -1))

    # accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    # sensitivity/recall
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    # specificity
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    # precision
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    # f1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print(f'acc: {accuracy:.2f}, recall: {recall:.2f},\nspec: {specificity:.2f}, prec: {precision:.2f}\nf1 score: {f1_score:.2f}')

    return accuracy, recall, specificity, precision, f1_score