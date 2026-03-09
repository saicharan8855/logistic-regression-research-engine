import numpy as np
import pandas as pd

def shuffle_data(X,y , seed = None ):
    if X is  None:
        raise ValueError("X is None")
    if y is None:
        raise ValueError("X is None")

    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"rows of X = {X.shape[0]} ,rows of y = {y.shape[0]} not matching")

    m = X.shape[0]
    rng = np.random.default_rng(seed)

    indices = np.arange(m)
    rng.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    return X_shuffled , y_shuffled

def binary_encode_target(y):

    if y is None:
        raise ValueError("y is None")

    if isinstance(y, (list, np.ndarray)):
        y = pd.Series(y)

    if y.dtype == object:
        y = y.astype(str).str.strip()

    if y.isnull().any():
        raise ValueError("target contains missing values")

    unq_labels = y.unique()

    if len(unq_labels) != 2:
        raise ValueError("number of classes must be equal to 2")

    unq_labels = sorted(unq_labels)

    first_label = unq_labels[0]
    second_label = unq_labels[1]

    mapping = {first_label: 1, second_label: 0}

    y_encoded = y.map(mapping).to_numpy()

    return y_encoded

def one_hot_encode(X , categorical_indices , feature_names):
    if X is None:
        raise ValueError("X is None")
        
    if isinstance(X , np.ndarray):
        X = pd.DataFrame(X , columns = feature_names)
        
    if len(feature_names) != X.shape[1]:
        raise ValueError("columns mismatch")
        
    if len(categorical_indices) == 0:
        raise ValueError("categorical indices are empty")
        
    if max(categorical_indices) >= X.shape[1]:
        raise ValueError("categorical index out of range")
        
    categorical_columns = [feature_names[i] for i in categorical_indices]
    numerical_columns = [col for col in feature_names if col not in categorical_columns]

    X_num = X[numerical_columns]
    X_cat = X[categorical_columns]

    X_cat_encoded = pd.get_dummies(X_cat)
    X_encoded_df = pd.concat([X_num , X_cat_encoded] , axis = 1)

    new_feature_names = X_encoded_df.columns.tolist()
    X_encoded = X_encoded_df.to_numpy()

    return X_encoded , new_feature_names

def train_test_split(X , y , test_size = 0.2 , seed = None):
    if X  is None:
        raise ValueError("X is None")
    if y is None:
        raise ValueError("y is None")

    X = np.asarray(X)
    y = np.asarray(y)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("no of rows in X must equals no of rows in  y")

    if not (0 < test_size < 1):
        raise ValueError("test size must be between 0 and 1")

    m = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(m)
    rng.shuffle(indices)

    split_index = int(m*(1 - test_size))

    training_set = indices[:split_index]
    test_set = indices[split_index:]

    X_train = X[training_set]
    X_test = X[test_set]

    y_train = y[training_set]
    y_test = y[test_set]
    
    return X_train , X_test , y_train , y_test

def handle_nulls(X , strategy = "median"):
    if X is None:
        raise ValueError("X is None")
    X = np.asarray(X).copy()

    if strategy not in ["median","mean"]:
        raise ValueError("strategy must be median or mean")

    if strategy == "median":
        col_values = np.nanmedian(X , axis = 0)
    else:
        col_values = np.nanmean(X , axis = 0)

    nan_rows , nan_columns = np.where(np.isnan(X))

    X[nan_rows , nan_columns] = col_values[nan_columns]

    return X

def calculate_vif(X):

    vif_values = []
    if X is None:
        raise ValueError("X is none")
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("dimensions must be equal to 2")
    m,n = X.shape

    for i in range(n):
        y = X[:,i]
        X_others = np.delete(X,i,axis = 1)
        X_others = np.column_stack([np.ones(X_others.shape[0]), X_others])

        X_transpose = X_others.T
        theta = np.linalg.inv(X_transpose @ X_others) @ X_transpose @ y
        y_pred = X_others @ theta

        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        vif = 1/(1 - r_squared)
        vif_values.append(vif) if (1-r_squared) != 0 else 0

    vif_values = np.array(vif_values)

    for i, vif in enumerate(vif_values):
        if vif < 5:
            interpretation = "Good"
        elif vif < 10:
            interpretation = "Moderate"
        else:
            interpretation = "Severe multicollinearity"

        print(f"Feature {i}: VIF = {vif:.2f} ({interpretation})")

    return vif_values

def remove_multicollinear_features(X,feature_names , threshold = 0.9):
    if X is None:
        raise ValueError("X is None")

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match number of columns in X")

    corr_matrix = np.corrcoef(X.T)

    drop_set = set()

    n = corr_matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):

            if abs(corr_matrix[i, j]) > threshold:
                drop_set.add(j)

    keep_indices = [i for i in range(n) if i not in drop_set]

    X_filtered = X[:, keep_indices]

    remaining_feature_names = [feature_names[i] for i in keep_indices]

    return X_filtered, remaining_feature_names

def fit_scaler(X_train):
    
    X_train = np.asarray(X_train)

    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array")
    if X_train.shape[0] == 0:
        raise ValueError("no samples")

    mean = np.mean(X_train , axis = 0)
    std = np.std (X_train , axis = 0)
    std[std==0] = 1

    return mean , std

def transform_scaler(X, mean, std):

    if X is None:
        raise ValueError("X is None")

    if mean is None or std is None:
        raise ValueError("mean and std must be provided")

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2d array")

    if len(mean) != X.shape[1] or len(std) != X.shape[1]:
        raise ValueError("mean/std length must match number of features")

    X_scaled = (X - mean) / std

    return X_scaled

def add_bias_column(X):
    if X is None:
        raise ValueError("X is None")
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2d")

    m = X.shape[0]
    ones = np.ones((m , 1))
    X_with_bias = np.column_stack([ones , X])

    return X_with_bias

def detect_perfect_separation(X, y):

    if X is None:
        raise ValueError("X is None")
    if y is None:
        raise ValueError("y is None")

    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    separating_features = []

    n = X.shape[1]

    for j in range(n):

        feature = X[:, j]

        class0_vals = feature[y == 0]
        class1_vals = feature[y == 1]

        if len(class0_vals) == 0 or len(class1_vals) == 0:
            continue

        max_0 = np.max(class0_vals)
        min_0 = np.min(class0_vals)

        max_1 = np.max(class1_vals)
        min_1 = np.min(class1_vals)

        if max_0 < min_1 or max_1 < min_0:
            separating_features.append(j)

    if separating_features:
        print("warning: perfect separation detected in features:", separating_features)

    return separating_features







    