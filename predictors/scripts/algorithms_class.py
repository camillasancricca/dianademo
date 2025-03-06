import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def encoding_categorical_variables(X):
    def encode(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], dummy_na=False)
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return (res)

    categorical_columns=list(X.select_dtypes(include=['bool','object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode(X,col)
    return X

def classification(X, y, classifier, param, n_splits):

    X = encoding_categorical_variables(X)
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X)
    clf = DecisionTreeClassifier()
    if classifier == "DecisionTree":
        clf = DecisionTreeClassifier(max_depth=int(param))
    elif classifier == "LogisticRegression":
        clf = LogisticRegression(C=param)
    elif classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=int(param))
    elif classifier == "RandomForest":
        clf = RandomForestClassifier(max_depth=int(param))
    elif classifier == "AdaBoost":
        clf = AdaBoostClassifier(n_estimators=int(param))
    elif classifier == "SVC":
        clf = SVC(max_iter=300, C=param)

    print("Training for "+classifier+"...")
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.3)
    model_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")
    f1_median = np.median(model_scores)
    print(f1_median)
    return f1_median
