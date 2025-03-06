import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

kb_accuracy = pd.read_csv("../kb/KBA.csv")

techniques = kb_accuracy.technique_accuracy.unique()
perc_outliers = kb_accuracy.percentage_outliers.unique()
objects = kb_accuracy.column_name.unique()
datasets = kb_accuracy.name.unique()

columns = ['name', 'column_name', 'technique_accuracy']

columns_X = ['n_tuples', 'uniqueness', 'min', 'max',
       'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr', 'p_min',
       'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy', 'density',
       'percentage_outliers']

columns_y = 'f1_technique'

def train_accuracy():
    for technique in techniques:
        data = kb_accuracy.copy()

        df = data[(data["technique_accuracy"] == technique)].copy()

        train = df

        X_train = train[columns_X]
        y_train = train[columns_y]

        X_train = StandardScaler().fit_transform(X_train)
        X_train = np.nan_to_num(X_train)

        knn = KNeighborsRegressor(n_neighbors=27, metric='manhattan')
        knn.fit(X_train, y_train)

        #Accuracy Regressor
        pickle.dump(knn, open('models/AR_'+technique, 'wb'))
        #loaded_model = pickle.load(open(filename, 'rb'))

if __name__ == '__main__':
    train_accuracy()
