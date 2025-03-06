import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

data_impact = pd.read_csv("../kb/KBR.csv")

dimensions = ['accuracy', 'completeness']

datasets_fd = ["BachChoralHarmony", "bank", "cancer", "mushrooms", "soybean"]
models = ['DecisionTree','LogisticRegression','KNN','RandomForest','AdaBoost','SVC']

def train():
    for model in models:
        for dimension in dimensions:
            data = data_impact.copy()

            df = data[(data["model"] == model) & (data["dimension"] == dimension)].copy()

            train = df

            columns = df.columns
            features = columns.drop(
                ["name", "dimension", "model", "score", "impact", "p_correlated_features_0.5",
                 "p_correlated_features_0.6", "p_correlated_features_0.7", "p_correlated_features_0.8",
                 "p_correlated_features_0.9"])

            print(features)

            X_train = train[features]
            y_train = train["impact"]

            X_train = StandardScaler().fit_transform(X_train)
            X_train = np.nan_to_num(X_train)

            knn = KNeighborsRegressor(n_neighbors=14, metric='manhattan')
            knn.fit(X_train, y_train)

            # Ranking Regressor
            pickle.dump(knn, open('models/RR_' + model + '_' + dimension, 'wb'))

if __name__ == '__main__':
    train()
