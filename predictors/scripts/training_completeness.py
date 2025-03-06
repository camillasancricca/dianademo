import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

kb_completeness = pd.read_csv("../kb/KBC.csv")

cols = ['name', 'column_name', 'n_tuples', 'missing_perc', 'uniqueness', 'min',
       'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr',
       'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
       'density', 'ml_algorithm', 'impute_standard_impact', 'impute_mean_impact', 'impute_median_impact',
       'impute_random_impact', 'impute_knn_impact', 'impute_mice_impact',
       'impute_linear_regression_impact', 'impute_random_forest_impact',
       'impute_cmeans_impact']

datasets = kb_completeness.name.unique()
objects = kb_completeness.column_name.unique()
ml_algorithms = kb_completeness.ml_algorithm.unique()

columns_X = ['n_tuples', 'uniqueness', 'min',
       'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr',
       'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
       'density', 'missing_perc']

techniques = ['impute_standard', 'impute_mean',
       'impute_median', 'impute_random', 'impute_knn', 'impute_mice',
       'impute_linear_regression', 'impute_random_forest', 'impute_cmeans']

def get_kb_completeness():

    kb_ = kb_completeness.drop_duplicates()

    return kb_

def get_kb_impact_completeness():

    kb_ = get_kb_completeness()

    kb_new = kb_.copy()

    kb_new = kb_new.drop_duplicates()

    ### impact = 1-df_clean/df_standard_value

    for tech in techniques:
        kb_new[tech + '_impact'] = 1 - kb_new[tech] / kb_new['impute_standard']

    kb_new = kb_new[cols]

    for tech in techniques:
        kb_new = kb_new.rename(columns={tech+'_impact': tech})

    return kb_new

def training_completeness():

    kb_completeness = get_kb_impact_completeness()

    for model in ml_algorithms:
        for technique in techniques:

            data = kb_completeness.copy()

            df = data[(data["ml_algorithm"] == model)].copy()

            train = df

            X_train = train[columns_X]
            y_train = train[technique]

            X_train = StandardScaler().fit_transform(X_train)
            X_train = np.nan_to_num(X_train)

            knn = KNeighborsRegressor(n_neighbors=35, metric='cosine')
            knn.fit(X_train, y_train)

            #Completeness Regressor
            pickle.dump(knn, open('models/CR_'+technique+'_'+model, 'wb'))
            #loaded_model = pickle.load(open(filename, 'rb'))

if __name__ == '__main__':
    training_completeness()
