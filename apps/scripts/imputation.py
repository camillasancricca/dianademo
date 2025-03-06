import numpy as np
import pandas as pd
from sklearn import linear_model
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from skfuzzy import cmeans, cmeans_predict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from predictors.scripts import utils
from sklearn.neighbors import KNeighborsClassifier

class no_impute:
    def __init__(self):
        self.name = 'No imputation'

    def fit(self, df):
        return df

class impute_standard:
    def __init__(self):
        self.name = 'Standard'

    def fit(self, df, col):
        if (df[col].dtype != "object"):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Missing")
        return df

class drop:
    def __init__(self):
        self.name = 'Drop'

    def fit_cols(self, df):
        df = df.dropna(axis=1, how='any')
        return df

    def fit_rows(self, df):
        df = df.dropna(axis=0, how='any')
        return df

class impute_mean:
    def __init__(self):
        self.name = 'Mean'

    def fit(self, df, col):
        if (df[col].dtype != "object"):
            df[col] = df[col].fillna(df[col].mean())
        return df

    def fit_mode(self, df, col):
        if (df[col].dtype != "object"):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_mode:
    def __init__(self):
        self.name = 'Mode'

    def fit(self, df, col):
        df = df.copy()
        df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_median():
    def __init__(self):
        self.name = 'Median'

    def fit(self, df, col):
        if (df[col].dtype != "object"):
            df[col] = df[col].fillna(df[col].median())
        return df

    def fit_mode(self, df, col):
        if (df[col].dtype != "object"):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_knn():
    def __init__(self):
        self.name = 'KNN'

    def fit(self, df, missing_column, n_neighbors=5):
        type_missing = df.dtypes[missing_column]
        X = df.copy()
        if type_missing in ["int64", "float64"]:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            X = utils.encoding_categorical_variables(X)
            columns = X.columns
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            df_m = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(X)))
            df_m.columns = columns
            df.loc[:, missing_column] = df_m[missing_column]
            return df

        elif type_missing in ["bool","object"]:
            train_columns = list(X.columns)
            train_columns.remove(missing_column)
            target = X[missing_column]

            X = utils.encoding_categorical_variables(X[train_columns])
            train_columns = X.columns
            X[missing_column] = target

            x_train = X.loc[target.notna(),train_columns]
            to_impute = X.loc[target.isna(), train_columns]
            y_train = target[target.notna()]

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            to_impute = scaler.transform(to_impute)

            imputer = KNeighborsClassifier(n_neighbors=n_neighbors)
            x_train = np.nan_to_num(x_train)
            to_impute = np.nan_to_num(to_impute)
            imputer.fit(x_train, y_train)
            df.loc[target.isna(), missing_column] = imputer.predict(to_impute)
            return df

class impute_mice:
    def __init__(self):
        self.name = 'Mice_mine'

    def fit(self, df, missing_column, estimator):
        type_missing = df.dtypes[missing_column]
        X = df.copy()
        if type_missing in ["int64", "float64"]:
            # one hot encoding
            X = utils.encoding_categorical_variables(X)
            columns = X.columns.copy()
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            imputer = IterativeImputer(max_iter=100, skip_complete=True, estimator=estimator)
            X = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(X)), columns=columns)
            df.loc[:,missing_column] = X[missing_column]
            return df

        elif type_missing in ["bool", "object"]:

            # one hot encoding
            fully_available_columns = list(X.columns)
            fully_available_columns.remove(missing_column)
            target = X[missing_column]
            X = utils.encoding_categorical_variables(X[fully_available_columns])
            columns = list(X.columns)
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=columns)
            X[missing_column] = target
            columns.append(missing_column)
            # encode the missing column only for avoiding runtime errors in the IterativeImputer object
            oe = OrdinalEncoder(handle_unknown='use_encoded_value',
                                unknown_value=np.nan)
            oe.fit(X[missing_column].values[:,None])

            X[missing_column] = oe.transform(X[missing_column].values[:,None])

            imputer = IterativeImputer(
                estimator=estimator, max_iter=100,
                initial_strategy="most_frequent", skip_complete=True)

            X = pd.DataFrame(imputer.fit_transform(X), columns=columns)

            columns.remove(missing_column)
            X[columns] = scaler.inverse_transform(X[columns])

            X[missing_column] = X[missing_column].astype('int64')
            X[missing_column] = X[missing_column].astype('str')

            for i in range(0,len(oe.categories_[0])):
                X[missing_column] = X[missing_column].replace({str(i) : oe.categories_[0][i]})

            #X[missing_column] = oe.transform(X[missing_column].values[:,None])
            df.loc[:,missing_column] = X[missing_column]
            return df

class impute_random:
    def __init__(self):
        self.name = 'Random'

    def fit(self, df, col):
        number_missing = df[col].isnull().sum()
        observed_values = df.loc[df[col].notnull(), col]
        df.loc[df[col].isnull(), col] = np.random.choice(observed_values, number_missing, replace=True)
        return df

    def fit_single_column(self, df, col):
        df = df.copy()
        number_missing = df[col].isnull().sum()
        observed_values = df.loc[df[col].notnull(), col]
        df.loc[df[col].isnull(), col] = np.random.choice(observed_values, number_missing, replace=True)
        return df


class impute_linear_regression:
    def __init__(self):
        self.name = 'Linear Regression'

    def fit(self, df, missing_column):
        X = df.copy()
        target = X[missing_column].copy()
        fully_available_columns = list(X.columns)
        fully_available_columns.remove(missing_column)
        X = utils.encoding_categorical_variables(X[fully_available_columns])
        X[missing_column] = target

        # here starts the imputation for the single column
        columns_for_imputation = list(X.columns)
        columns_for_imputation.remove(missing_column)

        features = X[columns_for_imputation]
        target = X[missing_column]

        X_train = features[target.notna()]
        y_train = target[target.notna()]
        mean_y = np.mean(y_train)
        std_y = np.std(y_train)
        y_train = (y_train - mean_y)/std_y

        to_impute = features[target.isna()]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_train = np.nan_to_num(X_train)
        to_impute = scaler.transform(to_impute)
        to_impute = np.nan_to_num(to_impute)
        y_train = np.nan_to_num(y_train)

        imputer = linear_model.LinearRegression()
        imputer.fit(X_train, y_train)

        #X.loc[target.isna(), missing_column] = imputer.predict(to_impute)*std_y + mean_y
        #return X
        df.loc[target.isna(), missing_column] = imputer.predict(to_impute)*std_y+mean_y
        return df


class impute_logistic_regression:
    def __init__(self):
        self.name = 'Logistic Regression'

    def fit(self, df, missing_column, C=1):
        X = df.copy()
        target = X[missing_column].copy()
        fully_available_columns = list(X.columns)
        fully_available_columns.remove(missing_column)
        X = utils.encoding_categorical_variables(X[fully_available_columns])
        X[missing_column] = target

        # here starts the imputation for the single column
        columns_for_imputation = list(X.columns)
        columns_for_imputation.remove(missing_column)

        features = X[columns_for_imputation]
        target = X[missing_column]

        X_train = features[target.notna()]
        y_train = target[target.notna()]

        to_impute = features[target.isna()]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        to_impute = scaler.transform(to_impute)

        imputer = linear_model.LogisticRegression(max_iter=1000, C=C)
        X_train = np.nan_to_num(X_train)
        to_impute = np.nan_to_num(to_impute)

        imputer.fit(X_train, y_train)

        #X.loc[target.isna(), missing_column] = imputer.predict(to_impute)
        #return X
        df.loc[target.isna(), missing_column] = imputer.predict(to_impute)
        return df

class impute_random_forest:
    def __init__(self):
        self.name = 'Random Forest'

    def fit(self, df, missing_column, max_depth=20):
        X = df.copy()
        target = X[missing_column].copy()
        fully_available_columns = list(X.columns)
        fully_available_columns.remove(missing_column)
        X = utils.encoding_categorical_variables(X[fully_available_columns])
        X[missing_column] = target

        # here starts the imputation for the single column
        columns_for_imputation = list(X.columns)
        columns_for_imputation.remove(missing_column)

        features = X[columns_for_imputation]
        target = X[missing_column]

        X_train = features[target.notna()]
        y_train = target[target.notna()]

        to_impute = features[target.isna()]

        type_missing = X.dtypes[missing_column]
        if type_missing == 'int64' or type_missing == 'float64':
            imputer = RandomForestRegressor(max_depth=max_depth)
        else:
            imputer = RandomForestClassifier(max_depth=max_depth)
        imputer.fit(X_train, y_train)

        to_impute = np.nan_to_num(to_impute)
        #X.loc[target.isna(), missing_column] = imputer.predict(to_impute)
        #return X
        df.loc[target.isna(), missing_column] = imputer.predict(to_impute)
        return df

class impute_clustering():
    def __init__(self):
        self.name = 'Clustering'

    def fit_num(self, df, missing_column, n_clusters=5, m=1.5):

        df = df.copy()
        # here starts the imputation for the single column
        features = df.copy()
        target = df[missing_column]
        columns_to_encode = list(features.columns)
        columns_to_encode.remove(missing_column)
        encoded_columns_df = utils.encoding_categorical_variables(features[columns_to_encode])

        X = encoded_columns_df.copy()
        X[missing_column] = target.copy()

        # scale the dataset and cluster the fully available data
        X_train = X[target.notna()]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_train = X_train.T
        centroids,_,_,_,_,_,_ = cmeans(X_train, n_clusters, m=m, error=0.001, maxiter=100000)
        centroid_miss_column_coeff = np.array([centroid[-1] for centroid in centroids])

        # predict cluster attributions for incomplete data
        to_impute = encoded_columns_df.copy()
        to_impute = to_impute[target.isna()]
        # intermediate_value = np.random.choice(centroid_miss_column_coeff,to_impute.shape[0])
        missing_column_mean = np.nanmean(target)
        intermediate_value = [missing_column_mean for i in range(to_impute.shape[0])]
        to_impute[missing_column] = intermediate_value
        to_impute = scaler.transform(to_impute)
        to_impute = to_impute.T
        memberships = cmeans_predict(to_impute, centroids, m=m, error=0.001, maxiter=1000)[0]
        # impute data based on centroids and cluster attributions
        imputed_values = (memberships.T @ centroid_miss_column_coeff)*scaler.scale_[-1] + scaler.mean_[-1]
        df.loc[target.isna(), missing_column] = imputed_values
        return df

    def fit_cat(self, df, missing_column, n_clusters=4):
        # here starts the imputation for the single column
        # X = impute_random().fit_single_column(df.copy(), missing_column)
        X = impute_mode().fit(df.copy(), missing_column)
        cat = list(df.select_dtypes(include=['bool', 'object']).columns)
        num = list(df.select_dtypes(include=['int64', 'float64']).columns)

        missing_column_index = 0
        for i in range(len(cat)):
            if missing_column == cat[i]:
                missing_column_index = i

        cat_indices = [df.columns.get_loc(col) for col in cat]
        for i in range(1):
            if len(num) != 0:
                model = KPrototypes(n_clusters=n_clusters, max_iter=10, init="random")
            else:
                model = KModes(n_clusters=n_clusters, max_iter=10)

            X = pd.DataFrame(X)
            X = utils.encoding_categorical_variables(X)

            for col in X.columns:
                if X[col].dtype in ['int64','float64']:
                    X[col] = X[col].fillna(-1)

            model.fit(X, categorical=cat_indices)
            labels = model.predict(X[df[missing_column].isna()], categorical=cat_indices)

            centroids_values = model.cluster_centroids_[:,len(num)+missing_column_index]

            imputed_values = np.array([centroids_values[label] for label in labels])

            df.loc[df[missing_column].isna(), missing_column] = imputed_values
        return df

def impute(df, method, missing_column):
    imputated_df = pd.DataFrame()
    try:
        if method == "no_impute":
            imputator = no_impute()
            imputated_df = imputator.fit(df)
        elif method == "impute_standard":
            imputator = impute_standard()
            imputated_df = imputator.fit(df, missing_column)
        elif method == "impute_mean":
            imputator = impute_mean()
            imputated_df = imputator.fit_mode(df, missing_column)
        elif method == "impute_mode":
            imputator = impute_mode()
            imputated_df = imputator.fit(df, missing_column)
        elif method == "impute_median":
            imputator = impute_median()
            imputated_df = imputator.fit_mode(df, missing_column)
        elif method == "impute_random":
            imputator = impute_random()
            imputated_df = imputator.fit(df, missing_column)
        elif method == "impute_knn":
            imputator = impute_knn()
            imputated_df = imputator.fit(df, missing_column)
        elif method == "impute_mice":
            imputator = impute_mice()
            if df[missing_column].dtype in ["float64","int64"]:
                imputated_df = imputator.fit(df, missing_column, estimator=BayesianRidge())
            else:
                imputated_df = imputator.fit(df, missing_column, estimator=KNeighborsClassifier())
        elif method == "impute_linear_regression":
            imputator = impute_linear_regression()
            imputated_df = imputator.fit(df, missing_column)
        elif method == "impute_logistic_regression":
            imputator = impute_logistic_regression()
            imputated_df = imputator.fit(df, missing_column)
        elif method == "impute_random_forest":
            imputator = impute_random_forest()
            imputated_df = imputator.fit(df, missing_column)
        elif method == "impute_cmeans":
            imputator = impute_clustering()
            imputated_df = imputator.fit_num(df, missing_column)
        elif method == "impute_kproto":
            imputator = impute_clustering()
            imputated_df = imputator.fit_cat(df, missing_column)
        return imputated_df
    except:
        imputator = impute_mode()
        imputated_df = imputator.fit(df, missing_column)
        return imputated_df

if __name__ == '__main__':
    df = pd.read_csv("../dataset/weather.csv")
    name_class = 'WeatherType'
    selected_features = ['Temperature', 'CloudCover', 'Season', name_class]
    selected_features_only = ['Temperature', 'CloudCover', 'Season']
    df = df[selected_features]
    techniques_completeness = ['impute_standard', 'impute_mean',
                               'impute_median', 'impute_random', 'impute_knn', 'impute_mice',
                               'impute_linear_regression', 'impute_random_forest', 'impute_cmeans']

    techniques_completeness_cat = ['impute_standard', 'impute_mode',
                                   'impute_random', 'impute_knn', 'impute_mice',
                                   'impute_logistic_regression', 'impute_random_forest', 'impute_kproto']
    #df_dirt = d.injection(df, name_class, 0.5, 10, 1)
    #print(df_dirt)
    df_dirt = df

    techniques_completeness_cat_remaining = ['impute_random_forest']

    for imp in techniques_completeness_cat_remaining:
        df_clean = impute(df_dirt[selected_features_only], imp, 'Temperature')
        print(df_clean)

    #print(df['Season'])

    #oe = OrdinalEncoder(handle_unknown='use_encoded_value',
    #                    unknown_value=np.nan)
    #oe.fit(df['Season'].values[:, None])
    #print(oe.categories_)
    #print(oe.transform(df['Season'].values[:, None]))

    ### kprototype non si può usare
