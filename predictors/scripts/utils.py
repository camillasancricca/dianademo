import pandas as pd
import numpy as np

def encoding_categorical_variables(X):
    def encode(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], dummy_na=True)
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return res

    categorical_columns = list(X.select_dtypes(include=['bool','object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode(X,col)
    return X

def decode_categorical_variables(df_clean, col_name):

    dummy = [col for col in df_clean if col.startswith(col_name)]

    df_clean[col_name] = pd.from_dummies(df_clean[dummy])
    df_clean = df_clean.drop(columns=dummy)
    df_clean = df_clean.replace({col_name+'_nan': np.nan})

    for d in dummy:
        df_clean = df_clean.replace({d: d.replace(col_name+'_', '')})

    return df_clean

def check_datatypes(df):
    for col in df.columns:
        if (df[col].dtype == "bool"):
            df[col] = df[col].astype('string')
            df[col] = df[col].astype('object')
    return df

def dirty_single_column(dataset, column_name, name_class, seed):
    np.random.seed(seed)
    # il metodo usato è solo uniform
    df_pandas = dataset[[column_name]].copy()
    df_list = []

    perc = [0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05]
    for p in perc:
        df_dirt = df_pandas.copy()
        comp = [p, 1 - p]
        df_dirt = check_datatypes(df_dirt)
        for col in df_dirt.columns:

            if col != name_class:
                rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)

                df_dirt.loc[rand == True, col] = np.nan

        # potrei fare qua il cambio da column a dataset
        df_dirt_complete = dataset.copy()
        df_dirt_complete[column_name] = df_dirt[column_name]
        df_list.append(df_dirt_complete)
        # print("saved {}-completeness{}%".format(column_name, round((1 - p) * 100)))
    return df_list
