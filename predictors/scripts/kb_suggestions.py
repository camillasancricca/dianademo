import pickle
import pandas as pd
from predictors.scripts import numerical
import numpy as np
from predictors.scripts import data_profile_extraction as dp
import warnings
warnings.filterwarnings("ignore")

path = '/Users/camillasancricca/PycharmProjects/diana-demo/predictors/scripts/'
path_kb = '/Users/camillasancricca/PycharmProjects/diana-demo/predictors/kb/'

selected_features = ['oldpeak','cp','thal']
numerical_feature1 = 'oldpeak'
categorical_feature1 = 'cp'
categorical_feature2 = 'thal'
numerical_features = [numerical_feature1]
categorical_features = [categorical_feature1,categorical_feature2]

dimensions = ['accuracy', 'completeness']

techniques_completeness = ['impute_standard', 'impute_mean',
       'impute_median', 'impute_random', 'impute_knn', 'impute_mice',
       'impute_linear_regression', 'impute_random_forest', 'impute_cmeans']

techniques_completeness_cat = ['impute_standard', 'impute_mode',
       'impute_random', 'impute_knn', 'impute_mice',
       'impute_logistic_regression', 'impute_random_forest']

techniques_accuracy = ['IQR', 'ISO', 'PERC', 'STD', 'ZSB', 'KNN', 'LOF']

models = ['DecisionTree','LogisticRegression','KNN','RandomForest','AdaBoost','SVC']

columns_C = ['n_tuples', 'uniqueness', 'min',
       'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr',
       'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
       'density', 'missing_perc']

columns_C_cat = ['n_tuples', 'constancy',
       'imbalance', 'uniqueness', 'unalikeability', 'entropy', 'density',
       'mean_char', 'std_char', 'skewness_char', 'kurtosis_char', 'min_char',
       'max_char', 'missing_perc']

columns_A = ['n_tuples', 'uniqueness', 'min', 'max',
       'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr', 'p_min',
       'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy', 'density',
       'percentage_outliers']

columns_R = ['n_tuples', 'n_attributes', 'p_num_var', 'p_cat_var', 'p_avg_distinct',
       'p_max_distinct', 'p_min_distinct', 'avg_density', 'max_density',
       'min_density', 'avg_entropy', 'max_entropy', 'min_entropy',
       'max_pearson', 'min_pearson', 'avg_pearson', 'duplication', 'min_min',
       'max_min', 'mean_min', 'min_max', 'max_max', 'mean_max', 'min_mean',
       'max_mean', 'mean_mean', 'min_median', 'max_median', 'mean_median',
       'min_std', 'max_std', 'mean_std', 'min_skewness', 'max_skewness',
       'mean_skewness', 'min_kurtosis', 'max_kurtosis', 'mean_kurtosis',
       'min_mad', 'max_mad', 'mean_mad', 'min_iqr', 'max_iqr', 'mean_iqr', 'perc']

columns_R_total = ['n_tuples', 'n_attributes', 'p_num_var', 'p_cat_var', 'p_avg_distinct',
       'p_max_distinct', 'p_min_distinct', 'avg_density', 'max_density',
       'min_density', 'avg_entropy', 'max_entropy', 'min_entropy',
       'max_pearson', 'min_pearson', 'avg_pearson', 'duplication', 'min_min',
       'max_min', 'mean_min', 'min_max', 'max_max', 'mean_max', 'min_mean',
       'max_mean', 'mean_mean', 'min_median', 'max_median', 'mean_median',
       'min_std', 'max_std', 'mean_std', 'min_skewness', 'max_skewness',
       'mean_skewness', 'min_kurtosis', 'max_kurtosis', 'mean_kurtosis',
       'min_mad', 'max_mad', 'mean_mad', 'min_iqr', 'max_iqr', 'mean_iqr', 'perc','impact']

columns_to_add = ['min_constancy', 'max_constancy', 'mean_constancy', 'min_imbalance',
       'max_imbalance', 'mean_imbalance', 'min_unalikeability',
       'max_unalikeability', 'mean_unalikeability', 'min_min_char',
       'max_min_char', 'mean_min_char', 'min_max_char', 'max_max_char',
       'mean_max_char', 'min_mean_char', 'max_mean_char', 'mean_mean_char',
       'min_std_char', 'max_std_char', 'mean_std_char', 'min_skewness_char',
       'max_skewness_char', 'mean_skewness_char', 'min_kurtosis_char',
       'max_kurtosis_char', 'mean_kurtosis_char']

def suggest_ranking(test, m):

    results = []
    #shap_results = []

    for dimension in dimensions:
        model = pickle.load(open(path+'models/RR_' + m + '_' + dimension, 'rb'))
        y_pred = model.predict(test)

        #import shap
        #explainer_ebm = shap.Explainer(model.predict, test)
        #shap_values_ebm = explainer_ebm(test)
        #shap_results.append(shap_values_ebm)

        results.append([dimension, y_pred])

    results = pd.DataFrame(results, columns=['dimension', 'pred'])
    results = results.sort_values(by=['pred'], ascending=True)
    results = results.reset_index(drop=True)

    dimension = results.iloc[[0]].dimension.values[0]

    return dimension

def suggest_accuracy(test):

    results = []

    for technique in techniques_accuracy:
        model = pickle.load(open(path+'models/AR_'+technique, 'rb'))
        y_pred = model.predict(test)
        results.append([technique,y_pred])

    results = pd.DataFrame(results,columns=['technique','pred'])
    results = results.sort_values(by=['pred'], ascending=False)
    results = results.reset_index(drop=True)

    return results.iloc[[0]].technique.values[0]

def suggest_completeness(test, m):

        results = []
        for technique in techniques_completeness:

            model = pickle.load(open(path+'models/CR_'+technique+'_'+m, 'rb'))
            y_pred = model.predict(test)
            results.append([technique, y_pred])

        results = pd.DataFrame(results, columns=['technique', 'pred'])
        results = results.sort_values(by=['pred'], ascending=True)
        results = results.reset_index(drop=True)

        technique = results.iloc[[0]].technique.values[0]

        return technique


def suggest_completeness_cat(test, m):
    test = np.nan_to_num(test)

    results = []
    for technique in techniques_completeness_cat:
        model = pickle.load(open(path+'models/CR_cat_' + technique + '_' + m, 'rb'))
        y_pred = model.predict(test)
        results.append([technique, y_pred])

    results = pd.DataFrame(results, columns=['technique', 'pred'])
    results = results.sort_values(by=['pred'], ascending=True)
    results = results.reset_index(drop=True)

    technique = results.iloc[[0]].technique.values[0]

    return technique

def extract_suggestion_accuracy(df, selected_features, perc):
    df = df[selected_features]

    suggestion = {}

    for s in selected_features:

        if s in list(df.select_dtypes(include=['int64','float64']).columns):
            test = np.array(numerical.get_features_num(df,s,perc))
            test = pd.DataFrame(test.reshape(-1, len(test)),columns=columns_A)

            suggestion[s] = suggest_accuracy(test)
        else:
            suggestion[s] = np.nan

    return suggestion

def extract_suggestion_completeness(df, selected_features, perc):

    suggestion = {}
    df = df[selected_features]

    for m in models:

        sugg_model = {}

        for s in selected_features:

            if s == list(df.select_dtypes(include=['int64','float64']).columns):

                test = np.array(numerical.get_features_num(df, s, perc))
                test = pd.DataFrame(test.reshape(-1, len(test)), columns=columns_C)
                sugg_model[s] = suggest_completeness(test,m)

            else:
                test = np.array(dp.cat_profile(df, s, perc))
                test = pd.DataFrame(test.reshape(-1, len(test)), columns=columns_C_cat)
                sugg_model[s] = suggest_completeness_cat(test, m)

        suggestion[m] = sugg_model

    return suggestion

def extract_suggestion_ranking(df, selected_features, perc):

    suggestion = {}
    df = df[selected_features]

    for m in models:

        test = dp.extract_profile_dataset(df, perc)
        test = test[columns_R]
        test[columns_to_add] = np.nan
        test = np.nan_to_num(test)

        suggestion[m] = suggest_ranking(test,m)

    return suggestion

if __name__ == '__main__':

    perc_nan = 0.1
    perc_out = 10
    perc = 0.8

    df = pd.read_csv("/Users/camillasancricca/PycharmProjects/diana-demo/dataset/iris_dirty.csv")

    #print(extract_suggestion_accuracy(df, selected_features, perc_out))

    #print(extract_suggestion_completeness(df, selected_features, perc_out))

    dimensions = extract_suggestion_ranking(df, df.columns, perc)
