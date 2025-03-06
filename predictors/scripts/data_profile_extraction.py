import pandas as pd
import numpy as np
import math
import scipy
from numpy import mean
import warnings
warnings.filterwarnings("ignore")

def correlations(df, par, bool):
    p_corr = 0
    num = list(df.select_dtypes(include=["int64","float64"]).columns)
    corr = df[num].corr()
    if bool:
        if len(num) != 0:
            for c in corr.columns:
                a = (corr[c] > float(par)).sum() - 1
                if a > 0:
                    p_corr += 1
            p_corr = p_corr / len(corr.columns)
            return round(p_corr,4)
        else:
            return np.nan
    else:
        return corr

def entropy(df, column):
    prob_attr = []

    for item in df[column].unique():
        p_attr = len(df[df[column] == item])/len(df)
        prob_attr.append(p_attr)

    en_attr = 0

    if 0 in prob_attr:
        prob_attr.remove(0)

    for p in prob_attr:
        en_attr += p*np.log(p)
    en_attr = -en_attr

    return en_attr

def density(df, column):

    n_distinct = df[column].nunique()
    prob_attr = []
    den_attr = 0

    for item in df[column].unique():
        p_attr = len(df[df[column] == item])/len(df)
        prob_attr.append(p_attr)

    avg_den_attr = 1/n_distinct

    for p in prob_attr:
        den_attr += math.sqrt((p - avg_den_attr) ** 2)
        den_attr = den_attr/n_distinct

    return den_attr*100

def profile_whole_dataset(df, profile):

    numeric = len(list(df.select_dtypes(include=["int64", "float64"]).columns))
    categoric = len(list(df.select_dtypes(include=["bool", "object"]).columns))

    density_value = [density(df, column) for column in df.columns]
    entropy_value = [entropy(df, column) for column in df.columns]

    correlations_value_09 = correlations(df, 0.9, True)
    correlations_value_08 = correlations(df, 0.8, True)
    correlations_value_07 = correlations(df, 0.7, True)
    correlations_value_06 = correlations(df, 0.6, True)
    correlations_value_05 = correlations(df, 0.5, True)
    correlation = correlations(df, 0.5, False)
    if correlation.empty:
        avg_correlation = np.nan
    else:
        avg_correlation = mean(correlation.replace(1, 0))

    profile["n_tuples"] = df.shape[0]
    profile["n_attributes"] = df.shape[1]
    profile["p_num_var"] = numeric/df.shape[1]
    profile["p_cat_var"] = categoric/df.shape[1]
    profile["p_avg_distinct"] = df.nunique().mean()/df.shape[0]
    profile["p_max_distinct"] = df.nunique().max()/df.shape[0]
    profile["p_min_distinct"] = df.nunique().min()/df.shape[0]
    profile["avg_density"] = mean(density_value)
    profile["max_density"] = max(density_value)
    profile["min_density"] = min(density_value)
    profile["avg_entropy"] = mean(entropy_value)
    profile["max_entropy"] = max(entropy_value)
    profile["min_entropy"] = min(entropy_value)
    profile["p_correlated_features_0.9"] = correlations_value_09
    profile["p_correlated_features_0.8"] = correlations_value_08
    profile["p_correlated_features_0.7"] = correlations_value_07
    profile["p_correlated_features_0.6"] = correlations_value_06
    profile["p_correlated_features_0.5"] = correlations_value_05
    profile["max_pearson"] = correlation.replace(1, 0).max().max()
    profile["min_pearson"] = correlation.replace(1, 0).max().max()
    profile["avg_pearson"] = avg_correlation
    profile["duplication"] = df.duplicated().sum()/df.shape[0]

    return profile

def numeric_profile(df, profile):

    numeric = list(df.select_dtypes(include=["int64", "float64"]).columns)

    minimum = []
    maximum = []
    mean_value = []
    std = []
    median = []
    skeweness = []
    kurtosis = []
    mad = []
    iqr = []

    for column in numeric:
        values = df[column].copy().values
        minimum.append(np.nanmin(values))
        maximum.append(np.nanmax(values))
        mean_value.append(np.nanmean(values))
        std.append(np.nanstd(values))
        median.append(np.nanmedian(values))
        skeweness.append(scipy.stats.skew(values, nan_policy="omit"))
        kurtosis.append(scipy.stats.kurtosis(values, nan_policy="omit"))
        mad.append(np.nanmedian(np.abs(values - np.nanmedian(values))))
        iqr.append(np.nanquantile(values, 0.75) - np.nanquantile(values, 0.25))

    profile["min_min"] = min(minimum)
    profile["max_min"] = max(minimum)
    profile["mean_min"] = mean(minimum)

    profile["min_max"] = min(maximum)
    profile["max_max"] = max(maximum)
    profile["mean_max"] = mean(maximum)

    profile["min_mean"] = min(mean_value)
    profile["max_mean"] = max(mean_value)
    profile["mean_mean"] = mean(mean_value)

    profile["min_median"] = min(median)
    profile["max_median"] = max(median)
    profile["mean_median"] = mean(median)

    profile["min_std"] = min(std)
    profile["max_std"] = max(std)
    profile["mean_std"] = mean(std)

    profile["min_skewness"] = min(skeweness)
    profile["max_skewness"] = max(skeweness)
    profile["mean_skewness"] = mean(skeweness)

    profile["min_kurtosis"] = min(kurtosis)
    profile["max_kurtosis"] = max(kurtosis)
    profile["mean_kurtosis"] = mean(kurtosis)

    profile["min_mad"] = min(mad)
    profile["max_mad"] = max(mad)
    profile["mean_mad"] = mean(mad)

    profile["min_iqr"] = min(iqr)
    profile["max_iqr"] = max(iqr)
    profile["mean_iqr"] = mean(iqr)

    return profile

def categoric_profile(df, profile):

    categoric = list(df.select_dtypes(include=["bool", "object"]).columns)

    constancy = []
    imbalance = []
    unalikeability = []
    min_char = []
    max_char = []
    std_char = []
    mean_char = []
    skewness_char = []
    kurtosis_char = []

    for column in categoric:

        values = df[column].copy().values
        values = np.array([str(value) for value in values])

        unique_values, counts = np.unique(values[values != "nan"], return_counts=True)
        valid_value_count = np.sum(counts)
        max_appearances = np.max(counts)
        min_appearances = np.min(counts)
        char_per_value = np.array([len(value) if value != "nan" else np.nan for value in values])

        constancy.append(max_appearances / valid_value_count)
        imbalance.append(max_appearances / min_appearances)

        unalikeability_sum = 0
        for i in range(len(counts)):
            for j in range(len(counts)):
                if i != j:
                    unalikeability_sum += counts[i] * counts[j]
        unalikeability.append(unalikeability_sum / (valid_value_count ** 2 - valid_value_count))

        mean_char.append(np.nanmean(char_per_value))
        std_char.append(np.nanstd(char_per_value))
        min_char.append(np.nanmin(char_per_value))
        max_char.append(np.nanmax(char_per_value))
        if min_char == max_char:  # there is a problem when the number of characters per value is always the same
            skewness_char.append(np.nan)
            kurtosis_char.append(np.nan)
        else:
            skewness = scipy.stats.skew(char_per_value, nan_policy="omit")
            skewness_char.append(1 / (1 + np.exp(-skewness)))
            kurtosis = scipy.stats.kurtosis(char_per_value, nan_policy="omit")
            kurtosis_char.append(1 / (1 + math.exp(-kurtosis)))

    profile["min_constancy"] = min(constancy)
    profile["max_constancy"] = max(constancy)
    profile["mean_constancy"] = mean(constancy)

    profile["min_imbalance"] = min(imbalance)
    profile["max_imbalance"] = max(imbalance)
    profile["mean_imbalance"] = mean(imbalance)

    profile["min_unalikeability"] = min(unalikeability)
    profile["max_unalikeability"] = max(unalikeability)
    profile["mean_unalikeability"] = mean(unalikeability)

    profile["min_min_char"] = min(min_char)
    profile["max_min_char"] = max(min_char)
    profile["mean_min_char"] = mean(min_char)

    profile["min_max_char"] = min(max_char)
    profile["max_max_char"] = max(max_char)
    profile["mean_max_char"] = mean(max_char)

    profile["min_mean_char"] = min(mean_char)
    profile["max_mean_char"] = max(mean_char)
    profile["mean_mean_char"] = mean(mean_char)

    profile["min_std_char"] = min(std_char)
    profile["max_std_char"] = max(std_char)
    profile["mean_std_char"] = mean(std_char)

    profile["min_skewness_char"] = min(skewness_char)
    profile["max_skewness_char"] = max(skewness_char)
    profile["mean_skewness_char"] = mean(skewness_char)

    profile["min_kurtosis_char"] = min(kurtosis_char)
    profile["max_kurtosis_char"] = max(kurtosis_char)
    profile["mean_kurtosis_char"] = mean(kurtosis_char)

    return profile


def cat_profile(df, column, perc):

        values = df[column].copy().values
        values = np.array([str(value) for value in values])

        rows = values.shape[0]
        uniqueness = round(len(df[column].unique()) / len(df[column].notnull()), 4)

        unique_values, counts = np.unique(values[values != "nan"], return_counts=True)
        valid_value_count = np.sum(counts)
        max_appearances = np.max(counts)
        min_appearances = np.min(counts)
        char_per_value = np.array([len(value) if value != "nan" else np.nan for value in values])

        constancy = max_appearances / valid_value_count
        imbalance = max_appearances / min_appearances

        unalikeability_sum = 0
        for i in range(len(counts)):
            for j in range(len(counts)):
                if i != j:
                    unalikeability_sum += counts[i] * counts[j]
        unalikeability = unalikeability_sum / (valid_value_count ** 2 - valid_value_count)

        mean_char = np.nanmean(char_per_value)
        std_char = np.nanstd(char_per_value)
        min_char = np.nanmin(char_per_value)
        max_char = np.nanmax(char_per_value)
        if min_char == max_char:  # there is a problem when the number of characters per value is always the same
            skewness_char = np.nan
            kurtosis = np.nan
        else:
            skewness = scipy.stats.skew(char_per_value, nan_policy="omit")
            skewness_char = 1 / (1 + np.exp(-skewness))
            kurtosis = scipy.stats.kurtosis(char_per_value, nan_policy="omit")
            kurtosis = 1 / (1 + math.exp(-kurtosis))

        entr = round(entropy(df, column), 4)
        dens = round(density(df, column), 4)

        return float(rows), float(constancy), float(imbalance), float(uniqueness), float(unalikeability), float(entr), float(dens),\
               float(mean_char), float(std_char), float(skewness_char), float(kurtosis), float(min_char), float(max_char), perc

def profile(df, dataset):

    profile = {}
    profile["name"] = dataset

    profile = profile_whole_dataset(df, profile)

    if list(df.select_dtypes(include=["int64", "float64"])):
        profile = numeric_profile(df, profile)

    if list(df.select_dtypes(include=["bool", "object"])):
        profile = categoric_profile(df, profile)

    return profile

def extract_profile_dataset(df, perc):
    profile = {}
    profile["name"] = 'weather'

    profile = profile_whole_dataset(df, profile)

    if list(df.select_dtypes(include=["int64", "float64"])):
        profile = numeric_profile(df, profile)

    if list(df.select_dtypes(include=["bool", "object"])):
        profile = categoric_profile(df, profile)

    profile["perc"] = perc

    return pd.DataFrame(profile, index=[0])

if __name__ == "__main__":

        df = pd.read_csv("../dataset/weather.csv")
        #selected_features = ['Temperature', 'Precipitation', 'AtmosphericPressure']

        #df = df[selected_features]
        #extract_profile_dataset(df,0.1)

        print(cat_profile(df, 'CloudCover', 0.1))


