import pandas as pd
import numpy as np
import numpy.random
import numpy
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def KNN(df, element):

    X = df[element].values
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto')
    X = np.nan_to_num(X)
    nbrs.fit(X.reshape(-1, 1))
    # distances and indexes of k-neighbors from model outputs
    distances, indexes = nbrs.kneighbors(X.reshape(-1, 1))
    # plot mean of k-distances of each observation
    # plt.plot(distances.mean(axis =1))
    outlier_index = np.where(distances.mean(axis=1) > 0.0001)
    # filter outlier values
    outlier_values = df[element].iloc[outlier_index]

    list_index = []
    list_outliers = []

    for x in outlier_values.index:
        list_index.append(x)
    #for x in outlier_values:
        #list_outliers.append(x)

    if len(list_index) >= df[element].notnull().sum():
        list_index.pop()

    return list_index


def ZSB(df, element):
    # Robust Zscore as a function of median and median
    # mean absolute deviation (MAD) defined as
    # z-score = |x – median(x)| / mad(x)
    data = df[element].values
    median = df[element].median()
    median_absolute_deviation = data - median
    median_absolute_deviation = abs(median_absolute_deviation)
    median_absolute_deviation = np.nanmedian(median_absolute_deviation)
    modified_z_scores = (data - median) / (median_absolute_deviation * 1.4826)
    #outliers = data[np.abs(modified_z_scores) > threshold]
    index = np.where(np.abs(modified_z_scores) > 2)[0].tolist()
    return index


def STD(df, element):
    data = pd.DataFrame(df[element])
    mean = data.mean().values[0]
    std = data.std().values[0]
    V1 = mean + 3 * std
    V2 = mean - 3 * std
    outliers = []
    outliers_ind = []
    for index, row in data.iterrows():
        if (row[element] > V1) | (row[element] < V2):
            #outliers.append(row[element])
            outliers_ind.append(index)

    return outliers_ind


def PERC(df, element):
    data = pd.DataFrame(df[element])
    V1 = data.quantile(.99).values[0]
    V2 = data.quantile(.1).values[0]
    outliers = []
    outliers_ind = []
    for index, row in data.iterrows():
        if (row[element] > V1) | (row[element] < V2):
            #outliers.append(row[element])
            outliers_ind.append(index)

    return outliers_ind


def ISO(data, element):
    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0)
    X = data[element]
    X = np.nan_to_num(X)
    X = X.reshape(-1, 1)
    model.fit(X)
    prova = pd.DataFrame()
    prova['scores'] = model.decision_function(X)
    prova['anomaly'] = model.predict(X)
    anomaly = prova.loc[prova['anomaly'] == -1]
    outliers_ind = list(anomaly.index)

    return outliers_ind

def IQR(df, element):
    data = pd.DataFrame(df[element])

    Q1 = data.quantile(.25).values[0]
    Q3 = data.quantile(.75).values[0]
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    outliers = []
    outliers_ind = []
    for index, row in data.iterrows():
        if (row[element] > upper_range) | (row[element] < lower_range):
            #outliers.append(row[element])
            outliers_ind.append(index)

    return outliers_ind


def LOF(df, element):

    X = pd.DataFrame(df[element])
    X = np.nan_to_num(X)
    # requires no missing value
    # select top 10 outliers

    # fit the model for outlier detection (default)
    clf = LocalOutlierFactor(n_neighbors=20, contamination='auto')

    clf.fit_predict(X)

    LOF_scores = clf.negative_outlier_factor_
    # Outliers tend to have a negative score far from -1

    outliers_index = np.array(np.where(LOF_scores < -1.1)[0]).tolist()
    return outliers_index

def outliers(df, method, col):

    try:
        if method == 'KNN':
            return KNN(df, col)
        if method == 'ZSB':
            return ZSB(df, col)
        if method == 'STD':
            return STD(df, col)
        if method == 'PERC':
            return PERC(df, col)
        if method == 'ISO':
            return ISO(df, col)
        if method == 'IQR':
            return IQR(df, col)
        if method == 'LOF':
            return LOF(df, col)
    except:
        return IQR(df, col)

if __name__ == '__main__':
    df = pd.read_csv("../dataset/weather.csv")
    name_class = 'WeatherType'
    selected_features = ['Temperature', 'Precipitation', 'AtmosphericPressure', name_class]
    df = df[selected_features]

    #dirty = dirty_data.injection(df, name_class, 0.1, 10, 1)

    #techniques_accuracy = ['IQR', 'ISO', 'PERC', 'STD', 'ZSB', 'KNN', 'LOF']
    #for t in techniques_accuracy:
        #print(outliers(dirty.copy(), t, 'Temperature'))
