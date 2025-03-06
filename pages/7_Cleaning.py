from cgitb import small
import os
import webbrowser
from predictors.scripts import algorithms_class, data_profile_extraction, imputation, kb_suggestions, numerical, outlier_detection, suggestions_extraction, utils
import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stateful_button import button
import streamlit_nested_layout
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import missingno as msno
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_extras.no_default_selectbox import selectbox
from streamlit import components
import uuid
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
import math
from neo4j import GraphDatabase
from joblib import load
from sklearn import linear_model
import apps.scripts.improve_quality as improve
import apps.scripts.imputation as i
import apps.scripts.outlier_detection as od
from ydata_profiling import ProfileReport
import json


st.set_page_config(page_title="Cleaning", layout="wide", initial_sidebar_state="expanded")

if 'my_dataframe' not in st.session_state:
    st.session_state.my_dataframe = st.session_state.df
    print('Initialize dataframe...')
if 'once' not in st.session_state:
    st.session_state.once = False

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:6em;
    width:6em;
}
</style>""", unsafe_allow_html=True)

e = st.markdown("""
<style>
div[data-testid="stExpander"] div[role="button"] p {
    border: 1px solid white;
    border-radius: 5px;
    padding: 10px;
    font-size: 2rem;
    color: grey; 
    font-family: 'Verdana'
}
</style>""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-color: white;  /* Cambia il colore dello sfondo qui */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<style>h1{color: black; font-family: 'Verdana', sans-serif;}</style>", unsafe_allow_html=True)
st.markdown("<style>h3{color: black; font-family: 'Verdana', sans-serif;}</style>", unsafe_allow_html=True)


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:auto;
    width:auto;
}
</style>""", unsafe_allow_html=True)

st.markdown(
    """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
    unsafe_allow_html=True
)

def get_suggested(dimension, suggestion_accuracy, suggestion_completeness, df, columns, algorithm):

    techniques_output = []

    numerical_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
    if dimension == 'ACCURACY':
        for col in numerical_cols:
                techniques_output.append(col+' - '+suggestion_accuracy[col]+' + '+suggestion_completeness[algorithm][col])

    elif dimension == 'COMPLETENESS':
        for col in columns:
            if df[col].isnull().sum() != 0:
                techniques_output.append(col + ' - ' + suggestion_completeness[algorithm][col])

    return techniques_output

def get_techniques(dimension, columns, df):

    numerical_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)

    techniques_completeness = ['impute_standard', 'impute_mean','impute_mode',
                               'impute_median', 'impute_random', 'impute_knn', 'impute_mice',
                               'impute_linear_regression','impute_logistic_regression', 'impute_random_forest', 'impute_cmeans']

    techniques_accuracy = ['IQR', 'ISO', 'PERC', 'STD', 'ZSB', 'KNN', 'LOF']

    techniques_output = []

    if dimension == 'ACCURACY':

        for col in columns:
            if col in numerical_cols:
                for ta in techniques_accuracy:
                    for tc in techniques_completeness:
                        techniques_output.append(col+' - '+ta+' + '+tc)

    elif dimension == 'COMPLETENESS':

        for col in columns:

            if df[col].isnull().sum() != 0:

                for t in techniques_completeness:
                    techniques_output.append(col+' - '+t)

    else:

        techniques_output.append('Check and Remove Inconsistencies')

    return techniques_output

def profileAgain(df):
    if os.path.exists("newProfile.json"):
        os.remove("newProfile.json")
    profile = ProfileReport(df)
    profile.to_file("newProfile.json")
    with open("newProfile.json", 'r') as f:
        report = json.load(f)
    st.session_state['profile'] = profile
    st.session_state['report'] = report
    st.session_state['df'] = df
    newColumns = []
    for item in df.columns:
        newColumns.append(item)
    st.session_state['dfCol'] = newColumns

def accuracy_value(df):
    # Syntactic Accuracy: Number of correct values/total number of values
    correct_values_tot = 0
    tot_n_values = len(df) * len(df.columns)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for var in numeric_cols:
        # Accedi agli intervalli dalla variabile di stato della sessione
        min_val, max_val = st.session_state.intervals.get(var, (float('-inf'), float('inf')))

        # Calcola l'accuratezza per la colonna corrente
        correct_values_i = sum(1 for item in df[var] if not pd.isna(item) and min_val <= item <= max_val)
        correct_values_tot += correct_values_i

    accuracy = correct_values_tot / tot_n_values * 100
    return accuracy

def completeness_value(df):
    completeness = (df.isnull().sum().sum()) / (df.columns.size * len(df))
    completeness_percentage = 100 - completeness
    return completeness_percentage

consistency_value=100

def correlations(df):
    p_corr = 0

    num = list(df.select_dtypes(include=['int64','float64']).columns)

    corr = df[num].corr()

    if len(num) != 0:
        for c in corr.columns:
            a = (corr[c] > 0.7).sum() - 1
            if a > 0:
                p_corr += 1

        p_corr = p_corr / len(corr.columns)

        return round(p_corr,4),round(corr.replace(1,0).max().max(),4),round(corr.min().min(),4)

    else:
        return np.nan,np.nan,np.nan

def rank_kb(ranking, algorithm):

    if ranking[algorithm] == 'accuracy':
        ranking_kb = ['ACCURACY','COMPLETENESS','CONSISTENCY']
    else:
        ranking_kb = ['COMPLETENESS','ACCURACY','CONSISTENCY']

    return ranking_kb

def rank_dim(accuracy, consistency, completeness):
    ordered_values = sorted([accuracy, consistency, completeness], reverse=False)
    ranking_dim = []
    for i in range(3):
        if ordered_values[i] == accuracy:
            ranking_dim.append('ACCURACY')
        if ordered_values[i] == completeness:
            ranking_dim.append('COMPLETENESS')
        if ordered_values[i] == consistency:
            ranking_dim.append('CONSISTENCY')
    return ranking_dim

def average_ranking(ranking_kb, ranking_dim):
    # Get the unique values in both lists using set() function
    print("kb  ")
    print(ranking_kb)
    print("dim  ")
    print(ranking_dim)
    accuracy = 0
    completeness = 0
    consistency = 0
    for i in range(3):
        if ranking_kb[i] == 'ACCURACY':
            if i == 0: accuracy = accuracy + 0.5 * 60
            if i == 1: accuracy = accuracy + 0.5 * 30
            if i == 2: accuracy = accuracy + 0.5 * 10

        if ranking_kb[i] == 'COMPLETENESS':
            if i == 0: completeness = completeness + 0.5 * 60
            if i == 1: completeness = completeness + 0.5 * 30
            if i == 2: completeness = completeness + 0.5 * 10
        if ranking_kb[i] == 'CONSISTENCY':
            if i == 0: consistency = consistency + 0.5 * 60
            if i == 1: consistency = consistency + 0.5 * 30
            if i == 2: consistency = consistency + 0.5 * 10

    for i in range(3):
        if ranking_dim[i] == 'ACCURACY':
            if i == 0: accuracy = accuracy + 0.5 * 60
            if i == 1: accuracy = accuracy + 0.5 * 30
            if i == 2: accuracy = accuracy + 0.5 * 10

        if ranking_dim[i] == 'COMPLETENESS':
            if i == 0: completeness = completeness + 0.5 * 60
            if i == 1: completeness = completeness + 0.5 * 30
            if i == 2: completeness = completeness + 0.5 * 10

        if ranking_dim[i] == 'CONSISTENCY':
            if i == 0: consistency = consistency + 0.5 * 60
            if i == 1: consistency = consistency + 0.5 * 30
            if i == 2: consistency = consistency + 0.5 * 10

    sort = sorted([accuracy, consistency, completeness], reverse=True)
    ranking = []
    for i in range(3):
        if sort[i] == accuracy:
            ranking.append('ACCURACY')
            accuracy=0
        if sort[i] == completeness:
            ranking.append('COMPLETENESS')
            completeness=0
        if sort[i] == consistency:
            ranking.append('CONSISTENCY')
            consistency=0
    # Print the new list with the average order
    return ranking
    #return ['COMPLETENESS','ACCURACY','CONSISTENCY']

def save_and_apply(tech, df, outlier_range):

    x = tech.split()

    if len(x) == 5: #outlier detection and correction

        col = x[0]
        acc_tech = x[2]
        com_tech = x[4]

        indexes = od.outliers(df, acc_tech, col)
        df.loc[indexes, col] = np.nan
        df = improve.outlier_removal(df, outlier_range, col) #remove the outliers set by users
        df = i.impute(df, com_tech, col)
        return df

    elif len(x) == 3: #data imputation

        col = x[0]
        com_tech = x[2]

        df = i.impute(df, com_tech, col)
        return df

    elif tech == 'Check and Remove Inconsistencies':

        return df

# imputing missing values

selected_techniques = st.session_state.setdefault("selected_techniques", [])
if "data_preparation_pipeline" not in st.session_state:
    st.session_state["data_preparation_pipeline"] = []
if "prova" not in st.session_state:
    st.session_state["prova"]=[]

try:
    outlier_range = st.session_state.intervals
except:
    intervals = st.session_state.setdefault("intervals", {})
    for col in st.session_state.my_dataframe.columns:
        min_val = st.session_state.my_dataframe[col].min()
        max_val = st.session_state.my_dataframe[col].max()
        intervals[col] = (min_val, max_val)
num_actions = len(st.session_state["data_preparation_pipeline"])

pages = {
    "Dataset": "pagina_1",
    "Automatic Cleaning": "pagina_2",
    "Supported Cleaning": "pagina_2"
}

# Crea un menu a tendina nella barra laterale
scelta_pagina = st.sidebar.selectbox("Select a page:", list(pages.keys()))



if scelta_pagina == "Dataset":
    st.subheader("Dataset")
    st.write(st.session_state.my_dataframe)

    selected_ml_technique = st.session_state.setdefault("selected_ml_technique", "KNN")

    # Aggiungi la st.radio per la selezione della tecnica di ML
    selected_ml_technique = st.selectbox("Select a ML technique:", ['DecisionTree','LogisticRegression','KNN','RandomForest','AdaBoost','SVC'],
                                         index=['DecisionTree','LogisticRegression','KNN','RandomForest','AdaBoost','SVC'].index(selected_ml_technique))

    # Aggiorna la variabile di stato con la tecnica di ML selezionata
    st.session_state.selected_ml_technique = selected_ml_technique
    st.write(st.session_state.selected_ml_technique)
    st.write("---")

elif scelta_pagina == "Automatic Cleaning":
    df = st.session_state['df']
    dfCol = st.session_state['dfCol']
    profile = st.session_state['profile']
    report = st.session_state['report']
    st.session_state['widget'] = 500

    if 'y' not in st.session_state:
        st.session_state['y'] = 0

    st.title("Automatic")
    slate = st.empty()
    body = slate.container()

    def clean2():
        slate.empty()
        st.session_state['y'] = 2
        st.session_state['toBeProfiled'] = True
        # st.rerun()

    def clean3():
        slate.empty()
        st.session_state['y'] = 3
        st.session_state['toBeProfiled'] = True
        # st.rerun()

    NoDupKey = st.session_state['widget']

    correlations = report["correlations"]
    phik_df = pd.DataFrame(correlations["phi_k"])

    ind = 1
    correlationList = []
    for col in phik_df.columns:
        if ind < (len(phik_df.columns) - 1):
            for y in range(ind, len(phik_df.columns)):
                x = float(phik_df[col][y]) * 100
                if x > 85:
                    correlationList.append([col, str(phik_df.columns[y]), x])
            ind += 1

    correlationSum = {}
    for y in range(0, len(phik_df.columns)):
        x = 0
        z = 0
        for column in phik_df.columns:
            z = float(phik_df[column][y])
            x += z
        correlationSum.update({str(phik_df.columns[y]): x})

    colNum = len(st.session_state.my_dataframe.columns)
    threshold = round(0.4 * colNum)  # if a value has 40% of the attribute = NaN it's available for dropping
    nullList = st.session_state.my_dataframe.isnull().sum(axis=1).tolist()
    nullToDelete = []
    dfToDelete = st.session_state.my_dataframe.copy()
    rangeList = list(range(len(nullList)))
    for i in range(len(nullList)):
        if nullList[i] >= threshold:
            nullToDelete.append(i)
    if len(nullToDelete) > 0:
        notNullList = [i for i in rangeList if i not in nullToDelete]
        percentageNullRows = len(nullToDelete) / len(st.session_state.my_dataframe.index) * 100

    droppedList = []


    with body:
        if st.session_state['y'] == 0:
            st.subheader("Original dataset preview")
            st.dataframe(st.session_state.my_dataframe.head(50))
            st.markdown("---")
            st.write(
                "Click the button to perform automatically all the actions that the system finds suitable for your dataset, later you'll have the possibility to check the preview of the new dataset and to rollback action by action.")
            #st.write(st.session_state['y'])
            if st.button("Go!"):
                st.session_state['y'] = 1
                box = st.empty()
                dfAutomatic = st.session_state.my_dataframe.copy()
                st.subheader("Original dataset preview")
                st.dataframe(st.session_state.my_dataframe.head(50))
                st.markdown("---")
                if len(nullToDelete) > 0:
                    stringDropAutomaticLoad = "Dropping the " + str(len(nullToDelete)) + " rows (" + str(
                        "%0.2f" % (percentageNullRows)) + "%) that have at least " + str(
                        threshold) + " null values out of " + str(len(st.session_state.my_dataframe.columns))
                    stringDropRollback = "Check to rollback the drop of " + str(len(nullToDelete)) + " incomplete rows"
                    stringDropAutomaticConfirmed = f"Successfully dropped **{str(len(nullToDelete))}** **rows** (" + str(
                        "%0.2f" % (percentageNullRows)) + "%) that had at least " + str(
                        threshold) + " null values out of " + str(len(st.session_state.my_dataframe.columns))
                    # dfAutomatic.drop(nullToDelete, axis=0, inplace=True)
                    droppedList.append(["rows", nullToDelete])
                    if st.session_state['once'] == True:
                        with st.spinner(text=stringDropAutomaticLoad):
                            time.sleep(0.5)
                    st.success(stringDropAutomaticConfirmed)
                    with st.expander("Why I did it?"):
                        st.write(
                            "Incomplete rows are one of the principal sources of poor information. Even by applying the imputing technique within these rows would just be almost the same as incresing the dataset's size with non-real samples.")
                        if st.checkbox(stringDropRollback, value=False, key=len(nullToDelete)) == True:
                            droppedList = droppedList[: -1]
                        else:
                            dfAutomatic.drop(nullToDelete, axis=0, inplace=True)
                else:
                    st.success("All the rows are complete at least for the 60%!")
                st.markdown("---")
                for i in range(0, len(correlationList)):
                    if correlationList[i][0] in dfAutomatic.columns and correlationList[i][1] in dfAutomatic.columns:
                        if correlationSum[correlationList[i][0]] > correlationSum[correlationList[i][1]]:
                            x = 0
                            y = 1
                        else:
                            x = 1
                            y = 0
                        f"save_button_{correlationList[i][0]}_{correlationList[i][1]}"
                        strDropAutomaticCorrLoad = "Dropping column " + correlationList[i][
                            x] + " because of it's high correlation with column " + correlationList[i][y]
                        strDropAutomaticCorrConfirmed = f"Successfully dropped column **{correlationList[i][x]}** because of its high correlation with column {correlationList[i][y]}"
                        strDropCorrRollback = f"Check to rollback the drop of column **{correlationList[i][x]}**"
                        # dfAutomatic = dfAutomatic.drop(correlationList[i][x], axis=1)
                        droppedList.append(["column", correlationList[i][x]])
                        if st.session_state['once'] == True:
                            with st.spinner(text=strDropAutomaticCorrLoad):
                                time.sleep(0.5)
                        st.success(strDropAutomaticCorrConfirmed)
                        with st.expander("Why I did it?"):
                            st.write(
                                "When two columns has an high correlation between each other, this means that the 2 of them together have almost the same amount of information with respect to have only one of them. ANyway some columns can be useful, for example, to perform aggregate queries. If you think it's the case with this column you should better rollback this action and keep it!")
                            if st.checkbox(strDropCorrRollback, key=NoDupKey) == True:
                                droppedList = droppedList[: -1]
                            else:
                                dfAutomatic = dfAutomatic.drop(correlationList[i][x], axis=1)
                            NoDupKey = NoDupKey - 1
                        # st.markdown("<p id=page-bottom>You have reached the bottom of this page!!</p>", unsafe_allow_html=True)
                        st.markdown("---")
                for col in dfAutomatic.columns:
                    # k = randint(1,100)
                    if len(pd.unique(dfAutomatic[col])) == 1:
                        strDropAutomaticDistLoad = "Dropping column " + col + " because has the same value for all the rows, that is " + str(
                            dfAutomatic[col][1])
                        strDropAutomaticDistConfirmed = f"Successfully dropped column **{col}** because has the same value for all the rows, that is {dfAutomatic[col][1]}"
                        strDropDistRollback = f"Check to rollback the drop of column **{col}**"
                        droppedList.append(["column", col])
                        if st.session_state['once'] == True:
                            with st.spinner(text=strDropAutomaticDistLoad):
                                time.sleep(0.5)
                        st.success(strDropAutomaticDistConfirmed)
                        with st.expander("Why I did it?"):
                            st.write(
                                "The fact that all the rows of the dataset had the same value for this attribute, doesn't bring any additional information with respect to removing the attribute. A dumb example could be: imagine a table of people with name, surname, date of birth...Does make sense to add a column called 'IsPerson'? No, because the answer would be the same for all the rows, we already know that every row here represent a person.")
                            if st.checkbox(strDropDistRollback, key=100) == True:
                                droppedList = droppedList[: -1]
                            else:
                                dfAutomatic = dfAutomatic.drop(col, axis=1)
                        st.markdown("---")
                for col in dfAutomatic.columns:
                    nullNum = dfAutomatic[col].isna().sum()
                    distinct = dfAutomatic[col].nunique()
                    percentageNull = nullNum / len(st.session_state.my_dataframe.index) * 100
                    if percentageNull > 1:
                        if dfAutomatic[col].dtype == "object":  # automatically fill with the mode
                            x = 0
                        elif dfAutomatic[col].dtype == "float64" or dfAutomatic[
                            col].dtype == "Int64":  # automatically fill with the average
                            x = 1
                        else:
                            x = 2
                            st.error("Unrecognized col. type")
                        if x != 2:
                            strFillAutomaticLoad = "Replacing all the " + str(nullNum) + " (" + "%0.2f" % (
                                percentageNull) + "%) null values of column " + col
                            strFillAutomaticRollback = f"Check to rollback the replacement of all the null values in column **{col}**"
                            originalCol = dfAutomatic[col].copy(deep=False)
                        if x == 0:
                            try:
                                strMode = report["variables"][col]["top"]
                                dfAutomatic[col].fillna(strMode, inplace=True)
                                strFillAutomaticConfirmed = f"Successfully replaced all the {nullNum} (" + str(
                                    "%0.2f" % (
                                        percentageNull)) + f"%) null values of the column **{col}** with the mode: {strMode}"
                                explanationWhy = "Unfortunately the column had a lot of null values. In order to influence less as possible this attribute, the mode is the value less invasive in terms of filling.  In the null values you'll have the possibility also to choose other values. If you want so, remind to rollback this change in order to still have the null values in your dataset."
                            except:
                                ()
                        elif x == 1:
                            avgValue = "{:.2f}".format(report["variables"][col]["mean"])
                            dfAutomatic[col].fillna(round(round(float(avgValue))), inplace=True)
                            strFillAutomaticConfirmed = f"Successfully replaced all the {nullNum} (" + str("%0.2f" % (
                                percentageNull)) + f"%) null values of the column **{col}** with the average value: {round(float(avgValue))}"
                            explanationWhy = "Unfortunately the column had a lot of null values. In order to influence less as possible the average value of this attribute, the mean is one of the best solution for the replacement. In the null values page you'll have the possibility also to choose other values. If you want so, remind to rollback this change in order to still have the null values in your dataset."
                        if x == 0 or x == 1:
                            droppedList.append(["nullReplaced", col, dfAutomatic[col]])
                            if st.session_state['once'] == True:
                                with st.spinner(text=strFillAutomaticLoad):
                                    time.sleep(0.5)
                            st.success(strFillAutomaticConfirmed)
                            with st.expander("Why I did it?"):
                                st.write(explanationWhy)
                                k = nullNum + distinct
                                if st.checkbox(strFillAutomaticRollback, key=k) == True:
                                    droppedList = droppedList[: -1]
                                else:
                                    if x == 0:
                                        dfAutomatic[col].fillna(dfAutomatic[col].mode(), inplace=True)
                                    elif x == 1:
                                        dfAutomatic[col].fillna(avgValue, inplace=True)
                        length = round(len(dfAutomatic.index) / 10)
                        limit = round(length * 60 / 100)
                        redundancyList = []
                        for col in dfAutomatic.columns:
                            for col1 in dfAutomatic.columns:
                                if col != col1:
                                    dup = 0
                                    for i in range(length):  #reindex?
                                        if i in dfAutomatic.index:
                                            if str(dfAutomatic[col][i]) in str(
                                                    dfAutomatic[col1][i]):  # col1/arg1 ubicazione, col/arg descrizioneVia
                                                dup += 1
                                    if dup > limit:
                                        # st.write(f"The column  **{col1}** cointans the ", "%0.2f" %(percentageDup), "%" + " of the information present in the column " + f"**{col}**")
                                        redundancyList.append([col, col1])
                        intk = 200
                        flag = 0
                        for item in redundancyList:
                            flag = 1
                            strRemoveRedLoad = "Removing the redundancy of information between column " + item[
                                0] + " and " + item[
                                                   1]
                            strRemoveRedConfirmed = f"Successfully removed all the redundancy of information between **{item[0]}** and **{item[1]}**! Now the information is present only in column **{item[0]}**."
                            strRemoveRedRollback = f"Check to restore the information in column **{item[1]}**"
                            if st.session_state['once'] == True:
                                with st.spinner(text=strRemoveRedLoad):
                                    time.sleep(1)
                            st.success(strRemoveRedConfirmed)
                            with st.expander("Why I did it?"):
                                st.write(
                                    "The two columns were partially representing the same instances. So the redundant information was dropped from the most complete column. This because it's usually best practise to do not aggregate too much information within only one column.")
                                if st.checkbox(strRemoveRedRollback, key=intk) == True:
                                    droppedList = droppedList[: -1]
                                else:
                                    for i in range(len(dfAutomatic.index)):
                                        if str(dfAutomatic[item[0]][i]) in str(dfAutomatic[item[1]][i]):
                                            try:
                                                dfAutomatic[item[1]][i] = str(dfAutomatic[item[1]][i]).replace(
                                                    str(dfAutomatic[item[0]][i]), "")
                                                intk += 1
                                            except:
                                                intk += 1

                        st.info("No other actions to be perfomed")
                        st.markdown("---")
                        st.subheader("New dataset real time preview")
                        st.write(dfAutomatic)
                        st.session_state['newdf'] = dfAutomatic.copy()
                        st.warning(
                            "If you see columns with poor information you've the chance to drop them. Remind that you're also applying *permanently* all the changes above.")
                        colSave, colSaveDrop, colIdle = st.columns([1, 1, 8], gap='small')
                        with colSave:
                            button_key1 = str(uuid.uuid4())
                            #button_key2 = str(uuid.uuid4())

                            # Now you can use button_key in your st.button call
                            if st.button("Save", key=button_key1, on_click=clean2):
                                #()
                                print('Overwrite dataframe...')
                                st.session_state.my_dataframe = st.session_state['newdf']
                                st.session_state['y'] = 2
                                st.session_state['toBeProfiled'] = True
                                st.rerun()
        if st.session_state['y'] == 2:
            successMessage = st.empty()
            successString = "Please wait while the dataframe is profiled again with all the applied changes.."
            st.session_state.my_dataframe = st.session_state['newdf']
            if st.session_state['toBeProfiled'] == True:
                successMessage.success(successString)
                # st.markdown('''(#automatic)''', unsafe_allow_html=True)
                with st.spinner(" "):
                    profileAgain(st.session_state.my_dataframe)
            successMessage.success(
                "Profiling updated! You can go to 'dataset info' in order to see the report of the new dataset or comeback to the homepage.")
            st.session_state['toBeProfiled'] = False
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 10], gap='small')
            #with col1:
                #button_key_back_to_homepage2 = str(uuid.uuid4())
                # Use the unique key in your st.button call
                #if st.button("Back To Homepage", key=button_key_back_to_homepage2):
                #    switch_page("homepage")
            #with col2:
            #    if st.button("Dataset Info"):
            #        switch_page("dataset_info")
        # st.markdown("---")

if scelta_pagina == "Supported Cleaning":

    df = st.session_state.my_dataframe

    st.header("Supported Cleaning")
    col1, col2, col3 = st.columns(3)

    with st.expander("Information on Supported Data Cleaning"):
        st.write("In this section you can clean your data supported by our predictors.")
        st.write("Our predictors suggest an optimal data preparation pipeline for optimizing your future analysis outcomes.")
        st.write("They rely on a knowledge-base that contains previously analyzed datasets. They recommend the most effective pipeline for a specific analysis based on the dataset information already collected in the knowledge-base that are most similar to the input dataset.")

        st.write("1) First column: consists of a RANKING OF DATA QUALITY DIMENSIONS in order of their importance for future analysis. They should be improved in that order to reach optimal analysis results.")
        st.write("2) Second column: for each data quality dimension, our predictors suggest, the OPTIMAL DATA IMPUTATION and OUTLIER DETECTION techniques to apply for each dataset's column to improve that data dimension.")
        st.write("3) Third column: list the pipeline actions in order.")

        st.write("You can apply step by step the suggested actions and check the effect of each task on your dataset with a preview.")
        st.write("You can also change the pipeline order and the task by selecting from all the available ones.")

    with st.expander("Information on Data Preparation Actions"):
        st.write("In this section you can improve COMPLETENESS, ACCURACY and CONSITENCY data quality dimensions using the following actions:")
        st.write("COMPLETENESS - DATA IMPUTATION (substitute missing values with approximated values)")
        st.write("1) standard imputation (impute_standard) filling null values with a defined value such as ``not specified''/zero for numerical/categorical columns, a simple statistic (mean (impute_mean) and median (impute_median) for numerical, or mode (impute_mode) for categorical variables), or a random value of the column's domain (impute_random);")
        st.write("2) ML-based imputation, which estimates the null values of a target column by training an ML model on the dataset's rows with a non-null value in that target column. We employ the following algorithms: KNN (impute_knn), random forest (impute_random_forest), logistic/linear regression for numerical/categorical columns (impute_logistic/linear_regression) and fuzzy c-means (impute_cmeans);")
        st.write("3) multiple imputation by chained equations (impute_mice) that is performed as follows: (i) random imputation is applied to each missing column; (ii) the missing values are set back one feature at a time; (iii) an ML model is fitted to impute the values using the rest of data as training set; (iv) the training set is updated with the predicted column.")

        st.write("ACCURACY - OUTLIER DETECTION AND CORRECTION (identify possibile outliers/anomalies and correct them with the most suitable data imputation technique)")
        st.write("1) statistic-based methods such as:")
        st.write("1.a) excluding values in the 1st and the 99th percentiles (PERC);")
        st.write("1.b) excluding values outside [mean-(th * std); mean+(th * std)] where mean is the mean value, std the standard deviation (STD) and th a fixed threshold;")
        st.write("1.c) the modified z-score (ZSB) and d) interquartile range (IQR) techniques;")
        st.write("2) density-based methods as local outlier factor (LOF);")
        st.write("3) two ML-based techniques: KNN and isolation forest (ISO).")

        st.write("CONSITENCY - DEPENDENCIES CHECK AND REMOVAL")
        st.write("Check and Remove Incosistencies action checks for possibile violations of the rules defined in section 'Functional Dependences' and simply drop the rows that violates them.")

    #    techniques = {
    #       "Completeness": {
    #          "Imputation - Imputation using functional dependencies": "imputation - imputation usingg functional dependencies",
    #            "Imputation - Mode Imputation": "imputation - mode imputation",
    #            "Imputation - softimpute imputation" : "imputation - softimpute imputation",
    #            "Imputation - Random Imputation" : "imputation - random imputation",
    #            "Imputation - No imputation" : "imputation - no imputation ",
    #            "Imputation - Linear and Logistic Regression Imputation" : "imputation - linear and logistic regression imputation",
    #            "Imputation - Logistic Regression Imputation" : "imputation - logistic regression imputation",
    #            "Imputation - Std imputation" : "imputation - std imputation",
    #            "Imputation - Standard Value Imputation" :"imputation - standard value imputation",
    #            "Imputation - Median Imputation" : "imputation - median imputation",
    #            "Imputation - Mean Imputation" : "imputation - mean imputation",
    #          "Imputation - Mice Imputation" : "imputation - mice imputation"

    #        },
    #        "Accuracy": {
    #            "Normalize Data": "normalize_data",
    #            "Detect and Correct Outliers": "outlier_correction"
    #        },
    #        "Consistency": {
    #            "Other Techniques...": "other_techniques"
    #        }
    #    }

    with col1:

        acc = accuracy_value(st.session_state.my_dataframe)
        com = completeness_value(st.session_state.my_dataframe)
        con = consistency_value
        assessment_ranking = rank_dim(acc,con,com)
        print(assessment_ranking)

        tot_quality = 100-((100-acc)+(100-com)+(100-con))/100
        ranking, c_tech, a_tech = suggestions_extraction.suggestions(df, df.columns, tot_quality, com/100, acc/100)
        kb_ranking = rank_kb(ranking, st.session_state.selected_ml_technique)

        total_ranking = average_ranking(kb_ranking,assessment_ranking)
        print(total_ranking)
        # Create separate selectboxes for the three sections with the same choices
        tab_section_1 = st.selectbox("Select section (1):", total_ranking, index=0)
        tab_section_2 = st.selectbox("Select section (2):", total_ranking, index=1)
        tab_section_3 = st.selectbox("Select section (3):", total_ranking, index=2)

    with col2:

        #output_data = prova_db()
        actions_1 = get_techniques(total_ranking[0],df.columns, df)
        actions_2 = get_techniques(total_ranking[1],df.columns, df)
        actions_3 = get_techniques(total_ranking[2],df.columns, df)

        suggested_accuracy = get_suggested('ACCURACY',a_tech,c_tech,df, df.columns, st.session_state.selected_ml_technique)
        suggested_completeness = get_suggested('COMPLETENESS',a_tech,c_tech,df, df.columns, st.session_state.selected_ml_technique)

        # Mostra le multiselect per le azioni
        selected_actions_1 = st.multiselect(f"Suggested actions for {tab_section_1}:", actions_1, default=suggested_accuracy if total_ranking[0] == 'ACCURACY' else suggested_completeness)
        selected_actions_2 = st.multiselect(f"Suggested actions for {tab_section_2}:", actions_2, default=suggested_completeness if total_ranking[1] == 'COMPLETENESS' else suggested_accuracy)
        selected_actions_3 = st.multiselect(f"Suggested actions for {tab_section_3}:", actions_3, default=actions_3[0])

        if "selected_actions" not in st.session_state:
            st.session_state.selected_actions = []

            # Combine selected actions from all three sections
        selected_actions = selected_actions_1 + selected_actions_2 + selected_actions_3

        # Remove deselected items from the pipeline in the correct order
        removed_items = [item for item in st.session_state.selected_actions if item not in selected_actions]
        st.session_state.selected_actions = selected_actions

        # Initialize the data preparation pipeline if not already present
        if "data_preparation_pipeline" not in st.session_state:
            st.session_state.data_preparation_pipeline = []

        # Remove deselected items from the pipeline
        for item in removed_items:
            if item in st.session_state.data_preparation_pipeline:
                st.session_state.data_preparation_pipeline.remove(item)


    with col3:
        # Show the ordered pipeline as text (tag-like)
        st.write("Ordered Data Preparation Pipeline:")

        # Aggiungi le azioni selezionate alla pipeline nell'ordine corretto
        for technique in selected_actions:
            if technique in st.session_state["data_preparation_pipeline"]:
                continue  # Evita di aggiungere tecniche duplicate
            st.session_state["data_preparation_pipeline"].append(technique)

        # Visualizza la pipeline nell'ordine in cui le azioni sono state selezionate
        for index, technique_name in enumerate(st.session_state["data_preparation_pipeline"]):
            st.write(f"{index + 1}. {technique_name}")

outlier_range = st.session_state.intervals
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
num_actions = len(st.session_state["data_preparation_pipeline"])

print('Pipeline: ')
print(st.session_state["data_preparation_pipeline"])

if st.session_state.current_index < num_actions:

    current_action = st.session_state["data_preparation_pipeline"][st.session_state.current_index]
    st.write(f"Current action: {current_action}")

    # Esegui l'azione corrente sulla copia del dataframe

    my_dataframe_temp = st.session_state.my_dataframe
    my_dataframe_temp = save_and_apply(current_action, my_dataframe_temp, outlier_range)
    st.write(my_dataframe_temp)

    # Bottone per confermare l'azione corrente
    if st.button(f"Confirm Action "):
            st.session_state.my_dataframe = my_dataframe_temp
            st.session_state.current_index = min(st.session_state.current_index + 1, num_actions)

    # Bottone per fare rollback all'azione precedente
    if st.button(f"Rollback to Previous Action"):
        my_dataframe_temp = st.session_state.my_dataframe
        st.session_state.current_index = max(st.session_state.current_index - 1, 0)

    st.markdown("---")

st.write("---")

if button("Change dataset", key="changedataset"):
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df
        print('Initialize dataframe...')
    st.warning("If you don't have the modified dataset downloaded, you'll lose all the changes applied.")
    if st.button("Proceed"):
        st.session_state['x'] = 0
        switch_page("upload")

if st.button("Continue", key="continue_cleaning"):
        switch_page("Duplicate")

if st.button("Come Back", key="come_back_profiling"):
    switch_page("Functional_Dependencies")




