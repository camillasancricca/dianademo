import pandas as pd
from openai import OpenAI
from ydata_profiling import ProfileReport
import json

OPENAI_API_KEY = "your-API-key"
GPT_CLIENT = OpenAI(api_key=OPENAI_API_KEY)

def query_llm(query):
    model_name = "gpt-3.5-turbo"
    client = GPT_CLIENT
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": query},
        ],
    )
    return response.choices[0].message.content


def get_explanation_summary(df):
    columns = str(df.columns.values)
    rows = df.head().to_string()
    description = df.describe().to_string()
    summary_prompt_not_expert = f"Introduce the dataset to a non-expert user: 1. Describing the content of this dataset. 2. Describing in detail each column. The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}.  This is the description for the numerical attributes: \n{description}. The first 5 rows and the statistics can be used to provide a more detailed description of the attributes."
    summary_prompt_expert = f"1. Summarize the content of this dataset (3 rows). 2. Provide a brief description of each attribute (1 row).  The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}. This is the description for the numerical attributes: \n{description}. The first 5 rows and the statistics are only provided to enable a better understanding of the dataset, do not mention them. Do not mention (3 rows) and (1 row) in the answer."
    string1 = "Brief explanation: "+query_llm(summary_prompt_expert)
    string2 = "Long explanation: "+query_llm(summary_prompt_not_expert)
    return string1, string2


def get_explanation_alerts(df):
    columns = str(df.columns.values)
    rows = df.head().to_string()
    description = df.describe().to_string()
    report = ProfileReport(df)
    report.to_file("report.json")
    f = open("report.json")
    report = json.load(f)
    alerts = str(report["alerts"])
    alerts_prompt_not_expert = f"Give a textual explanation for the alerts provided for this dataset to a non-expert user. The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}.  This is the description for the numerical attributes: \n{description}. The alerts: \n{alerts}."
    alerts_prompt_expert =  f"Give a short explanation (max 2 rows) for the alerts provided for this dataset. The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}.  This is the description for the numerical attributes: \n{description}. The alerts: \n{alerts}."
    string1 = "Brief explanation: "+query_llm(alerts_prompt_expert)
    string2 = "Long explanation: "+query_llm(alerts_prompt_not_expert)
    return string1, string2


def get_explanation_correlations(df):
    columns = list(df.select_dtypes(include=['int64', 'float64']).columns)
    rows = df.head().to_string()
    description = df.describe().to_string()
    corr = df[columns].corr().to_string()
    correlation_prompt_not_expert = f"Give a textual explanation for the correlations of this dataset to a non-expert user. The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}. This is the description for the numerical attributes: \n{description}. The correlation matrix: \n{corr}."
    correlation_prompt_expert = f"Give a short explanation (max 3 rows) for the correlations of this dataset. The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}. This is the description for the numerical attributes: \n{description}. The correlation matrix: \n{corr}."
    string1 = "Brief explanation: "+query_llm(correlation_prompt_expert)
    string2 = "Long explanation: "+query_llm(correlation_prompt_not_expert)
    return string1, string2


def get_explanation_missing(df):
    columns = str(df.columns.values)
    rows = df.head().to_string()
    description = df.describe().to_string()
    missing = df.isnull().sum().to_string()
    missing_prompt_not_expert = f"Give a textual explanation for the missing values distribution of this dataset. Explain to a non-expert user what are the most relevant pieces of information they should know given this missing data in the dataset. The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}. This is the description for the numerical attributes: \n{description}. The missing values distribution: \n{missing}."
    missing_prompt_expert = f"Give a short explanation (max 3 rows) for the missing values distribution of this dataset. The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}. This is the description for the numerical attributes: \n{description}. The missing values distribution: \n{missing}."
    string1 = "Brief explanation: "+query_llm(missing_prompt_expert)
    string2 = "Long explanation: "+query_llm(missing_prompt_not_expert)
    return string1, string2


def get_explanation_outliers(df):
    columns = str(df.columns.values)
    rows = df.head().to_string()
    description = df.describe().to_string()
    Q1 = df.quantile(0.25, numeric_only=True)
    Q3 = df.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1
    soglia = 1.5
    outliers_lower = df.lt(Q1 - soglia * IQR)
    outliers_upper = df.gt(Q3 + soglia * IQR)
    outliers = outliers_lower | outliers_upper
    out = outliers.sum().to_string()
    outlier_prompt_not_expert = f"Give a textual explanation for the outliers of this dataset. Explain to a non-expert user what are the most relevant pieces of information they should know given the outliers in the dataset. The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}. This is the description for the numerical attributes: \n{description}. The outliers distribution: \n{out}."
    outlier_prompt_expert = f"Give a short explanation (max 3 rows) for the outliers of this dataset. The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}. This is the description for the numerical attributes: \n{description}. The outliers distribution: \n{out}."
    string1 = "Brief explanation: "+query_llm(outlier_prompt_expert)
    string2 = "Long explanation: "+query_llm(outlier_prompt_not_expert)
    return string1, string2


def get_explanation_fd(df):
    columns = str(df.columns.values)
    rows = df.head().to_string()
    fd_prompt_not_expert = f"Explain to a non-expert user what is a functional dependency between two attributes. Give an example of functional dependency given this dataset. The columns are:\n{columns}. The first 5 rows of the dataset are: \n{rows}."
    string = "Explanation: "+query_llm(fd_prompt_not_expert)
    return string


def get_explanation_attribute(df, name):
    statistics = df[name].describe().to_string()
    distinct = str(df[name].unique())
    count = df[name].value_counts().to_string()
    attribute_prompt_not_expert = f"Describe in detail all the information related to this column to a non-expert user. The column name is: \n{name}. The statistics related to the column are: \n{statistics}. The list of distinct values is: \n{distinct}. The list of value counts is: \n{count}."
    attribute_prompt_expert = f"Describe shortly (max 3 rows) the information related to this column. The column name is: \n{name}. The statistics related to the column are: \n{statistics}. The list of distinct values is: \n{distinct}.The list of value counts is: \n{count}."
    string1 = "Brief explanation: " + query_llm(attribute_prompt_expert)
    string2 = "Long explanation: " + query_llm(attribute_prompt_not_expert)
    return string1, string2

