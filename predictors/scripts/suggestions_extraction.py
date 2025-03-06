import pandas as pd
from predictors.scripts import kb_suggestions

def suggestions(df, selected_features, perc_q, perc_nan, perc_out):

    ranking = kb_suggestions.extract_suggestion_ranking(df, selected_features, perc_q)
    c_tech = kb_suggestions.extract_suggestion_completeness(df, selected_features, perc_nan)
    a_tech = kb_suggestions.extract_suggestion_accuracy(df, selected_features, perc_out)

    return ranking, c_tech, a_tech

if __name__ == '__main__':

    df = pd.read_csv("../../dataset/iris_dirty.csv")

    selected_features = df.columns

    ranking, c_tech, a_tech = suggestions(df, selected_features, 0.8, 0.9, 10)

    print(ranking)
    print(c_tech)
    print(a_tech)
