from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier


import seaborn as sn
import matplotlib.pyplot as plt

def     load_data(filename: str):
    """
    Load  dataset
    Parameters
    ----------
    filename: str
        Path to  dataset
    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    features = full_data[["h_booking_id",
                          "hotel_id",
                          "accommadation_type_name",
                          "hotel_star_rating",
                          "customer_nationality"]]
    labels = full_data["cancellation_datetime"]

    return features, labels


def evaluate_and_export(estimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.
    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.
    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction
    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses
    filename:
        path to store file at
    """
    pd.DataFrame(estimator.predict(X),
                 columns=["predicted_values"]).to_csv(filename, index=False)


def feature_evaluation(X: pd.DataFrame, y: pd.Series):
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    y_axis = []

    y_np = y.to_numpy()
    y_np = y_np.ravel()

    for i, feature in enumerate(X.columns):
        y_axis.append(
            np.cov(X[feature], y_np)[0, 1] / (
                    np.std(X[feature]) * np.std(y_np)))

    for i, feature in enumerate(X):
        create_scatter_for_feature(X[feature], y_np, round(y_axis[i], 3),
                                   feature)


def create_scatter_for_feature(X: pd.DataFrame, y: np.array, title,
                               feature_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers"))
    fig.update_layout(title="Pirson: " + str(title),
                      xaxis_title=feature_name,
                      yaxis_title="y")

    # fig.show()


def is_in_str(s: str, words: set):
    if type(s) != str:
        return False
    for w in words:
        if s.find(w) != -1:
            return True
    return False


from datetime import date
from datetime import datetime


def proces_time(x):
    today = datetime.strptime("6/2/2022", '%d/%m/%Y')
    print(type(x))
    x = datetime.strptime(x[:10], '%d/%m/%Y')
    print(type(x))
    print(type((x - today).days))


def how_much(x, d: dict):
    if x in d:
        d[x] += 1
    else:
        d[x] = 0


def preprocess(df: pd.DataFrame):
    # cur_date

    today = datetime.strptime("6/2/2022", '%d/%m/%Y')
    df["Diagnosis date"] = df["Diagnosis date"].apply(
        lambda x: (datetime.strptime(x[:10], '%d/%m/%Y') - today).days)

    # make categorical from Hospital and Form Name
    X = pd.get_dummies(df, columns=[" Hospital", " Form Name"])

    # Her2 preprocessing
    set_pos = {"po", "PO", "Po", "2", "3", "+", "חיובי", 'בינוני', "Inter",
               "Indeter", "indeter", "inter"}
    set_neg = {"ne", "Ne", "NE", "eg", "no", "0", "1", "-", "שלילי"}
    print(X.columns)
    X["Her2"] = X["Her2"].astype(str)
    X["Her2"] = X["Her2"].apply(lambda x: 1 if is_in_str(x, set_pos) else x)
    X["Her2"] = X["Her2"].apply(lambda x: 0 if is_in_str(x, set_neg) else x)
    X["Her2"] = X["Her2"].apply(lambda x: 0 if type(x) == str else x)

    # more simple but same i think todo chek with elad
    a = df["Her2"].apply(lambda x: 1 if is_in_str(x, set_pos) else 0)
    val = (set(a == X["Her2"]))

    # Age  preprocessing FIXME buggy, chek what need to do (remove line, get mean)
    # X = X[0 < X["Age"] < 120]

    # Basic stage preprocessing
    X["Basic stage"] = X["Basic stage"].replace(
        {'Null': 0, 'c - Clinical': 1, 'p - Pathological': 2,
         'r - Reccurent': 3})

    # margin type
    margin_neg = {'נקיים', 'ללא'}
    margin_pos = {'נגועים'}
    X["Margin Type"] = X["Margin Type"].apply(
        lambda x: 1 if is_in_str(x, margin_pos) else 0)

    return X


if __name__ == '__main__':
    np.random.seed(0)

    # Load data and preprocess
    full_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")
    # for f in full_data.columns:
    #     f.replace('-הנחבא','')
    full_data.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)
    corrMatrix = full_data.corr(method = 'pearson')
    print(corrMatrix)
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    # full_data['Basic stage'].corr(full_data[])
    X, y = load_data("test.csv")
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    data_path, y_location_of_distal, y_tumor_path = sys.argv[1:]

    original_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")

    # remove heb prefix
    original_data.rename(columns=lambda x: x.replace('אבחנה-', ''),
                         inplace=True)

    y_tumor = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.1.csv")
    y_tumor.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)

    print({f: original_data[f].unique().size for f in original_data.columns})
    print(set(original_data["Histological diagnosis"]))

    d = {}
    original_data["Histological diagnosis"].apply(lambda x: how_much(x, d))
    print(d)

    X = preprocess(original_data)
    feature_evaluation(X[["Age", "Her2", "Basic stage"]], y_tumor)
    print("this is me")
