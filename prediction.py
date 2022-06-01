from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier


def load_data(filename: str):
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


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = "."):
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
    x_axis = X.columns

    # fixme not good
    y_np = y.to_numpy()
    y_np = y_np.ravel()

    for i, feature in enumerate(X.columns):
        y_axis.append(
            np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y)))

    for i, feature in enumerate(X):
        create_scatter_for_feature(X[feature], y_np, round(y_axis[i], 3),
                                   feature,
                                   output_path)


def create_scatter_for_feature(X_feature: pd.DataFrame, prices, title,
                               feature_name: str, output_path: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_feature, y=prices, mode="markers"))
    fig.update_layout(title="Pirson: " + str(title),
                      xaxis_title=feature_name,
                      yaxis_title="price")

    fig.show()
    # fig.write_image(output_path + feature_name+".png")


def preprocess(df):
    df = df.rename(
        columns={" Hospital": "Hospital", " Form Name": "Form Name"})

    X = pd.concat(
        [pd.get_dummies(df["Hospital"], columns="Hospital"),
         pd.get_dummies(df["Form Name"], columns="Form Name"),
         df], axis=1)
    print(X)

    return X


if __name__ == '__main__':
    np.random.seed(0)

    # Load data and preprocess
    df = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")
    df.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)

    y_tumor = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.1.csv")
    y_tumor.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)

    for feature in df.columns:
        print(feature)
        print(df[feature].unique().size)

    print(set(df["Histological diagnosis"]))

    X = preprocess(df)

    # feature_evaluation(X, y_tumor, "")
    print("this is me")
