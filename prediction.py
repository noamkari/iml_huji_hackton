from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def find_in_str(s, words):
    if type(s) != str:
        return False
    for w in words:
        if s.find(w) != -1:
            return True
    return False

def load_data(filename: str):
    """
    Load  datasetvi
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
    full_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")
    full_data.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)


    #Her2 preprocessing
    set_pos = {"po", "PO", "Po", "2", "3","+","חיובי",'בינוני', "Inter","Indeter","indeter", "inter"}
    set_neg = {"ne","Ne","NE", "eg","no","0","1","-","שלילי"}
    full_data["Her2"] = full_data["Her2"].astype(str)
    full_data["Her2"] = full_data["Her2"].apply(lambda x: 1 if find_in_str(x, set_pos) else x)
    full_data["Her2"] = full_data["Her2"].apply(lambda x: 0 if find_in_str(x, set_neg) else x)
    full_data["Her2"] = full_data["Her2"].apply(lambda x: 0 if type(x)==str else x)
    #Age  preprocessing
    full_data = full_data[0 < full_data["age"] < 120]
    #Basic stage  preprocessing
    full_data["Basic stage"] = full_data["Basic stage"].replace({ 'Null':0, 'c - Clinical':1,'p - Pathological':2, 'r - Reccurent':3})




    for feature in full_data.columns:
        print(feature)
    for feature in full_data.columns:
        print(feature)
        print(full_data[feature].unique())

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


if __name__ == '__main__':
    np.random.seed(0)
    full_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv", )
    train_labels_0 = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.0.csv")
    train_labels_1 = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.1.csv")

    # Load data and preprocess
    X, y = load_data("./Mission 2 - Breast Cancer/train.feats.csv")
    train_X, test_X, train_y, test_y = train_test_split(X, y)

    # Fit model over data
    estimator = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=100,
        random_state=0)
    estimator.fit(train_X, train_y)

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "predictions.csv")

    print("this is me")
