import numpy as np
import pandas as pd
from sklearn import multiclass
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    np.random.seed(0)

    full_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")
    full_data.rename(columns=lambda x: x.replace('אבחנה_', ''), inplace=True)
    labels = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.0.csv")
    classified_labels = np.where(labels == '[]', 0, 1)
    train_X, test_X, train_y, test_y = train_test_split(full_data, labels)
    ##########
    # STEP 1:#
    ##########
    # Trying to find high correlation between features and having metastases
    correlations = []
    for feature in full_data.columns:
        print(feature, ":")
        corr = full_data[feature].corr(pd.Series(classified_labels))
        print(corr)
        correlations.append(corr)

    # Taking 5 top correlated features to handle with, and trying to fit them with ???? algorithm
    # TODO

    ##########
    # STEP 2:#
    ##########

    # Trying to find high correlation between features and having metastases
    correlations = []
    for feature in full_data.columns:
        print(feature, ":")
        corr = full_data[feature].corr(pd.Series(labels))
        print(corr)
        correlations.append(corr)

    # After predicting metastases, we want to predict which.
    # Taking 5 top correlated features to handle with, and trying to fit them with multiclassifier by sklearn
    for i in range(predictions.length):
        if predictions[i] == 1: # having metastases
            predictions[i] = multiclass()
            #TODO
            

    #TODO EXPORT RESULTS TO CSV





