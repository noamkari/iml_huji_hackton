import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_validate


if __name__ == '__main__':
    np.random.seed(0)

    full_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")
    full_data.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)
    labels = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.0.csv")
    classified_labels = np.where(labels == '[]', 0, 1)
    processed_data = preprocess(full_data).drop(['Side', 'User Name','Stage','Surgery date1','Surgery date2',
                                                 'Surgery date3','Surgery name1','Surgery name2','Surgery name3',
                                                 'Surgery sum','surgery before or after-Activity date',
                                                 'surgery before or after-Actual activity',
                                                 'id-hushed_internalpatientid'],axis=1)
    train_X, test_X, train_y, test_y = train_test_split(processed_data,labels)
    scores = []
    for i in range(1, 500):
        ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),n_estimators=i)
        ada.fit(train_X,train_y)
        # pred = ada.predict(test_X)
        score = ada.score(test_X,test_y)
        scores.append(score)



