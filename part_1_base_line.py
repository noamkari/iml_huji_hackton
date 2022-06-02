import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score

# pio.renderers.default = "browser"
from prediction import preprocess

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
    # train_X, test_X = processed_data[:int(processed_data.shape[0]*0.75)],processed_data[int(processed_data.shape[0]*0.75):]
    # train_y, test_y = labels[:int(labels.size*0.75)], labels[int(labels.size*0.75):]
    stump_tree = DecisionTreeClassifier(max_depth=1)
    max_depth_tree = DecisionTreeClassifier(max_depth=200)
    stump_tree.fit(train_X,train_y)
    max_depth_tree.fit(train_X,train_y)
    st_score = stump_tree.score(test_X,test_y)
    mdt_score = max_depth_tree.score(test_X,test_y)

    stump_pred = stump_tree.predict(test_X)
    max_depth_tree_pred = max_depth_tree.predict(test_X)

    # print('Stump score: ', st_score," Max tree score: ", mdt_score)
    # print('Stump Accuracy', accuracy_score(stump_pred,test_y,normalize=False), 'Tree Accuracy', accuracy_score(mdt_score,train_y,normalize=False))


    print('Test set size = ', len(test_y))

    multi_target_stump = MultiOutputClassifier(stump_tree, n_jobs=2)
    multi_target_stump.fit(train_X, train_y)
    stump_multi_score = multi_target_stump.score(test_X, test_y)
    print("Stump multi score ", stump_multi_score, "Accuracy ", accuracy_score(multi_target_stump.predict(test_X),test_y,normalize=False))

    multi_target_tree = MultiOutputClassifier(max_depth_tree, n_jobs=2)
    multi_target_tree.fit(train_X, train_y)
    tree_multi_score = multi_target_tree.score(test_X, test_y)
    print("Tree multi score ", tree_multi_score,  "Accuracy ", accuracy_score(multi_target_tree.predict(test_X),test_y,normalize=False))

    forest = RandomForestClassifier(random_state=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
    multi_target_forest.fit(train_X, train_y)
    forest_multi_score = multi_target_forest.score(test_X,test_y)
    print("Forst score ", forest_multi_score,  "Accuracy ", accuracy_score(multi_target_forest.predict(test_X),test_y,normalize=False   ))

