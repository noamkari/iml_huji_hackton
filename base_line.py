import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import plotly.graph_objects as go

# pio.renderers.default = "browser"
from prediction import preprocess

if __name__ == '__main__':
    np.random.seed(0)

    full_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")
    full_data.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)
    labels = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.0.csv")
    classified_labels = np.where(labels == '[]', 0, 1)
    train_X, test_X, train_y, test_y = train_test_split(preprocess(full_data).drop('User Name',axis=1), classified_labels)
    stump_tree = DecisionTreeClassifier(max_depth=1)
    max_depth_tree = DecisionTreeClassifier(max_depth=200)
    scores_stump = cross_validate(stump_tree, train_X, np.array(train_y))
    scores_max_tree = cross_validate(max_depth_tree, train_X, np.array(train_y))
    fig = go.Figure([go.Scatter(x=train_X, y=scores_stump, mode="markers+lines", name="stump errors",
                                 marker=dict(color="red", opacity=.7)),
                      go.Scatter(x=train_X, y=scores_max_tree, mode="markers+lines", name="max depth tree errors",
                                 marker=dict(color="blue", opacity=.7))
                      ])
    fig.update_layout(
        title_text=rf"$\text{{Stump vs very deep Tree cross validation results}}$",
        xaxis={"title": r""},
        yaxis={"title": r"error", })
    fig.show()
