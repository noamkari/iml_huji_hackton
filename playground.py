import numpy as np
import pandas as pd


def foo(n):
    lst = list()
    for i in range(n):
        lst.append(i)
        lst = lst[::-1]

    return lst


def tra(vec):
    sp = {i: val for i, val in enumerate(vec) if val}
    return sp


if __name__ == '__main__':
    a = np.array([[1, 10, 100],
                  [2, 20, 200],
                  [3, 30, 300]])
    df = pd.DataFrame({'a': ["g1", "g2", "g3"], 'b': [0, 0, -1]})
    d = pd.get_dummies(df, columns=['a'])
    b = np.zeros((4, 2))
    s = [1, 10, 100]
    b[0][0] = 1
    b[1][1] = 2
    print(np.sum(b))
