import numpy as np


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
    b = np.zeros((4, 2))
    print(np.apply_along_axis(lambda x: x + x[0], 1, a))

    b[0][0] = 1
    b[1][1] = 2
    print(np.sum(b))