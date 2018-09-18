import numpy as np

a = np.array([
    [
        [0, 4, 1],
        [0, 0, 0],
        [2, 3, 4],
        [1, 2, 3]
    ],
    [[0, 4, 1], [0, 0, 0], [9, 9, 9], [1, 2, 3]]
])
a = (np.where(a == (np.array([9,9,9])).all()))
print(a)
