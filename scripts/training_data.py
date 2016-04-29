
import numpy as np

def get_xor_data_unsupervised():
    X_train = np.array([
                        [[1], [0], [1]],
                        [[1], [1], [0]],
                        [[0], [1], [1]],
                        [[0], [0], [0]]
                    ])
    X_val = np.array([
                        [[0], [1], [1]],
                        [[1], [1], [0]]
                    ])

    return {'X_train':X_train, 'X_val':X_val}

def get_xor_data_supervised():
    X_train = np.array([
                        [[1], [0]],
                        [[1], [1]],
                        [[0], [1]],
                        [[0], [0]]
                    ])
    y_train = np.array([[1], [0], [1], [0]])
    X_val = np.array([
                        [[0], [1]],
                        [[1], [1]]
                    ])
    y_val = np.array([[1], [0]])

    return {'X_train':X_train, 'y_train':y_train, 'X_val':X_val, 'y_val':y_val}

def get_alternating_data():
    X_train = np.array([[[0], [1], [0], [1]], [[1], [0], [1], [0]]])
    y_train = np.array([[0], [1]])

    X_val = np.array([[[0], [1], [0]], [[1], [0], [1]]])
    y_val = np.array([[1], [0]])
    return {'X_train':X_train, 'y_train':y_train, 'X_val':X_val, 'y_val':y_val}


