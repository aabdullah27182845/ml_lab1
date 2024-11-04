import _pickle as cp
import numpy as np
import matplotlib.pyplot as mp

RED_WINE = 'data/winequality-red.pickle'
WHITE_WINE = 'data/winequality-white.pickle'

X, y = cp.load(open(WHITE_WINE, 'rb'))

N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train

X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]

def bar_chart():
    y_int = y.astype(int)

    # Initialize a dictionary to store the counts of values from 3 to 9
    counts = {value: np.sum(y_int == value) for value in range(3, 10)}

    keys = list(counts.keys())
    values = list(counts.values())

    mp.bar(keys, values)
    mp.xlabel('Values')
    mp.ylabel('Counts')
    mp.title('Counts of Values 3 to 9')
    mp.show()


def mean_squared_error():
    mean = np.mean(y)
    squared_mean = np.mean(y**2)
    return mean, squared_mean - mean**2

def lin_model_least_squares():
    column_means = np.mean(X, axis=0)
    column_stds = np.std(X, axis=0)

    X_standardized = (X - column_means) / column_stds

    w_part_1 = np.linalg.inv(X_standardized.transpose() @ X_standardized)
    # not what they asked: just reporting the column means and column stds is enough
    # this may be useful tho
    w = w_part_1 @ X_standardized.transpose() @ y
    print(w)

def main():
    
    #TASK 1:
    # bar_chart()

    #TASK 2:
    mean, variance = mean_squared_error()
    print(variance)

    #TASK 3:
    lin_model_least_squares()

if __name__ == "__main__":
    main()
