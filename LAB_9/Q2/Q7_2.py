import matplotlib.pyplot as plt
import numpy as np

def augment_vector(X):
    bias = np.ones((len(X), 1))
    return np.hstack((X, bias))

def normalise_train_set(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    train_set = augment_vector(X)
    labels_ = np.unique(Y)
    idx = Y == labels_[1]  # Select one class as negative class
    train_set[idx] = -train_set[idx]  # Make the x coordinate of selected class as negative
    dim = train_set.shape[1]  # Return Dimensionality of feature space // train set
    return X, Y, train_set, dim

def relaxation_algo_with_margin(X, Y, learning_rate, margin):
    # Obtained the values as required (Augmented and negated)
    X, Y, train_set, dim = normalise_train_set(X, Y)
    weights = [0, 0, 1]
    k = -1
    i = 0
    count = 0
    while i != len(train_set):
        k = (k + 1) % len(train_set)
        if np.dot(train_set[k], weights) <= margin:
            i = 0
            temp1 = (margin - np.dot(train_set[k], weights))
            temp2 = np.dot(train_set[k], train_set[k])
            temp3 = float((float(temp1) / float(temp2)) * 2)
            temp = np.dot(temp3, train_set[k])
            weights = weights + (learning_rate * temp)
        else:
            i += 1
            count += 1

    plot_boundary(weights, train_set, X, Y)
    return weights


def plot_boundary(weights, train_set, X, Y):
    # plot data-points
    x_points = X[:, 0]
    y_points = X[:, 1]
    length = len(x_points)
    length = int(length)
    x_points_1 = x_points[0:int(length / 2)]
    x_points_2 = x_points[int(length / 2):length]
    y_points_1 = y_points[0:int(length / 2)]
    y_points_2 = y_points[int(length / 2):length]

    plt.plot(x_points_1, y_points_1, 'ro');
    plt.axis([0, 10, 0, 10])
    plt.plot(x_points_2, y_points_2, 'bo');

    a, b, c = weights
    xchord_1 = 0
    xchord_2 = -(float(c)) / (float(a))
    ychord_2 = 0
    ychord_1 = -(float(c)) / (float(b))
    plt.title("Linear classifier using relaxation criteria")
    plt.plot([xchord_1, xchord_2], [ychord_1, ychord_2], 'black')
    plt.show()


def predict(test_set, weights):
    test_set = augment_vector(test_set)
    pred_list = []
    for i in range(len(test_set)):
        if np.dot(test_set[i], weights) < 0:
            pred_list.append(2)
        elif np.dot(test_set[i], weights) > 0:
            pred_list.append(1)
    return pred_list


def compute_accuracy(pred_labels_, Y_test):
    count = 0
    length = len(pred_labels_)
    for k in range(len(pred_labels_)):
        if pred_labels_[k] == Y_test[k]:
            count = count + 1
    accuracy = (float(count) / float(length)) * 100
    return round(accuracy,3)

def main():
    print("\nLinear classifier using relaxation criteria \n")
    X = [(1, 6), (7, 2), (8, 9), (9, 9), (4, 8), (8, 5), (2, 1), (3, 3), (2, 4), (7, 1), (1, 3), (5, 2)]
    Y = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    testset = [(0, 1), (8, 1), (2, 6), (2, 4.5), (6, 1.5), (4, 3)]
    test = [(6, 4), (6, 6), (9, 4), (0, 0), (0, -2), (1, 1)]
    Y_test = [1, 1, 1, 2, 2, 2]
    Xnew = [(1, 6), (7, 6), (8, 9), (9, 9), (4, 8), (8, 5), (2, 1), (3, 3), (2, 4), (7, 1), (1, 3), (5, 2)]
    X_no_sep = [(2, 1), (7, 2), (2, 4), (9, 9), (4, 8), (5, 2), (1, 6), (3, 3), (8, 9), (7, 1), (1, 3), (8, 5)]

    learning_rate = 1
    margin = 1.0
    weights = relaxation_algo_with_margin(X, Y, learning_rate, margin)
    pred_labels = predict(test, weights)
    print("Predicted Classes: ")
    print(pred_labels)
    accuracy = compute_accuracy(pred_labels, Y_test)
    print('Accuracy for the test data set is  {}%'.format(accuracy))


main()

