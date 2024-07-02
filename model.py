import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt

"""
        A few types of images the model tends to do poorly on include:
            *Cat body in an unusual position
            *Cat appears against a background of a similar color
            *Unusual cat color and species
            *Camera Angle
            *Brightness of the picture
            *Scale variation (cat is very large or small in image) 
"""

## Forward


def sigmoid(z):
    z = np.clip(z, -500, 500)
    s = 1 / (1 + np.exp(-z))
    cache = z
    return s, cache

def relu(z):

    A = np.maximum(0, z)
    cache = z

    return A, cache

def initialize_parameters_deep(dims):
    np.random.seed(1)

    parameters = {}
    L = len(dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(dims[l], dims[l - 1]) / np.sqrt(dims[l-1])
        parameters['b' + str(l)] = np.zeros((dims[l], 1))
    return parameters

def linear_forward(A, W, b):

    Z = W.dot(A) + b
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    else:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):

    m = Y.shape[1]
    epsilon = 1e-8
    cost = -1 / m * np.sum(Y * np.log(AL + epsilon) + (1 - Y) * np.log(1 - AL + epsilon))
    #cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db



## BACKWARD



def sigmoid_backward(dA, cache):

    z = cache
    s = 1 / (1 + np.exp(-z))
    dZ = dA * s * (1 - s)

    return dZ


def relu_backward(dA, cache):

    z = cache
    dZ = np.array(dA, copy=True)
    dZ[z <= 0] = 0

    return dZ

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    else:
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):

    grads = {}

    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    epsilon = 1e-8
    dAL = - (np.divide(Y, AL + epsilon) - np.divide(1 - Y, 1 - AL + epsilon))
    #dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")

    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters



## Preparing


def load_data(name_train, name_test):

    train_dataset = h5py.File('datasets/{}'.format(name_train), "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/{}'.format(name_test), "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_y = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y = test_y_orig.reshape((1, test_y_orig.shape[0]))

    return train_x_orig, train_y, test_x_orig, test_y, classes

def preprocess_data(train_x_orig, test_x_orig):

    train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255.
    test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255.

    return train_x, test_x



## Model


def L_layer_model(X, Y, dims, learning_rate = 0.007, iterations = 3000, print_cost=False):

    costs = []
    parameters = initialize_parameters_deep(dims)

    for i in range(0, iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0 or i == iterations - 1:
            print("Iteration {} : Costs {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == iterations:
            costs.append(cost)

    return parameters, costs


def predict(X, Y, parameters, image):

    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = L_model_forward(X, parameters)


    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    accuracy = np.sum((p == Y) / m) * 100

    if image:
        answer = int(np.squeeze(p))
        if answer == 1:
            print("\nThe neural network says : It's a cat !")
        else:
            print("\nThe neural network says : It's not a cat !")

    else :
        print("Accuracy - {:.2f}%\n".format(accuracy))

    return p

def predict_image(path, label_y, parameters, num_px):

    image = np.array(Image.open(path))
    image_resized = np.array(Image.fromarray(image).resize((num_px, num_px))).reshape((num_px * num_px * 3, 1))
    plt.imshow(image)
    plt.show()
    predict(image_resized / 255., label_y, parameters, True)


if __name__ == '__main__':

    print('-' * 80 + "\n\n")
    name_train = input("Enter the train dataset name with the file extension (.h5) : \n")
    name_test = input("\nEnter the test dataset name with the file extension (.h5) : \n")
    print("\n\n" + '-' * 80)

    print('-' * 80 + "\n\n" + "Loading data..." + "\n\n" + '-' * 80)
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data(str(name_train), str(name_test))
    num_px = train_x_orig.shape[1]

    train_x, test_x = preprocess_data(train_x_orig, test_x_orig)
    # Edit dims for more layers/neurons
    dims = [12288, 256, 64, 16, 1]
    while(True):
        print('-' * 80 + "\n\n")
        print_cost = input("Do you want to see the costs ? Enter Yes or No\n").strip().lower() == 'yes'
        learning_rate = float(input("\nEnter the learning rate\n"))
        iterations = int(input("\nEnter the number of the interations\n"))
        if not print_cost:
            print("\nTraining the model...\n")
        else:
            print("\n")
        parameters, costs = L_layer_model(train_x, train_y, dims, learning_rate, iterations, print_cost)
        print("\n\n" + '-' * 80)

        print('-' * 80 + "\n\n")
        print("Train Dataset")
        predictions_train = predict(train_x, train_y, parameters, False)
        print("Test Dataset")
        predictions_test = predict(test_x, test_y, parameters, False)
        repeat_params = input("Do you want to change the parameters? Enter Yes or No\n").strip().lower() == 'yes'
        print('-' * 80)
        if not repeat_params:
            break

    while(True):
        print('-' * 80 + "\n\n")
        question = input("Want to see what the model predicts for your image ? Enter Yes or No\n").strip().lower() == 'yes'
        if not question:
            print("\n\n" + '-' * 80)
            break
        image = input("\nEnter the image name with the file extension : \n")
        label_y = [int(input("\nEnter 1 for true (cat) or 0 for false (not cat) as label : \n"))]
        predict_image("images/" + str(image), label_y, parameters, num_px)
        print("\n\n" + '-' * 80)