import numpy as np
from enum import Enum


# Activation functions and their gradients -----------------------------------------------------------------------------
def sigmoid(x):
    """
    Sigmoid activation function
    :param x: input value
    :return: output value
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Hyperbolic tangent activation function
    :param x: input value
    :return: output value
    """
    return 2 * sigmoid(2 * x) - 1


def relu(x):
    """
    Rectified linear unit
    :param x: input value
    :return: output value
    """
    return np.maximum(0.0, x)


def leaky_relu(x, a=0.01):
    """
    Parameterized leaky rectified linear unit
    :param x: input value
    :param a: negative value scaling factor
    :return: output value
    """
    return np.piecewise(x, condlist=[x < 0, x >= 0], funclist=[lambda v: a * v, lambda v: v])


def exponential_relu(x, a=0.1):
    """
    Exponential rectified linear unit
    :param x: input value
    :param a: negative value scaling factor
    :return: output value
    """

    def negative_case(x2):
        return a * (np.exp(x2) - 1)

    return np.piecewise(x, condlist=[x < 0, x >= 0], funclist=[negative_case, lambda v: v])


def softplus(x, beta=1.0, thr=20):
    """
    Softplus activation function
    :param x: input value
    :param beta: hyperparameter
    :param thr: threshold above which the activation function becomes linear
    :return: output value
    """

    def below_threshold_case(x):
        return 1 / beta * np.log(1 + np.exp(beta * x))

    return np.piecewise(x, condlist=[x > thr, x <= thr], funclist=[lambda v: v, below_threshold_case])


def swish(x):
    """
    Swish activation function

    source:
        Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for activation functions. arXiv preprint arXiv:1710.05941.

    :param x: input value
    :return: output value
    """
    return x * sigmoid(x)


def mish(x, beta=1.0, thr=20):
    """
    Mish activation function

    source:
        Misra, D. (2019). Mish: A self regularized non-monotonic neural activation function. arXiv preprint arXiv:1908.08681, 4(2), 10-48550.

    :param x: input value
    :param beta: hyperparameter for softplus
    :param thr: threshold for softplus
    :return: output value
    """
    return x * tanh(softplus(x, beta, thr))


def sigmoid_gradient(x):
    """
    Gradient of the sigmoid activation function
    :param x: input value
    :return: output value
    """
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_gradient(x):
    """
    Gradient of the hyperbolic tangent activation function
    :param x: input value
    :return: output value
    """
    return 1 - np.square(tanh(x))


def relu_gradient(x, zero_derivative=0):
    """
    Gradient of the relu activation function
    :param x: input values
    :param zero_derivative: the derivative of relu at x=0 is undefined, thus this value will be used in that case
    :return: gradient of relu at x
    """
    return np.piecewise(x, condlist=[x < 0, x > 0, x == 0], funclist=[0, 1, zero_derivative])


def leaky_relu_gradient(x, a=0.01, zero_derivative=0):
    """
    Gradient of the leaky rectified linear unit
    :param x: input value
    :param a: negative scaling factor
    :param zero_derivative: value returned for x == 0, which is undefined
    :return:
    """
    return np.piecewise(x, condlist=[x < 0, x > 0, x == 0], funclist=[a, 1, zero_derivative])


def exponential_relu_gradient(x, a=0.1):
    """
    Gradient of the exponential rectified linear unit
    :param x: input value
    :param a: negative scaling factor
    :return: output value
    """

    def negative_case(x2):
        return a * (np.exp(x2) - 1) + a

    return np.piecewise(x, condlist=[x <= 0, x > 0], funclist=[negative_case, 1])


def swish_gradient(x):
    """
    Gradient of the swich activation function
    :param x: input value
    :return: output value
    """
    return swish(x) * sigmoid(x) * (1 - swish(x))


def softplus_gradient(x):
    """
    Gradient of the softplus activation function
    :param x: input value
    :return: output value
    """
    return sigmoid(x)


def mish_gradient(x):
    """
    Gradient of the mish activation function
    :param x: input value
    :return: output value
    """

    def sech(x2):
        return 1 / np.cosh(x2)

    return sech(softplus(x)) * (x * sigmoid(x)) + mish(x) / x


# Loss functions and their gradients -----------------------------------------------------------------------------------
def mse(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss function
    :param y_true: true value
    :param y_pred: predicted value
    :return: mean squared error
    """
    return np.mean(np.square(y_pred - y_true))


def mse_gradient(y_true, y_pred):
    """
    Gradient of the mean squared error loss function
    :param y_true: true value
    :param y_pred: predicted value
    :return: gradient of the mean squared error
    """
    return 2 * np.mean(y_pred - y_true)


# convenience functions for data batching ------------------------------------------------------------------------------
def minibatch(X, Y, batch_size):
    num_samples = len(X)
    # create an array of indices and shuffle them
    indices = np.arange(num_samples)
    # np.random.shuffle(indices)
    # loop through the number of batches we have
    for i in range(0, num_samples - batch_size + 1, batch_size):
        batch_indices = indices[i:i + batch_size]
        # yield the X and Y batches
        yield X[batch_indices], Y[batch_indices]


# classes for neural networks and layers -------------------------------------------------------------------------------
from enum import Enum


class Initializer(Enum):
    UNIFORM = 0
    HE = 1
    GLODOT = 2


class Layer:
    def __init__(self, num_neurons, activation_function, activation_function_gradient, is_input_layer=False,
                 is_output_layer=False):
        self.is_input_layer = is_input_layer
        self.is_output_layer = is_output_layer
        if is_input_layer and is_output_layer:
            raise ValueError("Layer cannot be both an input and an output layer at the same time.")
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.activation_function_gradient = activation_function_gradient

        # this will hold the last activation output of the layer, which we will need for backpropagation
        self.last_activation = None

    def activate(self, x):
        """
        Propagate given data forward through the activation function
        """
        if not self.is_input_layer:
            y = self.activation_function(x)
        else:
            y = x
        self.last_activation = y
        return y

    def gradient(self, x):
        """
        Calculates the gradient of the activation function at the given value
        """
        return self.activation_function_gradient(x)

    def last_activation_gradient(self):
        """
        Calculate the gradient of the activation function with respect to the net input to each neuron, which is equal
        to the gradient evaluated at the last activation
        :return:
        """
        return self.gradient(self.last_activation)


class NeuralNetwork:
    def __init__(self, num_inputs):
        self.weights = []
        self.biases = []
        self.weight_gradients = []
        self.bias_gradients = []
        self.layers = [Layer(num_inputs, None, None, is_input_layer=True), ]

    def add_layer(self,
                  num_neurons,
                  activation_function,
                  activation_function_gradient,
                  weight_init_method=Initializer.UNIFORM,
                  bias_init_method=Initializer.UNIFORM,
                  uniform_weight_min=-0.5,
                  uniform_weight_max=0.5,
                  uniform_bias_min=-0.1,
                  uniform_bias_max=0.1,
                  is_output_layer=False):
        # shape of weight matrix between last layer and new layer
        weights_shape = (self.layers[-1].num_neurons, num_neurons)
        # shape of bias vector for this layer
        bias_shape = num_neurons
        # initialize weights based on requested method
        if weight_init_method == Initializer.UNIFORM:
            new_weights = np.random.uniform(low=uniform_weight_min, high=uniform_weight_max, size=weights_shape)
        else:
            raise ValueError("The provided weight initialization method is not yet implemented.")
        # initialize biases based on requested method
        if bias_init_method == Initializer.UNIFORM:
            new_biases = np.random.uniform(low=uniform_bias_min, high=uniform_bias_max, size=bias_shape)
        else:
            raise ValueError("The provided weight initialization method is not yet implemented.")
        # create new layer
        new_layer = Layer(num_neurons, activation_function, activation_function_gradient,
                          is_output_layer=is_output_layer)
        # store the new neuron layer and weight matrix
        self.layers.append(new_layer)
        self.weights.append(new_weights)
        self.biases.append(new_biases)
        # store some zero-filled arrays which will contain the weight and bias gradients we will use for backpropagation
        self.weight_gradients.append(np.zeros_like(new_weights))
        self.bias_gradients.append(np.zeros_like(new_biases))

    def forward(self, inputs):
        """
        Run data forward through the network
        :param inputs: input data vector
        :return: output vector
        """
        value = inputs
        for index, layer in enumerate(self.layers):
            if layer.is_input_layer:
                value = layer.activate(value)
            else:
                # dot product between previous layer activations and weight matrix between previous and current layer plus the bias vector of current layer
                net_input = self.weights[index - 1].T @ value.T + self.biases[index - 1].T
                value = layer.activate(net_input).T
                # print("output", value.shape, value)
                # print("Stop")
        return value

    def predict(self, inputs):
        """
        Performs a prediction on the given input data. Just an alias for the forward function
        :param inputs: input data vector
        :return: output vector
        """
        return self.forward(inputs)

    def train(self,
              X_train,
              Y_train,
              X_validation,
              Y_validation,
              loss_function,
              loss_function_gradient,
              learning_rate=0.001,
              epochs=500,
              batch_size=8,
              momentum=0.0):
        training_loss = []
        validation_loss = []
        for i in range(epochs):
            # this will store historical data on weight changes, which we need for momentum
            weight_delta_history = [np.zeros_like(wg) for wg in self.weight_gradients]
            for x_batch, y_batch in minibatch(X_train, Y_train, batch_size):
                # Run the network through the batch and calculate the average gradient
                for x_sample, y_sample in zip(x_batch, y_batch):
                    # get prediction for this sample
                    y_predicted = self.forward(x_sample)

                    # Now we will calculate the gradient of the loss with respect to each weight and bias in the network
                    # using backpropagation
                    # last_deltas will contain the backpropagated deltas that were last calculated
                    last_deltas = None
                    for index, layer in reversed(list(enumerate(self.layers))):
                        if layer.is_input_layer:
                            break
                        # partial derivative of loss with respect to neuron activation in this layer
                        # outer layers and inner layers need to be handled differently
                        dL_dA = loss_function_gradient(y_sample, y_predicted) if layer.is_output_layer else last_deltas

                        # partial derivative of the activation function with respect to the net input (T)
                        dA_dT = layer.last_activation_gradient()

                        # partial derivative of the net input with respect to weights, which is just the inputs from the
                        # layer before this one
                        dT_dW = self.layers[index - 1].last_activation

                        # partial derivative of the loss with respect to the weights
                        dL_dW = np.atleast_2d(dT_dW).T @ np.atleast_2d((dL_dA * dA_dT))

                        # partial derivative of the net input with respect to bias, which is just 1
                        dT_dB = np.ones_like(self.biases[index - 1])

                        # partial derivative of the net input with respect to the activation of neurons from the layer
                        # before this one, which is just the weight matrix.
                        dT_dAp = self.weights[index - 1]

                        # partial derivative of the loss with respect to the activation of neurons from the layer
                        # before this one, which is the deltas that we will carry over
                        last_deltas = dT_dAp @ (dL_dA * dA_dT)

                        # partial derivative of loss with respect to the biases
                        dL_dB = dL_dA * dA_dT * dT_dB

                        # now accumulate the weight and bias gradients for the batch
                        self.weight_gradients[index - 1] += np.atleast_2d(dL_dW)
                        self.bias_gradients[index - 1] += dL_dB

                # calculate average gradients over the batch
                for gw, gb in zip(self.weight_gradients, self.bias_gradients):
                    gw /= batch_size
                    gb /= batch_size

                # update the weights and biases
                for i in range(len(self.weights)):
                    # calculate the amount by which we will change the weights (weight delta)
                    weight_delta = learning_rate * self.weight_gradients[i] + momentum * weight_delta_history[i]

                    # apply weight change
                    self.weights[i] -= weight_delta

                    # store this weight delta so that we can use it for the momentum calculation next epoch
                    weight_delta_history[i] = weight_delta
                for i in range(len(self.biases)):
                    self.biases[i] -= learning_rate * self.bias_gradients[i]

                # now clear the gradients for the next batch
                self.clear_gradients()

            # Calculate the loss for the training and validation sets now that we have run over every batch
            epoch_training_loss = 0.0
            epoch_validation_loss = 0.0
            for x_sample, y_sample in zip(X_train, Y_train):
                # get prediction for this sample
                y_predicted = self.forward(x_sample)
                # calculate loss for this prediction
                sample_loss = loss_function(y_sample, y_predicted)
                epoch_training_loss += sample_loss
            for x_sample, y_sample in zip(X_validation, Y_validation):
                # get prediction for this sample
                y_predicted = self.forward(x_sample)
                # calculate loss for this prediction
                sample_loss = loss_function(y_sample, y_predicted)
                epoch_validation_loss += sample_loss
            # calculate average loss over both the training and validation sets
            epoch_training_loss /= len(X_train)
            epoch_validation_loss /= len(X_validation)

            # store the epoch training and validation loss
            training_loss.append(epoch_training_loss)
            validation_loss.append(epoch_validation_loss)

        return training_loss, validation_loss

    @property
    def num_layers(self):
        """
        Convenience function to get the number of neuron layers in the network
        :return: number of layers in network
        """
        return len(self.layers)

    @property
    def num_neurons(self):
        """
        Convenience function to get the total number of neurons in the network (including input and output neurons
        :return:
        """
        total = 0
        for l in self.layers:
            total += l.num_neurons
        return total

    def clear_gradients(self):
        """
        Clears the weight and bias gradient arrays
        :return: None
        """
        for a in self.weight_gradients:
            a.fill(0.0)
        for a in self.bias_gradients:
            a.fill(0.0)
