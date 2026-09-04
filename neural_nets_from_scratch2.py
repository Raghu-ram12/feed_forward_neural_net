import numpy as np

class Loss:

    def mse(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

    def mse_prime(self, predicted, actual):

        return predicted - actual

    def binary_cross_entropy(self, predicted, actual, eps=1e-9):
        p = np.clip(predicted, eps, 1 - eps)
        return -np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p))

    def bce_prime(self, predicted, actual, eps=1e-9):
        p = np.clip(predicted, eps, 1 - eps)

        return (p - actual) / (p * (1 - p))


class Activation:

    def tanh(self, z):
        # np.tanh is numerically stable; avoid computing exp(z)/exp(-z) manually
        return np.tanh(z)

    def tanh_prime(self, z):
        return 1 - self.tanh(z) ** 2

    def relu(self, z):
        return np.maximum(0, z)

    def relu_prime(self, z):
        return np.where(z >= 0, 1, 0)

    def leaky_relu(self, z):
        return np.maximum(self.leaky_alpha * z, z)

    def leaky_relu_prime(self, z):
        return np.where(z >= 0, 1, self.leaky_alpha)

    def sigmoid(self, z):
        # stable sigmoid: avoids overflow in np.exp for large negative/positive z
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z)),
        )

    def sigmoid_prime(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)


class DenseLayer(Activation):

    def __init__(self, input_dims, output_dims):

        self.input_dims = input_dims
        self.output_dims = output_dims

        self.weights = None
        self.bias = None

        self.activation_func = None
        self.optimizer = None
        self.leaky_alpha = 0.01  

        self.z = None
        self.prev_activation = None
        self.A = None

        self.dw = None
        self.db = None

    def initialize_weights(self, activation_func):

        self.activation_func = activation_func

        initialization_factor = 0.1

        if self.activation_func in ["relu", "leaky-relu"]:
            initialization_factor = np.sqrt(2 / self.input_dims)

        elif self.activation_func in ["sigmoid", "tanh"]:
            initialization_factor = np.sqrt(1 / self.input_dims)

        self.weights = np.random.randn(self.input_dims, self.output_dims) * initialization_factor

        self.bias = np.zeros((1, self.output_dims))

    def initialize_optimizer(self, optimizer="grad", learning_rate=0.01, beta=0.9, gamma=0.99):
        """
        Initializes the optimizer state.

        optimizer     : "grad", "grad-momentum", "adagrad", or "rmsprop"
        learning_rate : step size used by every optimizer variant
        beta          : momentum factor for gradient descent with momentum
        gamma         : decay rate for rmsprop
        """

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.delta = 1e-7  # small constant to avoid division by zero

        if self.optimizer == "grad":
            pass

        elif self.optimizer == "grad-momentum":
            self.dw_v = np.zeros_like(self.weights)
            self.db_v = np.zeros_like(self.bias)
            self.beta = beta

        elif self.optimizer == "adagrad":
            self.prev_weight_sum_sq = np.zeros_like(self.weights)
            self.prev_bias_sum_sq = np.zeros_like(self.bias)

        elif self.optimizer == "rmsprop":
            self.sw = np.zeros_like(self.weights)
            self.sb = np.zeros_like(self.bias)
            self.gamma = gamma

    def _get_activated_output(self):

        if self.activation_func == "relu":
            self.A = self.relu(self.z)

        elif self.activation_func == "leaky-relu":
            self.A = self.leaky_relu(self.z)

        elif self.activation_func == "tanh":
            self.A = self.tanh(self.z)

        elif self.activation_func == "sigmoid":
            self.A = self.sigmoid(self.z)

        return self.A

    def forward_pass(self, prev_activation):

        self.prev_activation = prev_activation

        self.z = np.dot(prev_activation, self.weights) + self.bias

        self.A = self._get_activated_output()

        return self.A

    def _get_derivative(self):

        if self.activation_func == "relu":
            return self.relu_prime(self.z)

        elif self.activation_func == "leaky-relu":
            return self.leaky_relu_prime(self.z)

        elif self.activation_func == "tanh":
            return self.tanh_prime(self.z)

        elif self.activation_func == "sigmoid":
            return self.sigmoid_prime(self.z)

    def backward_pass(self, prev_delta):

        delta = prev_delta * self._get_derivative()

        m = self.prev_activation.shape[0]

        self.dw = np.dot(self.prev_activation.T, delta) / m
        self.db = np.sum(delta, axis=0, keepdims=True) / m

        return np.dot(delta, self.weights.T)

    def optimize_weights(self):

        if self.optimizer == "grad":
            self.weights = self.weights - self.learning_rate * self.dw
            self.bias = self.bias - self.learning_rate * self.db

        elif self.optimizer == "grad-momentum":
            self.dw_v = self.beta * self.dw_v + (1 - self.beta) * self.dw
            self.db_v = self.beta * self.db_v + (1 - self.beta) * self.db

            self.weights -= self.learning_rate * self.dw_v
            self.bias -= self.learning_rate * self.db_v

        elif self.optimizer == "adagrad":
            self.prev_weight_sum_sq += np.square(self.dw)
            self.prev_bias_sum_sq += np.square(self.db)

            scaled_lr_w = self.learning_rate / np.sqrt(self.delta + self.prev_weight_sum_sq)
            scaled_lr_b = self.learning_rate / np.sqrt(self.delta + self.prev_bias_sum_sq)

            self.weights = self.weights - scaled_lr_w * self.dw
            self.bias = self.bias - scaled_lr_b * self.db

        elif self.optimizer == "rmsprop":
            self.sw = self.gamma * self.sw + (1 - self.gamma) * np.square(self.dw)
            self.sb = self.gamma * self.sb + (1 - self.gamma) * np.square(self.db)

            scaled_lr_w = self.learning_rate / np.sqrt(self.delta + self.sw)
            scaled_lr_b = self.learning_rate / np.sqrt(self.delta + self.sb)

            self.weights = self.weights - scaled_lr_w * self.dw
            self.bias = self.bias - scaled_lr_b * self.db


class Network:

    def __init__(self, optimizer="grad", cost="mse"):

        self.optimizer = optimizer

        self.layers = []

        self.output = None

        self.L = Loss()

        self.cost = cost

    def add_layer(self, input_dims, output_dims, activation="relu", optimizer=None, alpha=0.01, beta=0.9, gamma=0.99):

        optimizer = optimizer or self.optimizer

        new_layer = DenseLayer(input_dims, output_dims)
        new_layer.initialize_weights(activation)
        new_layer.initialize_optimizer(optimizer, alpha, beta, gamma)
        self.layers.append(new_layer)

    def forward(self, x):

        prev_activation = x

        for layer in self.layers:

            layer.forward_pass(prev_activation)

            prev_activation = layer.A

        self.output = prev_activation

        return self.output

    def get_loss_gradient(self, predicted, actual):

      
        if self.cost == "binary-cross-entropy" and self.layers[-1].activation_func == "tanh":

            raise Warning("binary-cross-entropy needs a value between (0,1)")

        if self.cost == "mse":

            return self.L.mse_prime(predicted, actual)

        elif self.cost == "binary-cross-entropy":

            return self.L.bce_prime(predicted, actual)

    def backward(self, y):

        prev_delta = self.get_loss_gradient(self.output, y)

        for layer in self.layers[::-1]:

            prev_delta = layer.backward_pass(prev_delta)

            layer.optimize_weights()

    def _compute_loss(self, predicted, actual):

        if self.cost == "mse":
            return self.L.mse(predicted, actual)

        elif self.cost == "binary-cross-entropy":
            return self.L.binary_cross_entropy(predicted, actual)

    def train(self, X, y, epochs=100, batch_size=None, shuffle=True, verbose=True):
        """
        batch_size = None or len(X)  -> batch gradient descent
        batch_size = 1               -> stochastic gradient descent
        batch_size = 32 (etc.)       -> mini-batch gradient descent
        """
        n_samples = X.shape[0]

        if batch_size is None:
            batch_size = n_samples  # full-batch

        for epoch in range(epochs):

            if shuffle:
                perm = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[perm], y[perm]
            else:
                X_shuffled, y_shuffled = X, y

            epoch_loss = 0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                self.forward(X_batch)
                batch_loss = self._compute_loss(self.output, y_batch)
                self.backward(y_batch)

                epoch_loss += batch_loss
                n_batches += 1

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss / n_batches:.6f}")

    def predict(self, x_test):
       
        self.forward(x_test)

        return self.output 
    