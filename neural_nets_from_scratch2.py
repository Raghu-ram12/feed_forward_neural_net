import numpy as np
from pathlib import Path

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

        else:
            self.A = self.z

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

        else:
            return np.ones_like(self.z)

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



def get_output_dims(H, W, kh, kw, stride, pad):
    H_out = (H + 2 * pad - kh) // stride + 1
    W_out = (W + 2 * pad - kw) // stride + 1
    return H_out, W_out


def im2col(x, kh, kw, stride=1, pad=0):
    """
    x : (N, C, H, W)
    
    returns cols of shape (N * H_out * W_out, C * kh * kw)
    """
    N, C, H, W = x.shape
    H_out, W_out = get_output_dims(H, W, kh, kw, stride, pad)

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    cols = np.zeros((N, C, kh, kw, H_out, W_out))

    for y in range(kh):
        y_max = y + stride * H_out
        for xk in range(kw):
            x_max = xk + stride * W_out
            cols[:, :, y, xk, :, :] = x_padded[:, :, y:y_max:stride, xk:x_max:stride]

    # -> (N, H_out, W_out, C, kh, kw) -> (N*H_out*W_out, C*kh*kw)
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
    return cols, H_out, W_out


def col2im(cols, x_shape, kh, kw, stride=1, pad=0):
    """
    Inverse of im2col: scatters column-gradients back onto an image-shaped
    array, accumulating overlapping contributions.
    """
    N, C, H, W = x_shape
    H_out, W_out = get_output_dims(H, W, kh, kw, stride, pad)

    cols_reshaped = cols.reshape(N, H_out, W_out, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)

    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    x_padded = np.zeros((N, C, H_padded, W_padded))

    for y in range(kh):
        y_max = y + stride * H_out
        for xk in range(kw):
            x_max = xk + stride * W_out
            x_padded[:, :, y:y_max:stride, xk:x_max:stride] += cols_reshaped[:, :, y, xk, :, :]

    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]


class Conv2DLayer(DenseLayer):
    """
    A 2D convolution layer inheriting from DenseLayer.
    Reuses activation logic, optimizer initialization, weight updates, and base layer interface.

    input_shape : (C, H, W)  channels, height, width of the incoming feature map
    """

    def __init__(self, input_shape, num_filters, kernel_size, stride=1, padding=0):

        self.C, self.H, self.W = input_shape
        self.num_filters = num_filters
        self.kh, self.kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding

        self.H_out, self.W_out = get_output_dims(self.H, self.W, self.kh, self.kw, stride, padding)
        self.output_shape = (num_filters, self.H_out, self.W_out)

        fan_in = self.C * self.kh * self.kw
        fan_out = num_filters * self.H_out * self.W_out
        super().__init__(input_dims=fan_in, output_dims=fan_out)

        self.x_shape = None
        self.x_cols = None

    def initialize_weights(self, activation_func):
        self.activation_func = activation_func

        fan_in = self.input_dims
        if self.activation_func in ["relu", "leaky-relu"]:
            factor = np.sqrt(2 / fan_in)
        elif self.activation_func in ["sigmoid", "tanh"]:
            factor = np.sqrt(1 / fan_in)
        else:
            factor = 0.1

        self.weights = np.random.randn(self.num_filters, self.C, self.kh, self.kw) * factor
        self.bias = np.zeros((self.num_filters, 1))

    def forward_pass(self, x):
        """
        x : (N, C, H, W)
        """
        N = x.shape[0]
        self.prev_activation = x
        self.x_shape = x.shape

        x_cols, H_out, W_out = im2col(x, self.kh, self.kw, self.stride, self.padding)

        self.x_cols = x_cols  # (N*H_out*W_out, C*kh*kw)  -- cached for backward_pass

        w_col = self.weights.reshape(self.num_filters, -1).T  # (C*kh*kw, num_filters)

        out = x_cols @ w_col + self.bias.T  # (N*H_out*W_out, num_filters)

        out = out.reshape(N, H_out, W_out, self.num_filters).transpose(0, 3, 1, 2)

        self.z = out
        self.A = self._get_activated_output()

        return self.A

    def backward_pass(self, prev_delta):
        """
        prev_delta : gradient w.r.t. this layer's output, shape (N, num_filters, H_out, W_out)
        returns    : gradient w.r.t. this layer's input, shape (N, C, H, W)
        """
        N = self.x_shape[0]

        delta = prev_delta * self._get_derivative()
        
        delta_reshaped = delta.transpose(0, 2, 3, 1).reshape(-1, self.num_filters)  # (N*H_out*W_out, F)

        self.dw = (delta_reshaped.T @ self.x_cols).reshape(self.weights.shape) / N
        self.db = np.sum(delta_reshaped, axis=0, keepdims=True).T / N

        w_col = self.weights.reshape(self.num_filters, -1)  # (F, C*kh*kw)

        dx_cols = delta_reshaped @ w_col

        dx = col2im(dx_cols, self.x_shape, self.kh, self.kw, self.stride, self.padding)
        return dx


class MaxPool2DLayer(DenseLayer):
    """
    Max Pooling layer inheriting from DenseLayer.
    No learnable parameters; overrides initialization, optimization, forward, and backward passes.
    """

    def __init__(self, input_shape, pool_size=2, stride=None):
        self.C, self.H, self.W = input_shape
        self.pool_h, self.pool_w = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = stride or self.pool_h

        self.H_out = (self.H - self.pool_h) // self.stride + 1
        self.W_out = (self.W - self.pool_w) // self.stride + 1
        self.output_shape = (self.C, self.H_out, self.W_out)

        super().__init__(input_dims=self.C * self.H * self.W, output_dims=self.C * self.H_out * self.W_out)

        self.x_shape = None
        self.arg_max = None

    def initialize_weights(self, activation_func=None):
        self.activation_func = activation_func
        self.weights = None
        self.bias = None

    def initialize_optimizer(self, *args, **kwargs):
        pass  # no parameters to optimize

    def forward_pass(self, x):
        N, C, H, W = x.shape
        self.prev_activation = x
        self.x_shape = x.shape

        out = np.zeros((N, C, self.H_out, self.W_out))
        self.arg_max = np.zeros_like(out, dtype=np.int64)

        for i in range(self.H_out):
            for j in range(self.W_out):
                h0, h1 = i * self.stride, i * self.stride + self.pool_h
                w0, w1 = j * self.stride, j * self.stride + self.pool_w

                window = x[:, :, h0:h1, w0:w1].reshape(N, C, -1)
                out[:, :, i, j] = np.max(window, axis=2)
                self.arg_max[:, :, i, j] = np.argmax(window, axis=2)

        self.z = out
        self.A = out
        return self.A

    def backward_pass(self, prev_delta):
        N, C, H, W = self.x_shape
        dx = np.zeros(self.x_shape)

        n_idx, c_idx = np.indices((N, C))

        for i in range(self.H_out):
            for j in range(self.W_out):
                h0 = i * self.stride
                w0 = j * self.stride

                flat_idx = self.arg_max[:, :, i, j]
                rows, cols_ = np.unravel_index(flat_idx, (self.pool_h, self.pool_w))

                dx[n_idx, c_idx, h0 + rows, w0 + cols_] += prev_delta[:, :, i, j]

        return dx

    def optimize_weights(self):
        pass  # no parameters to update


class FlattenLayer(DenseLayer):
    """
    Bridges a conv/pool feature map (N, C, H, W) into the 2D (N, features) shape that DenseLayer expects.
    Inherits from DenseLayer for interface consistency.
    """

    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        if input_shape is not None and len(input_shape) == 3:
            dims = int(np.prod(input_shape))
        else:
            dims = None
        super().__init__(input_dims=dims, output_dims=dims)
        self.x_shape = None

    def initialize_weights(self, activation_func=None):
        self.activation_func = activation_func
        self.weights = None
        self.bias = None

    def initialize_optimizer(self, *args, **kwargs):
        pass

    def forward_pass(self, x):
        self.prev_activation = x
        self.x_shape = x.shape
        self.z = x.reshape(x.shape[0], -1)
        self.A = self.z
        return self.A

    def backward_pass(self, prev_delta):
        return prev_delta.reshape(self.x_shape)

    def optimize_weights(self):
        pass


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

    def add(self, layer):
        """Append any pre-built layer object (Conv2DLayer, MaxPool2DLayer, FlattenLayer, ...)."""
        self.layers.append(layer)
        return getattr(layer, "output_shape", None)

    def add_conv2d(self, input_shape, num_filters, kernel_size, stride=1, padding=0,
                   activation="relu", optimizer=None, alpha=0.01, beta=0.9, gamma=0.99):
        optimizer = optimizer or self.optimizer

        layer = Conv2DLayer(input_shape, num_filters, kernel_size, stride, padding)
        layer.initialize_weights(activation)
        layer.initialize_optimizer(optimizer, alpha, beta, gamma)
        self.layers.append(layer)
        return layer.output_shape  # feed this straight into the next layer's input_shape

    def add_maxpool2d(self, input_shape, pool_size=2, stride=None):
        layer = MaxPool2DLayer(input_shape, pool_size, stride)
        self.layers.append(layer)
        return layer.output_shape

    def add_flatten(self, input_shape=None):
        layer = FlattenLayer(input_shape)
        self.layers.append(layer)

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

    def save_network(self,name="network",path=None):

        np_arr=np.array([layer for layer in self.layers]) 

        save_path=Path(path)/name if path else Path(name) 

        np.save(str(save_path),np_arr,allow_pickle=True) 

        print()
        print("Model saved to the path "+str(save_path)+"successfully") 


        
    def load_network(self,path):

        load_path = Path(path)
        if load_path.suffix != '.npy' and not load_path.exists():
            load_path = load_path.with_suffix('.npy')

        np_arr = np.load(str(load_path), allow_pickle=True)
        
        self.layers=[]

        for layer in np_arr:

            self.layers.append(layer) 
         
        print()
        print("model in the path"+ str(load_path) + "loaded successfully!")
        
       


