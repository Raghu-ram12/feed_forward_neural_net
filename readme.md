# Feedforward Neural Network from Scratch (NumPy)

A fully connected neural network implemented from scratch using only NumPy — no PyTorch, no TensorFlow. Built to understand the mechanics of forward propagation, backpropagation, and gradient-based optimization at the matrix-math level, and tested end-to-end on MNIST digit classification.

## Project Structure

```
├── neural_nets_from_scratch2.py   # Core library: Loss, Activation, DenseLayer, Network
├── train_mnist.py                 # End-to-end training script on the MNIST dataset
└── readme.md
```

## Core Components

### `Loss`
- **MSE** (`mse` / `mse_prime`) — mean squared error and its gradient
- **Binary Cross-Entropy** (`binary_cross_entropy` / `bce_prime`) — with epsilon clipping for numerical stability

### `Activation`
Mixed into `DenseLayer`. Implements both the function and its derivative for:
- ReLU
- Leaky ReLU (configurable `leaky_alpha`, default `0.01`)
- Sigmoid (numerically stable — branches on sign of `z` to avoid overflow)
- Tanh (uses `np.tanh` directly)

### `DenseLayer`
A single fully connected layer. Handles:
- **Weight initialization** — He initialization (`sqrt(2/n_in)`) for ReLU/Leaky ReLU, Xavier-style (`sqrt(1/n_in)`) for Sigmoid/Tanh, small random init otherwise
- **Forward pass** — linear transform + activation, caching `z`, `A`, and the input activation for use in backprop
- **Backward pass** — computes `dw`, `db` (both normalized by batch size `m`), and propagates the delta to the previous layer
- **Per-layer optimizer state** — each layer can independently use its own optimizer

### Optimizers (set per layer via `initialize_optimizer`)
- `"grad"` — vanilla gradient descent
- `"grad-momentum"` — momentum with configurable `beta`
- `"adagrad"` — adaptive learning rate from cumulative squared gradients
- `"rmsprop"` — exponential moving average of squared gradients, decay `gamma`

### `Network`
The container class that ties everything together:
- `add_layer(input_dims, output_dims, activation, optimizer, alpha, beta, gamma)` — appends a new `DenseLayer`, initializing its weights and optimizer. If `optimizer` is omitted, the layer inherits the `Network`'s default optimizer.
- `forward(x)` — runs input through every layer in sequence
- `backward(y)` — computes the loss gradient at the output layer, then backpropagates through layers in reverse, updating weights as it goes
- `train(X, y, epochs, batch_size, shuffle, verbose)` — full training loop supporting batch, mini-batch, or stochastic gradient descent depending on `batch_size`
- `predict(x_test)` — forward pass only, returns network output

Loss is configurable at the `Network` level via `cost="mse"` or `cost="binary-cross-entropy"` (the latter raises a warning if paired with a `tanh` output layer, since BCE expects outputs in `(0, 1)`).

## Usage

```python
import numpy as np
from neural_nets_from_scratch2 import Network

# Initialize a network with a default optimizer and loss
net = Network(optimizer="rmsprop", cost="mse")

# Add layers: (input_dims, output_dims, activation, optimizer=None, alpha=lr, beta, gamma)
net.add_layer(4, 8, activation="relu", alpha=0.01)
net.add_layer(8, 4, activation="relu", alpha=0.01)
net.add_layer(4, 1, activation="sigmoid", alpha=0.01)

# Train (batch_size=None -> full-batch, batch_size=1 -> SGD, else mini-batch)
net.train(X_train, y_train, epochs=100, batch_size=32, shuffle=True)

# Predict
predictions = net.predict(X_test)
```

## MNIST Example (`train_mnist.py`)

Trains a `784 → 128 → 64 → 10` network on MNIST digits, downloaded via `sklearn.datasets.fetch_openml`:

- Inputs scaled to `[0, 1]`, labels one-hot encoded
- Architecture: ReLU → ReLU → Sigmoid, all layers using `rmsprop` (`alpha=0.001`)
- Loss: MSE
- 30 epochs, mini-batches of 64, with train/test accuracy reported at the end (via `argmax` over the one-hot outputs)

Run it with:

```bash
python train_mnist.py
```

**Dependencies:** `numpy`, `scikit-learn` (for `fetch_openml` and `train_test_split`)

## Known Limitations / Roadmap

- No dedicated multi-class loss (cross-entropy) or softmax activation yet — MNIST currently trains with sigmoid outputs + MSE, which works but converges more slowly than a proper softmax + categorical cross-entropy setup would
- No model saving/loading
- No validation split / early stopping during training
- No gradient checking or unit tests

## Requirements

- Python 3.x
- NumPy
- scikit-learn (only needed for `train_mnist.py`)