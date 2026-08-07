# Simple NumPy Neural Network

A minimal feedforward neural network implemented from scratch with NumPy — no deep learning framework required. Supports arbitrary depth, several activation functions, and trains with plain (full-batch) gradient descent.

## Features

- Fully connected (`Dense`) layers, added one at a time
- Activations: `relu`, `leaky-relu`, `tanh`, `sigmoid`, or none (linear/identity)
- He initialization for ReLU-family activations, Xavier-style initialization for sigmoid/tanh
- Manual forward pass and backpropagation (no autograd)
- Mean squared error loss

## Requirements

```
numpy
```

Install with:

```bash
pip install numpy
```

## Usage

```python
import numpy as np
from neural_network import NeuralNetwork

# Toy dataset
X = np.random.randn(100, 3)
Y = (np.sum(X, axis=1, keepdims=True) > 0).astype(float)

# Build the network
nn = NeuralNetwork()
nn.add_layer(input_dims=3, output_dims=8, activation="relu")
nn.add_layer(input_dims=8, output_dims=1, activation="sigmoid")

# Train
nn.train(X, Y, epochs=500, learning_rate=0.5)

# Predict
predictions = nn.predict(X)
```

## API

### `NeuralNetwork`

| Method | Description |
|---|---|
| `add_layer(input_dims, output_dims, activation="relu")` | Appends a new layer. `input_dims` must match the previous layer's `output_dims` (or the input feature count, for the first layer). |
| `train(X_train, Y_train, epochs=100, learning_rate=0.01)` | Runs full-batch forward + backward passes for the given number of epochs. |
| `predict(x_test)` | Runs a forward pass and returns the network's output. |

### `Layer`

Each layer stores its own weights (`Weight`), bias (`bias`), and gradients (`dw`, `db`), and knows how to compute its activation and that activation's derivative. You generally won't need to touch `Layer` directly — use `NeuralNetwork.add_layer`.

## Notes / Limitations

- **Loss function**: the network is wired for **mean squared error** only. The output-layer gradient (`output - actual_data`) is the correct MSE gradient; if you add a `softmax` output for classification, you'll need cross-entropy loss instead, which has a different gradient formula.
- **Optimizer**: plain (vanilla) gradient descent, full-batch — no momentum, Adam, or mini-batching. Every call to `train()` uses the entire `X_train` as one batch per epoch.
- **Batch-size normalization**: the `1/m` averaging is applied once, inside each layer's `backward_pass`, when computing `dw`/`db`. It should not be applied anywhere else (e.g. to the raw output error) — doing so would double-count the averaging and shrink gradients by an extra factor of the batch size in earlier layers.
- No bias/weight regularization (L1/L2, dropout, etc.) is implemented.


