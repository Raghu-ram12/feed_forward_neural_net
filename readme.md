# NumPy Neural Network

A lightweight, fully connected neural network implementation built from scratch using NumPy — no deep learning frameworks required. This project is designed for learning and experimentation with the core mechanics of forward propagation, backpropagation, and gradient-based optimization.

## Features

- **Custom Layers**: Simple `layer` class supporting fully connected (dense) layers with configurable input/output dimensions.
- **Multiple Activation Functions**:
  - ReLU
  - Leaky ReLU (with configurable `alpha`)
  - Sigmoid
  - Tanh
  - Linear (default fallback)
- **Multiple Optimizers**:
  - Vanilla Gradient Descent (`grad`)
  - Gradient Descent with Momentum (`grad-momentum`)
  - Adagrad
  - RMSProp
- **Modular Design**: Layers are added dynamically to a `NeuralNetwork` container, making it easy to build networks of arbitrary depth.

## Project Structure

```
├── layer            # Represents a single fully connected layer
└── NeuralNetwork     # Container class that manages layers, training, and prediction
```

### `layer`

Each `layer` instance handles:
- Weight initialization (He initialization for ReLU/Leaky ReLU, Xavier-style for Sigmoid/Tanh)
- Forward pass computation
- Activation and activation derivative calculations
- Backward pass (gradient computation)
- Weight updates based on the selected optimizer

### `NeuralNetwork`

The `NeuralNetwork` class handles:
- Stacking layers together
- Forward propagation through all layers
- Error/loss computation
- Backward propagation through all layers
- Training loop (`train`) and inference (`predict`)

## Usage

```python
import numpy as np
from neural_network import NeuralNetwork

# Initialize a network with an optimizer
nn = NeuralNetwork(optimizer="grad-momentum")

# Add layers: (input_dims, output_dims, activation)
nn.add_layer(4, 8, activation="relu")
nn.add_layer(8, 4, activation="relu")
nn.add_layer(4, 1, activation="sigmoid")

# Train
nn.train(X_train, Y_train, epochs=100, learning_rate=0.01)

# Predict
predictions = nn.predict(X_test)
```

## Configuration Options

### Activations
Set via the `activation` parameter when adding a layer:
- `"relu"`
- `"leaky-relu"`
- `"sigmoid"`
- `"tanh"`
- default (linear) if none of the above are specified

### Optimizers
Set at the `NeuralNetwork` level and applied to all layers:
- `"grad"` — standard gradient descent
- `"grad-momentum"` — gradient descent with momentum (`beta1`)
- `"adagrad"` — adaptive gradient algorithm
- `"rmsprop"` — root mean square propagation (`beta2`)

## Roadmap

- [ ] Configurable loss functions (currently uses mean squared error by default)
- [ ] Support for classification-oriented losses (e.g., cross-entropy)
- [ ] Batch training / mini-batch gradient descent
- [ ] Model saving/loading
- [ ] Additional activation functions (softmax, ELU, etc.)

## Requirements

- Python 3.x
- NumPy

## License

This project is currently under active development. License terms to be determined.