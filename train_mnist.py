
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from neural_nets_from_scratch2 import Network


def one_hot(labels, num_classes=10):
    labels = labels.astype(int)
    out = np.zeros((labels.shape[0], num_classes))
    out[np.arange(labels.shape[0]), labels] = 1
    return out


def accuracy(net, X, y_true_labels):
    preds = net.predict(X)
    pred_labels = np.argmax(preds, axis=1)
    return np.mean(pred_labels == y_true_labels)


def main():
    print("Loading MNIST (this may take a minute on first run)...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

  
    X = X.astype(np.float64) / 255.0

   
    y_onehot = one_hot(y, num_classes=10)

    X_train, X_test, y_train_oh, y_test_oh, y_train_labels, y_test_labels = train_test_split(
        X, y_onehot, y, test_size=10000, random_state=42
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Build network: 784 -> 128 -> 64 -> 10
    net = Network(cost="mse")
    net.add_layer(784, 128, activation="relu", optimizer="rmsprop", alpha=0.001)
    net.add_layer(128, 64, activation="relu", optimizer="rmsprop", alpha=0.001)
    net.add_layer(64, 10, activation="sigmoid", optimizer="rmsprop", alpha=0.001)

    net.train(
        X_train, y_train_oh,
        epochs=30,
        batch_size=64,
        shuffle=True,
        verbose=True,
    )

    train_acc = accuracy(net, X_train, y_train_labels)

    test_acc = accuracy(net, X_test, y_test_labels)

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")


if __name__ == "__main__":
    main()
