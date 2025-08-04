import matplotlib.pyplot as plt

def plot_loss(train_loss, test_loss, save_path=None):
    """
    Plots training and test loss curves.
    """
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, test_loss, marker="o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_accuracy(train_acc, test_acc, save_path=None):
    """
    Plots training and test accuracy curves.
    """
    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
    plt.plot(epochs, test_acc, marker="o", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    train_loss = [0.6928332329417268, 0.194095296934247, 0.11588891198237737, 0.09429386264955004, 0.08131259231207272, 0.07002846148912795, 0.06519407248376713, 0.05933527468985024, 0.054962590598734096, 0.05346418419392042] 
    train_accuracy = [80.36, 93.98666666666666, 96.67166666666667, 97.36, 97.68833333333333, 98.00166666666667, 98.14833333333334, 98.30166666666666, 98.41499999999999, 98.53166666666667] 
    test_loss = [0.07691371543798596, 0.04695178278372623, 0.04214055527292657, 0.032474400950595735, 0.030258921243948862, 0.033908909980091266, 0.033033393171208444, 0.03626859174517449, 0.0458122877535061, 0.038057960965888926] 
    test_accuraacy = [97.63, 98.67, 98.81, 99.0, 98.95, 98.96000000000001, 98.99, 99.11999999999999, 98.74000000000001, 98.97]

    plot_loss(train_loss, test_loss)
    plot_accuracy(train_accuracy, test_accuraacy)

