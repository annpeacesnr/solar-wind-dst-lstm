import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure()
    for name, values in history.history.items():
        plt.plot(values, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
