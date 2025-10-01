import numpy as np
import matplotlib.pyplot as plt

from toolbox.csv_loader import load_data


def plot_constellation(csvfile, title):
    data = load_data(csvfile)
    x, y, bits = data["x"], data["y"], data["bits"]
    base_colors = plt.get_cmap('tab20')(np.linspace(0, 1, 16))
    plt.scatter(x, y, c=[base_colors[b] for b in bits], s=6)

    plt.title(title)
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True, alpha=0.6)
    plt.axis('equal')
    plt.axhline(0, color='k', linewidth=1)
    plt.axvline(0, color='k', linewidth=1)

    plt.show()


if __name__ == "__main__":
    # Original 16QAM Constellation
    # plot_constellation("test5.csv", "Original Noisy 16QAM Constellation")

    # SVM Distilled Constellation
    # plot_constellation("../dd_svm/distilled_dataset.csv", "SVM Distilled 16QAM Constellation")

    # GAN Distilled Constellation
    plot_constellation("../dd_gan/distilled_test5_2000.csv", "Original Noisy 16QAM Constellation")
