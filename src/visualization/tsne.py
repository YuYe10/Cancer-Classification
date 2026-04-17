from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(X, y, save_path):
    tsne = TSNE(n_components=2)
    X_2d = tsne.fit_transform(X)

    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    plt.title("t-SNE")
    plt.savefig(save_path)