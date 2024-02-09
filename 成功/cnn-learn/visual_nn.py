"""
Visual Representations Implementation
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.cm as cm

from configure import Configuration
from model import myCNN


def vis_nn(configure: Configuration):
    device = configure.device
    exp_name = configure.exp_name
    valid_loader = configure.valid_loader

    # Load the trained model
    model = myCNN().to(device)
    state_dict = torch.load(f"{exp_name}_best.ckpt")
    model.load_state_dict(state_dict)
    model.eval()

    print(model)

    # Extract the representations for the specific layer of model
    index = 19  # You should find out the index of layer which is defined as "top" or 'mid' layer of your model.
    features = []
    labels = []
    for batch in tqdm(valid_loader):
        imgs, lbls = batch
        with torch.no_grad():
            logits = model.cnn[:index](imgs.to(device))
            logits = logits.view(logits.size()[0], -1)
        labels.extend(lbls.cpu().numpy())
        logits = np.squeeze(logits.cpu().numpy())
        features.extend(logits)

    features = np.array(features)
    colors_per_class = cm.rainbow(np.linspace(0, 1, 11))

    # Apply t-SNE to the features
    features_tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(features)

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=label, s=5)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    labels = [0]
    for label in np.unique(labels):
        plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=label, s=5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Your training and validation data
    train_losses = [1.06431, 0.45550, 0.30161, 0.23264, 0.19464, 0.15258, 0.13334, 0.12424]
    train_accs = [0.66544, 0.85483, 0.89986, 0.92542, 0.93990, 0.95298, 0.95757, 0.96046]

    valid_losses = [0.42691, 0.35227, 0.35170, 0.34853, 0.32489, 0.32720, 0.33710, 0.35164]
    valid_accs = [0.86553, 0.88521, 0.88998, 0.89354, 0.90413, 0.90550, 0.90440, 0.89952]

    # Epochs
    epochs = np.arange(1, len(train_losses) + 1)

    # Plotting loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, valid_losses, label='validation')
    plt.title('loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, valid_accs, label='validation')
    plt.title('accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Display the plot
    plt.show()
