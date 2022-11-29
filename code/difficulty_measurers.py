"""
A collection of various methods for sorting the data from easiest
to hardest.

All data is assumed to be a sparse matrix
"""

import numpy as np
import torch
from part_a.neural_network import AutoEncoder
from utils import *

def autoencoder_difficulty(sparse_matrix, reverse=False):
    """Sorts the data by looking at the accuracy of the autoencoder when reconstructing it."""
    model = torch.load("Autoencoder.pt")
    data = sparse_matrix.toarray()

    # Put the data in a form the autoencoder can understand,
    # By replace missing entries with 0s, and turning it into a tensor
    zero_data = data.copy()
    zero_data[np.isnan(data)] = 0
    zero_data = torch.FloatTensor(zero_data)

    # Reconstruct the data using the model
    reconstructions = model(zero_data)

    def _number_correct(user, user_reconstruction):
        count = 0
        for i in range(len(user)):
            if user[i] == round(user_reconstruction[i].item()):
                count += 1
        return count

    sorted_reconstructions = sorted(
        [(zero_data[i], reconstructions[i], i) for i in range(len(reconstructions))],
        key=lambda n: _number_correct(n[0], n[1]),
        reverse=not reverse
    )

    sorted_data = np.array(data)
    for i in range(len(sorted_reconstructions)):
        sorted_data[i] = data[sorted_reconstructions[i][2]]

    breakpoint()

    return sorted_data

def number_of_entries_difficulty(sparse_matrix, reverse=False):
    """Sorts the data by looking at the number of questions answered by each user"""
    return sorted(sparse_matrix, key=lambda n: np.count_nonzero(np.isnan(n)), reverse=not reverse)

if __name__ == "__main__":
    h = autoencoder_difficulty(load_train_sparse("data"))
    breakpoint()