"""
A collection of various methods for sorting the data from easiest
to hardest.

All data is assumed to be in the dictionary form, not a sparse matrix
"""

import numpy as np
import torch
from part_a.neural_network import AutoEncoder
from utils import *


def _sort_data(data, function, reverse):
    """A helper function for sorting the data given an ordering on the users.
    function is a function that takes in the user idea and returns a value. The ordering
    produced is from largest to smallest, if reverse is True, and smallest to largest otherwise.
    """
    user_array = [(x, i) for i, x in enumerate(data["user_id"])]
    user_array = sorted(user_array, key=lambda n: function(n[0]), reverse=not reverse)
    sorted_users = [data["user_id"][i] for (_, i) in user_array]
    sorted_questions = [data["question_id"][i] for (_, i) in user_array]
    sorted_is_correct = [data["is_correct"][i] for (_, i) in user_array]

    return {"user_id": sorted_users, "question_id": sorted_questions, "is_correct": sorted_is_correct}


def autoencoder_difficulty(data, num_questions, num_students, reverse=False):
    """Sorts the data by looking at the accuracy of the autoencoder when reconstructing it."""
    model = torch.load("Autoencoder.pt")
    train_matrix = data_to_matrix(data, num_questions, num_students)
    zero_train_matrix = train_matrix.copy()

    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)

    # Reconstruct the data using the model
    reconstructions = model(zero_train_matrix)

    def _number_correct(user):
        return np.sum(train_matrix[user] == np.round(reconstructions[user].detach().numpy()))

    return _sort_data(data, _number_correct, reverse)


def number_of_entries_difficulty(data, num_questions, num_students, reverse=False):
    """Sorts the data by looking at the number of questions answered by each user"""
    return _sort_data(data, lambda n: data["user_id"].count(n), reverse)

