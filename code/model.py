import sys

import numpy as np

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch

import matplotlib.pyplot as plt

from utils import data_to_matrix

from part_a.neural_network import AutoEncoder, train, evaluate


class Model:
    num_users: int
    num_questions: int

    def __init__(self, num_students, num_questions):
        self.num_users = num_students
        self.num_questions = num_questions

    def train(self, data, val_data, lr, iterations):
        raise NotImplementedError

    def evaluate(self, data, val_data):
        raise NotImplementedError

    def predict(self, data, user_id, question_id):
        raise NotImplementedError


class Curriculum_Learner(Model):
    def __init__(self, data, validation_data, num_students, num_questions, difficulty_measurer):
        """
            Makes the assumption that the difficulty_measurer inputted
            if of the right form, i.e.
        """
        self.difficulty_measurer = difficulty_measurer
        self.data = data
        self.validation_data = validation_data

        self.accuracy_record = [[], []]
        super().__init__(num_students, num_questions)

    def baby_steps(self, num_buckets, epochs_per_bucket, reverse=False, lr=0.1):
        """
        Implements the baby steps algorithm in page 7 of A Survey on Curriculum Learning

        difficulty_measurer is a function that sorts the data from easiest to hardest.
        num_buckets is the number of buckets we will train the model on.
        epochs_per_bucket is the number of epochs we need until we increase the difficulty.
        """
        def split_dict_into_buckets(data):
            bucket_size = len(sorted_data["user_id"]) // num_buckets
            user_buckets = [data["user_id"][i * bucket_size : (i + 1) * bucket_size] for i in range(num_buckets)]
            # Add the leftovers to the final bucket
            user_buckets[-1] += data["user_id"][(num_buckets - 1) * bucket_size:]

            question_buckets = [data["question_id"][i * bucket_size : (i + 1) * bucket_size] for i in range(num_buckets)]
            # Add the leftovers to the final bucket
            question_buckets[-1] += data["question_id"][(num_buckets - 1) * bucket_size:]

            is_correct_buckets = [data["is_correct"][i * bucket_size : (i + 1) * bucket_size] for i in range(num_buckets)]
            # Add the leftovers to the final bucket
            is_correct_buckets[-1] += data["is_correct"][(num_buckets - 1) * bucket_size:]

            return user_buckets, question_buckets, is_correct_buckets

        # Split up the data into buckets
        sorted_data = self.difficulty_measurer(self.data, self.num_questions, self.num_users, reverse)
        data_buckets = split_dict_into_buckets(sorted_data)

        current_data = {"user_id": [], "question_id": [], "is_correct": []}
        for current_bucket in range(num_buckets):
            current_data["user_id"] += data_buckets[0][current_bucket]
            current_data["question_id"] += data_buckets[1][current_bucket]
            current_data["is_correct"] += data_buckets[2][current_bucket]

            # c = list(zip(current_data["user_id"], current_data["question_id"], current_data["is_correct"]))
            # np.random.shuffle(c)
            # current_data["user_id"], current_data["question_id"], current_data["is_correct"] = map(list, zip(*c))

            self.train(
                current_data,
                self.validation_data,
                lr,
                epochs_per_bucket
            )
            self.record_accuracy()
            print("Completed Bucket", current_bucket)
            print()

    def continuous_learning(self, goal_accuracy, number_of_epochs, reverse=False, lr=0.1):
        """
        Implements a continuous curriculum learning algorithm that
        uses the current validation accuracy to decide on how difficult the
        current data to train on should be.

        If the current accuracy is n% of the goal_accuracy, then
        we will train on first n% of the data sorted from easiest to hardest.

        difficulty_measurer is a function that sorts the data from easiest
        to hardest.

        number_of_epochs is the number of epochs we train the model for.
        """
        def slice_data(data, proportion):
            size = len(data["user_id"]) * proportion
            return {"user_id": data["user_id"][:size],
                    "question_id": data["question_id"][:size],
                    "is_correct": data["is_correct"][:size]}


        sorted_data = self.difficulty_measurer(self.data, self.num_questions, self.num_users, reverse)
        sorted_validation = self.difficulty_measurer(self.validation_data, self.num_questions, self.num_users, reverse)

        # Then, yield the buckets to train on:
        for _ in range(number_of_epochs):
            # Gets the current accuracy
            current_accuracy = self.evaluate(self.data, self.sparse_validation)
            proportion = current_accuracy / goal_accuracy

            self.train(slice_data(sorted_data, proportion), slice_data(sorted_validation, proportion), lr, 10)
            self.record_accuracy()

    def record_accuracy(self):
        self.accuracy_record[0].append(self.evaluate(self.data, self.data))
        self.accuracy_record[1].append(self.evaluate(self.data, self.validation_data))

    def plot_accuracy(self):
        plt.plot(self.accuracy_record[0], label="Training Accuracy")
        plt.plot(self.accuracy_record[1], label="Validation Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


class AutoEncoder_Model(Curriculum_Learner):
    def __init__(self, data, validation_data, num_students, num_questions, difficulty_measurer):
        super().__init__(data, validation_data, num_students, num_questions, difficulty_measurer)
        self.model = AutoEncoder(num_questions)

    def _create_data_matrices(self, data):
        train_matrix = data_to_matrix(data, self.num_questions, self.num_users)
        zero_train_matrix = train_matrix.copy()
        # Fill in the missing entries to 0.
        zero_train_matrix[np.isnan(train_matrix)] = 0
        # Change to Float Tensor for PyTorch.
        zero_train_matrix = torch.FloatTensor(zero_train_matrix)
        train_matrix = torch.FloatTensor(train_matrix)

        return train_matrix, zero_train_matrix

    def train(self, data, val_data, lr, iterations):
        train_matrix, zero_train_matrix = self._create_data_matrices(data)
        train(self.model, lr, 0.1, train_matrix, zero_train_matrix, val_data, iterations)

    def evaluate(self, data, val_data):
        _, zero_train_matrix = self._create_data_matrices(data)
        return evaluate(self.model, zero_train_matrix, val_data)

    def predict(self, data, user_id, question_id):
        _, zero_train_matrix = self._create_data_matrices(data)
        inputs = Variable(zero_train_matrix[user_id]).unsqueeze(0)
        output = self.model(inputs)
        return output[0][question_id] > 0.5


class IRT_Model(Curriculum_Learner):
    theta: np.array
    beta: np.array

    def __init__(self, data, validation_data, num_students, num_questions, difficulty_measurer):
        super().__init__(data, validation_data, num_students, num_questions, difficulty_measurer)
        self.theta = np.zeros(self.num_users)
        self.beta = np.zeros(self.num_questions)

    @staticmethod
    def sigmoid(x):
        return np.exp(x) / (1 + np.exp(x))

    def _neg_log_likelihood(self, data):
        diffs = self.beta[data["question_id"]] - self.theta[data["user_id"]]
        signed_diffs = np.where(data["is_correct"], diffs, -diffs)
        return np.mean(np.logaddexp(0, signed_diffs))

    def _update_theta_beta(self, data, lr):
        user_id, question_id, is_correct = data["user_id"], data["question_id"], data["is_correct"]
        diffs = self.theta[user_id] - self.beta[question_id]
        sigmoid_diffs = lr * (is_correct - IRT_Model.sigmoid(diffs))

        theta_i_diffs = [[] for _ in self.theta]
        beta_j_diffs = [[] for _ in self.beta]
        for k in range(len(sigmoid_diffs)):
            theta_i_diffs[user_id[k]].append(sigmoid_diffs[k])
            beta_j_diffs[question_id[k]].append(sigmoid_diffs[k])

        theta_derivs = np.array([(np.mean(x) if len(x) > 0 else 0) for x in theta_i_diffs])
        beta_derivs = np.array([(np.mean(x) if len(x) > 0 else 0) for x in beta_j_diffs])
        self.theta = self.theta + theta_derivs
        self.beta = self.beta - beta_derivs

    def train(self, data, val_data, lr, iterations):
        for _ in range(iterations):
            self._update_theta_beta(data, lr)

    def evaluate(self, data, val_data):
        pred = []
        for i, q in enumerate(val_data["question_id"]):
            u = val_data["user_id"][i]
            x = (self.theta[u] - self.beta[q]).sum()
            p_a = IRT_Model.sigmoid(x)
            pred.append(p_a >= 0.5)
        return np.sum((val_data["is_correct"] == np.array(pred))) \
            / len(val_data["is_correct"])

    def predict(self, data, user_id, question_id):
        x = self.theta[user_id] - self.beta[question_id]
        return IRT_Model.sigmoid(x) >= 0.5


