"""
A collection of python generators that, given the data,
returns a subset that should be trained on this iteration.
"""


def baby_steps(data, difficulty_measurer, num_buckets, epochs_per_bucket, reverse=False):
    """
    Implements the baby steps algorithm in page 7 of A Survey on Curriculum Learning

    difficulty_measurer is a function that sorts the data from easiest to hardest.
    num_buckets is the number of buckets we will train the model on.
    epochs_per_bucket is the number of epochs we need until we increase the difficulty.
    """
    # Split up the data into buckets
    sorted_data = difficulty_measurer(data, reverse)
    bucket_size = len(sorted_data) / num_buckets
    buckets = [sorted_data[i * bucket_size : (i + 1) * bucket_size] for i in range(num_buckets)]
    # Add the leftovers to the final bucket
    buckets[-1] += sorted_data[(num_buckets - 1) * bucket_size]

    # Yield buckets to train on
    current_data = []
    for current_bucket in range(num_buckets):
        current_data += buckets[current_bucket]
        for _ in range(epochs_per_bucket):
            yield current_data


def continuous_learning(data, difficulty_measurer, goal_accuracy, number_of_epochs, reverse=False):
    """
    Implements a continuous curriculum learning algorithm that
    uses the current validation accuracy to decide on how difficult the
    current data to train on should be.

    If the current accuracy is n% of the goal_accuracy, then
    we will train on first n% of the data sorted from easiest to hardest.

    difficulty_measurer is a function that sorts the data from easiest
    to hardest.

    number_of_epochs is the number of epochs we train the model for.

    Note: The user of this function should call .send() to report the current
    accuracy of the model.
    """
    # First, sort the data from easiest to hardest
    sorted_data = difficulty_measurer(data, reverse)

    # Then, yield the buckets to train on:
    for _ in range(number_of_epochs):
        # Gets the current accuracy
        current_accuracy = yield
        proportion_of_data = current_accuracy / goal_accuracy
        yield sorted_data[:int(len(sorted_data) * proportion_of_data)]