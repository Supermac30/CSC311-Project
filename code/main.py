from model import *
from utils import *
from difficulty_measurers import *

data = load_train_csv("data")
valid_data = load_valid_csv("data")
test_data = load_public_test_csv("data")
num_users = len(set(data["user_id"]))
num_questions = len(set(data["question_id"]))


def test_autoencoder_baby_steps(difficulty_measure, num_iterations_per_bucket, num_buckets, reverse):
    """
    Run an experiment where we use an AutoEncoder with
    the baby-steps algorithm

    difficulty_measure is a function that measures difficulty
    num_iterations_per_bucket is the number of iterations per bucket
    num_buckets is the number of buckets
    reverse is True if we train from hardest to easiest, and False otherwise.
    """

    model = AutoEncoder_Model(data, valid_data, num_users, num_questions, difficulty_measure)
    model.baby_steps(num_buckets, num_iterations_per_bucket, reverse)
    print("Test Accuracy:", model.evaluate(data, test_data))
    model.plot_accuracy()


def test_autoencoder_linear_continuous_learning(difficulty_measure, num_iterations, reverse):
    """
    Run an experiment where we use an AutoEncoder with
    the continuous learning algorithm.

    The proportion function is as follows: lambda(t) = min(1, slope * t + bias)


    difficulty_measure is a function that measures difficulty
    num_iterations_per_bucket is the number of iterations per bucket
    num_buckets is the number of buckets
    reverse is True if we train from hardest to easiest, and False otherwise.
    """
    model = AutoEncoder_Model(data, valid_data, num_users, num_questions, difficulty_measure)

    slope = 0.001
    bias = 0.5

    def proportion_function(epoch_number):
        return min(1, bias + slope * epoch_number)

    model.continuous_learning(num_iterations, proportion_function, reverse)
    print("Test Accuracy:", model.evaluate(data, test_data))
    model.plot_accuracy()


def test_autoencoder_continuous_validation_learning(difficulty_measure, goal_accuracy, num_iterations, reverse):
    """
    Run an experiment where we use an AutoEncoder with
    the baby-steps algorithm and the number of entries difficulty measure
    """

    model = AutoEncoder_Model(data, valid_data, num_users, num_questions, difficulty_measure)
    model.continuous_validation_learning(goal_accuracy, num_iterations, reverse)
    print("Test Accuracy:", model.evaluate(data, test_data))
    model.plot_accuracy()


def test_irt_baby_steps(difficulty_measure, num_iterations_per_bucket, num_buckets, reverse):
    """
    Run an experiment where we use an AutoEncoder with
    the baby-steps algorithm and the number of entries difficulty measure

    difficulty_measure is a function that measures difficulty
    num_iterations_per_bucket is the number of iterations per bucket
    num_buckets is the number of buckets
    reverse is True if we train from hardest to easiest, and False otherwise.
    """

    model = IRT_Model(data, valid_data, num_users, num_questions, difficulty_measure)
    model.baby_steps(num_buckets, num_iterations_per_bucket, reverse, lr=0.065)
    print("Test Accuracy:", model.evaluate(data, test_data))
    model.plot_accuracy()

def test_irt_linear_continuous_learning(difficulty_measure, num_iterations, reverse):
    model = IRT_Model(data, valid_data, num_users, num_questions, difficulty_measure)

    slope = 0.001
    bias = 0.5

    def proportion_function(epoch_number):
        return min(1, bias + slope * epoch_number)

    model.continuous_learning(num_iterations, proportion_function, reverse, lr=0.065)
    print("Test Accuracy:", model.evaluate(data, test_data))
    model.plot_accuracy()

def test_irt_continuous_validation_learning(difficulty_measure, goal_accuracy, num_iterations, reverse):
    """
    Run an experiment where we use an AutoEncoder with
    the baby-steps algorithm and the number of entries difficulty measure
    """

    model = IRT_Model(data, valid_data, num_users, num_questions, difficulty_measure)
    model.continuous_validation_learning(goal_accuracy, num_iterations, reverse, lr=0.065)
    print("Test Accuracy:", model.evaluate(data, test_data))
    model.plot_accuracy()


if __name__ == "__main__":
    # np.random.seed(42)
    test_irt_continuous_validation_learning(number_of_entries_difficulty, 0.8, 500, False)
    test_irt_continuous_validation_learning(autoencoder_difficulty, 0.8, 500, False)

    test_irt_linear_continuous_learning(number_of_entries_difficulty, 500, False)
    test_irt_linear_continuous_learning(autoencoder_difficulty, 500, False)

    test_irt_baby_steps(number_of_entries_difficulty, 20, 25, False)
    test_irt_baby_steps(autoencoder_difficulty, 20, 25, False)


    test_autoencoder_continuous_validation_learning(number_of_entries_difficulty, 0.8, 500, False)
    test_autoencoder_continuous_validation_learning(autoencoder_difficulty, 0.8, 500, False)

    test_autoencoder_linear_continuous_learning(number_of_entries_difficulty, 500, False)
    test_autoencoder_linear_continuous_learning(autoencoder_difficulty, 500, False)

    test_irt_baby_steps(number_of_entries_difficulty, 20, 25, False)
    test_irt_baby_steps(autoencoder_difficulty, 20, 25, False)