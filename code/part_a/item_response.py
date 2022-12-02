from utils import *

import numpy as np
import matplotlib.pyplot as plt
import json


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################

    # theta[i]-beta[j]
    diffs = beta[data["question_id"]] - theta[data["user_id"]]
    signed_diffs = np.where(data["is_correct"], diffs, -diffs)
    log_likelihood = -np.mean(np.logaddexp(0, signed_diffs))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_likelihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id, question_id, is_correct = data["user_id"], data["question_id"], data["is_correct"]
    # update theta
    diffs = theta[user_id] - beta[question_id]
    sigmoid_diffs = lr * (is_correct - sigmoid(diffs))

    theta_i_diffs = [[] for _ in theta]
    for k in range(len(sigmoid_diffs)):
        theta_i_diffs[user_id[k]].append(sigmoid_diffs[k])

    theta_derivs = np.array([np.mean(x) for x in theta_i_diffs])
    theta = theta + theta_derivs
    # print(theta[0:5])
    # beta0 = beta[0:5]
    diffs = theta[user_id] - beta[question_id]
    sigmoid_diffs = lr * (is_correct - sigmoid(diffs))
    beta_j_diffs = [[] for _ in beta]
    for k in range(len(sigmoid_diffs)):
        beta_j_diffs[question_id[k]].append(sigmoid_diffs[k])

    beta_derivs = np.array([np.mean(x) for x in beta_j_diffs])

    beta = beta - beta_derivs

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_accs, train_neg_lld_list, val_neg_lld_list)
    """
    # Initialize theta and beta.
    num_users = max(data["user_id"]) + 1
    num_questions = max(data["question_id"]) + 1
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    val_accs = []

    train_neg_lld_list = []
    val_neg_lld_list = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta, beta)
        train_neg_lld_list.append(neg_lld)
        val_neg_lld_list.append(val_neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_accs.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_accs, train_neg_lld_list, val_neg_lld_list
    # return {"theta":theta, "beta":beta, "val_accs": val_accs, "train_neg_lld_list": train_neg_lld_list, "val_neg_lld_list": val_neg_lld_list}


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def save_hyper_param_data(file_name: str, irts, learning_rates):
    with open(file_name, "w") as f:
        cleaned_up = {}
        for i in range(len(irts)):
            theta, beta, val_accs, train_llds, val_llds = irts[i]
            cleaned_up[learning_rates[i]] = {"theta": list(theta), "beta": list(
                beta), "val_acc_list": val_accs, "train_llds": train_llds, "val_llds": val_llds}
        json.dump(cleaned_up, f)


def get_hyper_param_data(file_name: str):
    with open(file_name, "r") as f:
        data_obj = json.load(f)

    irts = []
    learning_rates = sorted(list(data_obj.keys()))
    for lr in learning_rates:
        run_obj = data_obj[lr]
        irts.append((np.array(run_obj["theta"]), np.array(
            run_obj["beta"]), run_obj["val_acc_list"], run_obj["train_llds"], run_obj["val_llds"]))

    learning_rates = [float(lr) for lr in learning_rates]

    return learning_rates, irts


def plot_lld_curves(train_llds, val_llds, save_file_name):
    fig, (ax_train, ax_val) = plt.subplots(2, sharey=True)

    ax_train.set(xlabel="iteration", ylabel="negative log likelihood",
                 title="training set")
    ax_val.set(xlabel="iteration", ylabel="negative log likelihood",
               title="validation set")
    max_iterations = len(train_llds)
    ax_train.plot(range(max_iterations), train_llds)
    ax_val.plot(range(max_iterations), val_llds)
    ax_train.label_outer()
    ax_val.label_outer()
    fig.tight_layout()

    fig.savefig("IRT_lld_curves.png", bbox_inches='tight')
    plt.close(fig)


def part_d(theta, beta, question_nums, save_file_name):
    beta_vals = beta[question_nums].reshape(-1, 1)
    theta_vals = np.arange(-5, 5, 0.1)
    diffs = np.apply_along_axis(lambda x: theta_vals - x, 1, beta_vals)
    curves = sigmoid(diffs)
    for i in range(len(question_nums)):
        plt.plot(theta_vals, curves[i], label=f"Question {question_nums[i]}")

    plt.xlabel = r"$\theta_i$"
    plt.ylabel = r"$p(c_{ij} | \theta_i, \beta_j)$"
    plt.legend()
    plt.savefig(save_file_name, bbox_inches='tight')


def hyperparam_grid_search(train_data, val_data, lr_incr, num_lr_incr, max_iter, save_file='save_hyper_param_runs.json'):
    learning_rates = [(lr_incr * i) for i in range(1, num_lr_incr+1)]
    irts = [irt(train_data, val_data, lr, max_iter) for lr in learning_rates]
    save_hyper_param_data(save_file, irts, learning_rates)
    val_accs = np.array([x[2] for x in irts])
    opt_lr_index, opt_iters = np.unravel_index(
        np.argmax(val_accs), val_accs.shape)
    opt_lr = learning_rates[opt_lr_index]
    #  theta, beta, val_accs, train_llds, val_llds = irts[opt_lr_index]
    return opt_lr, opt_iters


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    # save_hyper_param_data('save_hyper_param_runs2.json', irts, learning_rates)
    # learning_rates, irts = get_hyper_param_data('save_hyper_param_runs.json')

    # opt_lr, opt_iters = hyperparam_grid_search(train_data, val_data, 0.005, 20, 500)
    opt_lr = 0.065
    opt_iters = 5

    theta, beta, val_accs, train_llds, val_llds = irt(
        train_data, val_data, opt_lr, opt_iters)
    print(f"optimal learning rate and iterations is {opt_lr, opt_iters}")
    print(f"final train accuracy is {evaluate(train_data, theta, beta)}")
    print(f"final validation accuracy is {evaluate(val_data, theta, beta)}")
    print(f"final test accuracy is {evaluate(test_data, theta, beta)}")

    plot_lld_curves(train_llds, val_llds, "lld_curves.png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Implement part (d)                                                #
    #####################################################################
    part_d(theta, beta, [3, 10, 100], "ITRd_question_curves.png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
