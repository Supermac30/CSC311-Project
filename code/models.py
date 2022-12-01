import numpy as np


class Model:

    num_users: int
    num_questions: int

    def __init__(self, num_students, num_questions):
        self.num_users = num_students
        self.num_questions = num_questions

    def train(self, data, val_data, lr, iterations):
        pass

    def evaluate(self, data):
        pass

    def predict(self, user_id, question_id):
        pass


class IRT_Model(Model):
    theta: np.array
    beta: np.array

    def __init__(self):
        super().__init__()
        self.beta = np.zeros(self.num_users)
        self.theta = np.zeros(self.num_questions)

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

        theta_i_diffs = [[] for _ in theta]
        beta_j_diffs = [[] for _ in beta]
        for k in range(len(sigmoid_diffs)):
            theta_i_diffs[user_id[k]].append(sigmoid_diffs[k])
            beta_j_diffs[question_id[k]].append(sigmoid_diffs[k])

        theta_derivs = np.array([np.mean(x) for x in theta_i_diffs])
        beta_derivs = np.array([np.mean(x) for x in beta_j_diffs])
        theta = theta + theta_derivs
        beta = beta - beta_derivs

    def train(self, data, val_data, lr, iterations):
        for _ in range(iterations):
            self.theta, self.beta = self._update_theta_beta(data, lr)

    def evaluate(self, data):
        pred = []
        for i, q in enumerate(data["question_id"]):
            u = data["user_id"][i]
            x = (self.theta[u] - self.beta[q]).sum()
            p_a = IRT_Model.sigmoid(x)
            pred.append(p_a >= 0.5)
        return np.sum((data["is_correct"] == np.array(pred))) \
            / len(data["is_correct"])

    def predict(self, user_id, question_id):
        x = self.theta[user_id] - self.beta[question_id]
        return IRT_Model.sigmoid(x) >= 0.5
