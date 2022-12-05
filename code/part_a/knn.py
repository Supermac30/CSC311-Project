from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize) # To print full matrix/array instead of truncated ones.


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    
    mat = nbrs.fit_transform(matrix)

    #print(matrix, mat)

    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
 
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    
    mat = (nbrs.fit_transform(matrix.transpose())).transpose()

    #print(matrix[0], mat[0])

    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #print("Sparse matrix:")
    #print(sparse_matrix)
    #print("Shape of sparse matrix:")
    #print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_val = [1, 6, 11, 16, 21, 26]
    valid_acc_user = []
    for k in k_val:
        acc_temp = knn_impute_by_user(sparse_matrix, val_data, k)
        print("knn_impute_by_user Validation Accuracy: {}".format(acc_temp) + " with k = " + str(k))
        valid_acc_user.append(acc_temp)
    plt.plot(k_val, valid_acc_user)
    plt.xlabel('k_val')
    plt.ylabel('validation accuracy')
    plt.title('Accuracy Of knn_impute_by_user On Validation Set')
    plt.show()

    k_best = k_val[valid_acc_user.index(max(valid_acc_user))]
    print("knn_impute_by_user: k_best = " + str(k_best) + "; final_test_accuracy = " + str(knn_impute_by_user(sparse_matrix, test_data, k_best)))

    valid_acc_item = []
    for k in k_val:
        acc_temp = knn_impute_by_item(sparse_matrix, val_data, k)
        print("knn_impute_by_item Validation Accuracy: {}".format(acc_temp) + " with k = " + str(k))
        valid_acc_item.append(acc_temp)
    plt.plot(k_val, valid_acc_item)
    plt.xlabel('k_val')
    plt.ylabel('validation accuracy')
    plt.title('Accuracy Of knn_impute_by_item On Validation Set')
    plt.show()

    k_best = k_val[valid_acc_item.index(max(valid_acc_item))]
    print("knn_impute_by_item: k_best = " + str(k_best) + "; final_test_accuracy = " + str(knn_impute_by_item(sparse_matrix, test_data, k_best)))
        
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
