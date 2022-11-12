# TODO: complete this file.

# - Bootstrap training set into 3 new training sets.
# - 3 different or same base models.
# - Average predictions from the 3 base models as final prediction.
# - Report final validation and test accuracy.

from utils import *
from knn import *
import numpy as np
import os
import csv
import sys
np.set_printoptions(threshold=sys.maxsize) # To print full matrix/array instead of truncated ones.


def _csv_to_list_of_line(path):
    # A helper function to load csv to list of list.
    # Specifically to format of train_data.csv (or val/test...), where each line (other than first line) is of form <question_id>, <user_id>, <is_correct>.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    with open(path, "r") as csv_file: # Python "with" statement closes file itself once we leave the "with block"!
        reader = csv.reader(csv_file)
        list_of_line = list(reader)
        list_of_line.pop(0)
        list_of_line = [[int(s) for s in x] for x in list_of_line]
        return list_of_line


def main():
    #sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    #print(sparse_matrix)
 

    # Bootstrap train data:
    #   1. Load csv to list of lines.
    original_train_data = _csv_to_list_of_line("../data/train_data.csv")
    

    #   2. Use numpy.random.choice (with replacement) to select lines for new data.
    train_size = len(original_train_data)
    train_data_1 = [original_train_data[x] for x in np.random.choice(train_size, train_size, replace=True)]
    train_data_2 = [original_train_data[x] for x in np.random.choice(train_size, train_size, replace=True)]
    train_data_3 = [original_train_data[x] for x in np.random.choice(train_size, train_size, replace=True)]


    #   3. Find out matrix dimension by finding largest value of <user_id> among all (similarly for <question_id>. NVM, this doesn't work because in handout, we have 542 students, but ina ny data files, we only have 541 as max. Will go with handout (though in npz matrix file we have 542 students).
    #train_data_1 = np.array(train_data_1)
    #print(np.amax(train_data_1, axis=0))
    user_id_dim = 542
    question_id_dim = 1774


    #   4. Convert to numpy matrix form by initially setting it to matrix full of NaN, then indexing to set some entries to 0/1.
    train_matrix_1 = np.empty((user_id_dim, question_id_dim))
    train_matrix_1[:] = np.NaN
    for line in train_data_1: # Set up 0/1 for entries.
        train_matrix_1[line[1], line[0]] = line[2]
    
    train_matrix_2 = np.empty((user_id_dim, question_id_dim))
    train_matrix_2[:] = np.NaN
    for line in train_data_2: # Set up 0/1 for entries.
        train_matrix_2[line[1], line[0]] = line[2]
    

    train_matrix_3 = np.empty((user_id_dim, question_id_dim))
    train_matrix_3[:] = np.NaN
    for line in train_data_3: # Set up 0/1 for entries.
        train_matrix_3[line[1], line[0]] = line[2]


    ## TODO: Bootstrap on validation data too? Probably not since with the extra computation requied, we might as well just create a new classifier by bootstrapping on training data, which I personally think will create more variety and thus decrease the variance of final predictions.



    # Train model 1 (also tune hyperparameters based using validation data):
    # KNN_inputed_by_user model:
    k_val = [1, 6, 11, 16, 21, 26]
    knn_best_k = k_val[0]
    knn_best_acc = 0_
    for k in k_val:
        acc_temp = knn_impute_by_user(train_matrix_1, val_data, k)
        if (acc_temp > knn_best_acc):
            knn_best_acc = acc_temp
            knn_best_k = k

    # Train model 2 (also tune hyperparameters based using validation data):
    # IRT model:
    # TODO: Train and find best parameter using <train_matrix_2>!



    # Train model 3 (also tune hyperparameters based using validation data):
    # Neural nets model:
    # TODO: Train and find best parameter using <train_matrix_3>!



    # Average predictions of 3 models on validation and testing data (with tuned hyperparameters) for final results, then report them:
    # Note that technically can tune hyperparameters of three base models at the same time, but then the combinations of hyperparameters are way too many for runtime, so we will just pick the best of each world.
    
    # KNN predict on validation and test set:
    nbrs = KNNImputer(n_neighbors=knn_best_k)
    pred_valid_knn = nbrs.fit_transform(train_matrix_1) 
    pred_test_knn = pred_valid_knn
    
    # IRT predict on validation and test set:
    pred_valid_irt = ...
    pred_test_irt = ...


    # Neural nets predict on validation and test set:
    pred_valid_neural_net = ...
    pred_test_neural_net = ...
    

    # Average final predictions on validation and test set:
    pred_valid_final = (pred_valid_knn + pred_valid_irt + pred_valid_neural_net) / 3
    pred_test_final = (pred_test_knn + pred_test_irt + pred_test_neural_net) / 3

    # Report/print results!
    acc_valid_final = sparse_matrix_evaluate(val_data, pred_valid_final) 
    acc_test_final = sparse_matrix_evaluate(test_data, pred_test_final)
    print("Final validation accuracy: " + str(acc_valid_final))
    print("Final test accuracy: " + str(acc_test_final))


if __name__ == "__main__":
    main()

