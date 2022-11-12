# TODO: complete this file.

# - Bootstrap training set into 3 new training sets.
# - 3 different or same base models.
# - Average predictions from the 3 base models as final prediction.
# - Report final validation and test accuracy.

from utils import *
import numpy as np
#import sys
#np.set_printoptions(threshold=sys.maxsize) # To print full matrix/array instead of truncated ones.





def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    print(sparse_matrix)
 
    # Bootstrap train data:
    train_data_1 = ...
    train_data_2 = ...
    train_data_3 = ...

    ## TODO: Bootstrap on validation data too? Probably not since with the extra computation requied, we might as well just create a new classifier by bootstrapping on training data, which I personally think will create more variety and thus decrease the variance of final predictions.

    


    # Train model 1 (also tune hyperparameters based using validation data):




    # Train model 2 (also tune hyperparameters based using validation data):




    # Train model 3 (also tune hyperparameters based using validation data):




    # Average predictions of 3 models on validation and testing data (with tuned hyperparameters) for final results, then report them:
    # Note that technically can tune hyperparameters of three base models at the same time, but then the combinations of hyperparameters are way too many for runtime, so we will just pick the best of each world.
    


if __name__ == "__main__":
    main()

