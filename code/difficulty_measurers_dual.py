"""
Question-based difficulty measure. Rank questions' difficulty (to our ML algo, not how "hard" a question is conceptually) depending on its subjects.

Will rank questions based on the "most difficult" subject it belongs to. Because it's hard to get questions correct since you need to understand everything. Thus it makes intuitive sense to foucs on the "most difficult subject" a question involves.

When ranking difficulty-level of subject, first 

"Difficult" subject here refers to subject that is hard for ML.
"""

import numpy as np
from utils import *
import math


"""
Rank subjects by the number of occurrence in all questions. "Less occurrence" = "More difficult for ML".

Return dictionary: dict[subject_id] = occurrence.
"""
def subject_occurrence():
    # 1. Work with "question_meta.csv". Check out <load_csv>.
    # 2. Transform to all elements into an array of id, not counting '0' as it occurs everywhere.
    # 3. Use Python list method to count occurrnece.
    
    # TODO: This function is part of <subject_correctness_entropy>.
    



"""
Rank subjects by the correctness rate. Perhaps use entropy. "Higher entropy" = "More difficult for ML".

Return a dictionary: dict[subject_id] = entropy.
"""
def subject_correctness_entropy():
    # 1. Work with "question_meta.csv" then put into dictionary form: dict[question_id] = a list of subject_id.
    question_meta_path = ... # TODO: path.
    question_meta_dict = {}
    with open(question_meta_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                subject_list = []
                for item in list(row[1]):
                    try: 
                        temp = int(item);
                        subject_list.append(temp)
                    except ValueError: # To filter out non-integer ones.
                        pass
                question_meta_dict[int(row[0])] = subject_list 
            except ValueError: # Pass first row.
                pass


    # 2. Work with "train_data.csv". Or maybe use "train_sparse_matrix" then use <load_npz>.
    sparse_matrix = load_train_sparse(...).toarray().transpose() # TODO: path! # Question-base matrix!


    # 3. Create a dictionary <subject_correctness_rate_dict>: dict[subject_id] = (num_occurrence = 0, correctness_rate = 0). If subject doesn't appear in any question that doesn't matter. If it does, then must start with 0.
    num_subject = 388 # According to "subject_meta.csv"
    subject_correctness_rate_dict = {}
    for i in range(num_subject):
        subject_correctness_rate_dict[i] = [0, 0, 0, 0]; # [a, b, c, d] where <a> is number of questions that involves this subject; <b> is total number of questions answer right; <c> is total number of questions that is answered; <d> is correctness rate. # TODO: <a> may be useless!


    # 4. Iterating through <question_meta_dict> for each question:
    #       4a. Find its correctness rate in train data by indexing the row in matirx then count and compute correctness rate. 
    #       4b. Go though each subject_id then add value then normalize (by division of <num_occurrence + 1>), then update <num_occurrence>.
    #   4
    #   +
    #   5
    # 5. Create new dict <subject_entropy>: dict[subject_id] = entropy = Formula with p(getting correct) = Correctness_rate.

    for item in question_meta_dict.items():
        question_id = item[0]
        subject_list = item[1]
        num_correct = len(np.nonzero(sparse_matrix[question_id] == 1))
        num_all = num_correct + len(np.nonzero(sparse_matrix[question_id] == 0))
        for subject in subject_list:
            subject_correctness_rate_dict[subject][0] += 1
            subject_correctness_rate_dict[subject][1] += num_correct
            subject_correctness_rate_dict[subject][2] += num_all
    
    subject_entropoy_dict = {}

    for subject in subject_correctness_rate_dict:
        # subject_correctness_rate_dict[subject][3] = subject_correctness_rate_dict[subject][1] / subject_correctness_rate_dict[subject][2]
        num_correct = subject_correctness_rate_dict[subject][1] 
        num_all = subject_correctness_rate_dict[subject][2]
        correctness_rate = num_correct / num_all
        p_1 = correctness_rate
        p_2 = 1 - p_1
        if (p_1 == 0):
            w_1 = 1 # Any finite number, just avoidning infinity.
            w_2 = math.log2(p_2)
        elif (p_2 == 0):
            w_1 = math.log2(p_1)
            w_2 = 1
        else:
            w_1 = math.log2(p_1) # Maybe natural log faster? Idk.
            w_2 = math.log2(p_2)

        subject_entropoy_dict[subject] = -(p_1 * w_1 + p_2 * w_2)

    return subject_entropoy_dict


"""

"""
def question_difficulty_occurrence(sparse_matrix, reverse=False):
    # 1. Use <sparse_matrix.toarray()>.
    # 2. Python sort dict key by dict value. Note "Less occurrence" = "More difficult for ML".
    matrix_question_based = sparse_matrix.toarray().transpose()
    return ... # TODO: How does Mark's sorting work on matrix exactly?



"""

"""
def question_difficulty_correctness_entropy(sparse_matrix, reverse=False):
    # Same as <question_difficulty_occurrence>. Note "Higher entropy" = "More difficult for ML".








# TODO: If have time, can merge two measures together, and weight entropy with number of occurrence. Although this needs careful treatment like some normalization.
"""

"""
def question_difficulty_CEO(): # Correctness Entropy + Occurrnece.
    pass


