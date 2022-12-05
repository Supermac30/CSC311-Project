"""
Question-based difficulty measure. Rank questions' difficulty (to our ML algo, not how "hard" a question is conceptually) depending on its subjects.

Will rank questions based on the "most difficult" subject it belongs to. Because it's hard to get questions correct since you need to understand everything. Thus it makes intuitive sense to foucs on the "most difficult subject" a question involves.

When ranking difficulty-level of subject, first 

"Difficult" subject here refers to subject that is hard for ML.
"""

import numpy as np
from utils import *
import math
import csv
import os
#from difficulty_measurers import _sort_data


def _sort_data_question_based(data, function, reverse):
    question_array = [(x, i) for i, x in enumerate(data["question_id"])]
    question_array = sorted(question_array, key=lambda item: function(item[0]), reverse=reverse) # Better than Mark's <reverse=not reverse>!
    sorted_users = [data["user_id"][i] for (_, i) in question_array]
    sorted_questions = [data["question_id"][i] for (_, i) in question_array]
    sorted_is_correct = [data["is_correct"][i] for (_, i) in question_array]
 
    return {"user_id": sorted_users, "question_id": sorted_questions, "is_correct": sorted_is_correct}



def string_list_parser(str_list):
    """
    Given a string that represents a list e.g. "[1, 23, 456]", return the actual list it represents.
    """
    new_str_list = str_list[1:len(str_list)-1]
    new_str_list = new_str_list.split(", ")
    return new_str_list
    


def find_subjects():
    """
    Return a list of <subject_id> as in "question_meta.csv".
    """ 
    # Work with "question_meta.csv" then put into dictionary form: dict[question_id] = a list of subject_id.
    question_meta_path = "./data/question_meta.csv"
    question_meta_dict = {}
    with open(question_meta_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                subject_list = []
                converted_list = string_list_parser(row[1])
                for item in converted_list:
                    try: 
                        temp = int(item);
                        subject_list.append(temp)
                    except ValueError: # To filter out non-integer ones.
                        pass
                question_meta_dict[int(row[0])] = subject_list 
            except ValueError: # Pass first row.
                pass
    return question_meta_dict



def subject_occurrence():
    """
    Rank subjects by the number of occurrence in all questions. "Less occurrence" = "More difficult for ML".
    
    Return dictionary: dict[subject_id] = occurrence.
    """
    # 1. Work with "question_meta.csv". Check out <load_csv>.
    # 2. Transform to all elements into an array of id, not counting '0' as it occurs everywhere.
    # 3. Use Python list method to count occurrnece.
    
    # TODO: This function is part of <subject_correctness_entropy>.
    



def subject_correctness_entropy():
    """
    Rank subjects by the correctness rate. Perhaps use entropy. "Higher entropy" = "More difficult for ML".
    
    Return a dictionary: dict[subject_id] = entropy.
    """

    # 1. Work with "question_meta.csv" then put into dictionary form: dict[question_id] = a list of subject_id.
    question_meta_dict = find_subjects()


    # 2. Work with "train_data.csv". Or maybe use "train_sparse_matrix" then use <load_npz>.
    sparse_matrix = load_train_sparse("./data").toarray().transpose() # Question-base matrix!


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
    
    subject_entropy_dict = {}

    for subject in subject_correctness_rate_dict:
        # subject_correctness_rate_dict[subject][3] = subject_correctness_rate_dict[subject][1] / subject_correctness_rate_dict[subject][2]
        num_correct = subject_correctness_rate_dict[subject][1] 
        num_all = subject_correctness_rate_dict[subject][2] 
        
        if ((num_correct == 0) and (num_all == 0)):
            correctness_rate = 0
        else:
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

        subject_entropy_dict[subject] = -(p_1 * w_1 + p_2 * w_2)

    return subject_entropy_dict



#def question_difficulty_occurrence(sparse_matrix, reverse=False):
    # 1. Use <sparse_matrix.toarray()>.
    # 2. Python sort dict key by dict value. Note "Less occurrence" = "More difficult for ML".
    
    #matrix_question_based = sparse_matrix.toarray().transpose()
    

def question_difficulty_correctness_entropy(data, reverse=False):
    # Same as <question_difficulty_occurrence>. Note "Higher entropy" = "More difficult for ML".
    # <data> is of form: <{"user_id": [user_id's], "question_id": [question_id's], "is_correct": [0's and 1's]}>

    subject_entropy_dict = subject_correctness_entropy()
    question_subjects_dict = find_subjects()


    def find_max_entropy(question_id):
        """
        Given <question_id>, return the entropy of the "most difficult" subject (i.e. higest entropy) this question belongs to from <subjbect_entropy_dict>.
        """

        # DEBUG:
        #print(question_id) 

        subjects = question_subjects_dict[question_id]
        return max([subject_entropy_dict[subject] for subject in subjects])


    #matrix_question_based = [list(row) for row in sparse_matrix.toarray().transpose()]
    #return sorted(matrix_question_based, key=lambda question_row: find_entropy(matrix_question_based.index(question_row)), reverse=reverse)  # Python <sorted()> function is default to be ascending order.

    return _sort_data_question_based(data, lambda question_id: find_max_entropy(question_id), reverse)



    # DID NOT Use dictionary form instead because <np.where> relies on assumption that every row is unique. Actually why not just check whether there is repeating elements. Tried in main section and ensured the question based matrix does not have duplicate!!! 

# TODO: If have time, can merge two measures together, and weight entropy with number of occurrence. Although this needs careful treatment like some normalization.
#def question_difficulty_CEO(): # Correctness Entropy + Occurrnece.
#    pass


if __name__ == "__main__":

    # Running main function for debugging purpose.

    train_data = load_train_csv("./data")
    
    # Ensure that matrix does not have duplicate.
    #matrix_copy = load_train_sparse("./data/").toarray().transpose()
    #matrix_copy = [list(row) for row in matrix_copy]
    #print(matrix_copy)
    
    #for row in matrix_copy:
    #    assert matrix_copy.count(row) == 1

    print(question_difficulty_correctness_entropy(train_data))
