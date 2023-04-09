#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:19:42 2022

@author: karanoberoi
"""

import numpy as np
import pandas as pd
import random


def train_test_split(df, test_perc):
    
    if isinstance(test_perc, float):
        test_perc = round(test_perc * len(df))

    loc = df.index.tolist()
    test_index = random.sample(population=loc, k=test_perc)

    df_test = df.loc[test_index]
    df_train = df.drop(test_index)
    
    return df_train, df_test


def check_purity(data):
    
    if len(np.unique(data[:, -1])) == 1:
        return True
    else:
        return False
    

def classify_data(data):
    
    unique_classes, unique_classes_counts = np.unique(data[:, -1], return_counts = True)
    
    max_class = unique_classes[unique_classes_counts.argmax()]
    
    return max_class


def get_potential_splits(data):
    
    potentialSplits = {}
    _, n_col = data.shape
    col_idx = list(range(n_col - 1)) # excluding the last column which is the label
        
    for col in col_idx:
        values = data[:, col]
        unq = np.unique(values)
    
        type_of_feature = FEATURE_TYPES[col]
        if type_of_feature == "continuous":    
            if len(unq) == 1:
                potentialSplits[col] = unq
            else:
                potentialSplits[col] = []
                for i in range(len(unq)):
                    if i != 0:
                        val_2 = unq[i]
                        val_1 = unq[i - 1]
                        potentialSplits[col].append((val_2 + val_1) / 2)
        
        elif len(unq) >= 1:
            potentialSplits[col] = unq
                    
    return potentialSplits


def split_data(data, split_col, split_val):
    
    split_col_val = data[:, split_col]

    type_of_feature = FEATURE_TYPES[split_col]
    if type_of_feature == "continuous":
        data_below = data[split_col_val <= split_val]
        data_above = data[split_col_val >  split_val]
     
    else:
        data_below = data[split_col_val == split_val]
        data_above = data[split_col_val != split_val]
        
    return data_below, data_above


def entropy(data):
    
    unique_classes, unique_classes_counts = np.unique(data[:, -1], return_counts = True)
    
    p = unique_classes_counts / unique_classes_counts.sum()
    
    return sum(p * -np.log2(p))


def overall_entropy(data_left, data_right):
    
    prob_left = len(data_left) / (len(data_left) + len(data_right))
    prob_right = len(data_right) / (len(data_left) + len(data_right))
    
    entropy_val = prob_left * entropy(data_left) + prob_right * entropy(data_right)

    return entropy_val


def calculate_best_split(data, potentialSplits):
    
    inf_entropy = 9999
    
    for column_index in potentialSplits:
        for val in potentialSplits[column_index]:
            data_left, data_right = split_data(data, split_col=column_index, split_val=val)
            calculated_entropy = overall_entropy(data_left, data_right)

            if calculated_entropy <= inf_entropy:
                inf_entropy = calculated_entropy
                best_split_col = column_index
                best_split_val = val
    
    return best_split_col, best_split_val


def determine_categorical(df):
    
    feature_type = []
    n_unique_values_treshold = 15
    for col in df.columns:
        if col != "label":
            unique_values = df[col].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_type.append("categorical")
            else:
                feature_type.append("continuous")
    
    return feature_type


def decision_tree_algo(df, counter=0,min_size=2,max_depth=5):
    
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_categorical(df)
        data = df.values
    else:
        data = df           
    
    if check_purity(data) or len(data) < min_size or counter == max_depth:
        return classify_data(data)

    # classification creation
    else:    
        counter += 1
 
        potentialSplits = get_potential_splits(data)        
        split_col, split_val = calculate_best_split(data, potentialSplits)
        data_below, data_above = split_data(data, split_col, split_val)
        
        if len(data_below) == 0 or len(data_above) == 0:
            return classify_data(data)
        
        else:
            # instantiate sub-tree
            feature_name = COLUMN_HEADERS[split_col]
            type_of_feature = FEATURE_TYPES[split_col]
            if type_of_feature == "continuous":
                question = "{} <= {}".format(feature_name, split_val)
                
            else:
                question = "{} = {}".format(feature_name, split_val)
                
            decision_tree = {question: []}
            
            # find answers (recursion)
            yes_answer = decision_tree_algo(data_below, counter, min_size,max_depth)
            no_answer = decision_tree_algo(data_above, counter, min_size,max_depth)
        
            if yes_answer == no_answer:
                decision_tree = yes_answer
            else:                
                decision_tree[question].append(yes_answer)
                decision_tree[question].append(no_answer)
        
            return decision_tree


def classify(sample,Tree):
    
    if not isinstance(Tree, dict):
        return Tree
    
    question = list(Tree.keys())[0]
    attr, val = question.split(" <= ")
    
    if sample[attr] <= float(val):
        answer = Tree[question][0]
    
    else:
        answer = Tree[question][1]
    
    return classify(sample, answer)


def tree_prediction(df, Tree):
    prediction = df.apply(classify, axis = 1, args = (Tree,))
    return prediction


def accuracy(prediction, category):
    acc = prediction == category
    return round(acc.mean(),2)
