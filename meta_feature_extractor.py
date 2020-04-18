import math
import statistics
from os import listdir, getcwd
from os.path import isfile, join
import pandas as pd
import json
import final_project as fp


def get_highest_voted_ans(dic_prob):
    max = 0
    highest_voted_ans = ''
    for ans in dic_prob:
        if dic_prob[ans] > max:
            max = dic_prob[ans]
            highest_voted_ans = ans
    return dic_prob[highest_voted_ans]

def get_consensus(dic_prob):
    consensus = 0
    for p in dic_prob:
        if dic_prob[p] == 0:
            continue
        consensus -= dic_prob[p] * math.log2(dic_prob[p])
    return consensus / math.log2(len(dic_prob))


def get_answer_distribution(all_possible_ans, df):
    dic = {}
    # initilize with 0
    for ans in all_possible_ans:
        dic[str(ans)] = 0

    # count answers
    for answer in df['Answer']:
        # print(answer)
        dic[str(answer)] += 1
    return dic

def get_init_features(df,num_of_answers, first_idx):
    features = {}
    all_possible_ans = df.columns[first_idx : first_idx + num_of_answers]
    dic = get_answer_distribution(all_possible_ans, df)

    # get probability
    num_of_subjects = len(df['Answer'])
    dic_prob = {}
    for ans in dic:
        dic_prob[ans] = dic[ans]/num_of_subjects

    features['consensus'] = get_consensus(dic_prob) #divided by log(n) 0: full consensus 1: no consensus
    features['highest_voted_ans'] = get_highest_voted_ans(dic_prob)
    features['variance'] = statistics.variance(list(dic.values()))
    return features


if __name__ == '__main__':

    # Load meta sdata file
    with open('/Users/yanivleedon/Desktop/final_project_clusters/meta_data/problems_meta_data.txt') as json_file:
        problem_dic = json.load(json_file)
        ans = {}
        # for each problem, extract features
        for problem in problem_dic:
            # needed meta data from file
            file_name = problem_dic[problem]['file_name']
            num_of_answers = problem_dic[problem]['num_of_answers']
            first_idx = problem_dic[problem]['first_idx']

            # read problom
            df = pd.read_csv(file_name, index_col='Worker ID')

            # get initial features
            ans[problem] = get_init_features(df, num_of_answers, first_idx)
        print(ans)
