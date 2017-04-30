#!/usr/bin/env python
# coding=utf-8

##this python file can use Information gain, mutal information and Weighted Log Likelihood Ration for feature selection


import os
import sys
import codecs
import jieba

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import feature_selection


def getChList(docStrByte):
    inputStr = str(docStrByte, encoding = 'gbk', errors = 'ignore')
    ## filter the first several sentence
    strList = list(i for i in inputStr.split('\n'))
    # print(strList)
    startLine = 0
    for i in range(len(strList)):
        if (strList[i].startswith('【')):
            startLine += 1
        else:
            break
    # print(strList[startLine:])

    rawTokens = list(jieba.tokenize(''.join(strList[startLine:])))
    # stopWord = {}.fromkeys([line for line in open('stopwords.txt','r',encoding = 'gbk', errors = 'ignore')])
    fSW = open('stopwords.txt', 'r', encoding = 'utf-8', errors = 'ignore').read()
    # print(fSW.split('\n')[:99])
    stopWord = {}.fromkeys(fSW.split('\n'))

    # print(stopWord)
    stopWord[''] = None
    # for (k,v) in stopWord.items():
    #     print(k, ',', v)
    final = ''
    for seg in rawTokens:
        # print(seg)
        # seg.encode('gbk')
        word = seg[0].strip()
        if  word not in stopWord:
            final += (' ' + word)#if using final is not good
        # else:
            # print(seg)
    # print(type(final))
    return final

def getDatas(dataset_dir_name):
    movie_reviews = load_files(dataset_dir_name)
    #movie_reviews具有data和target两个属��?
    doc_str_list_train, doc_str_list_test, doc_class_list_train, doc_class_list_test = train_test_split(movie_reviews.data, movie_reviews.target, test_size = 0.2, random_state = 0)

    vectorizer = CountVectorizer(binary = True, decode_error = u'ignore')
    word_tokenizer = vectorizer.build_tokenizer()

    doc_str_list_train = list(getChList(doc_str) for doc_str in doc_str_list_train)
    doc_str_list_test = list(getChList(doc_str) for doc_str in doc_str_list_test)

    doc_terms_list_train = [word_tokenizer(doc_str) for doc_str in doc_str_list_train]
    return vectorizer, doc_str_list_train, doc_str_list_test,doc_class_list_train, doc_class_list_test, doc_terms_list_train


def text_classifly_twang(vectorizer, doc_str_list_train, doc_str_list_test,doc_class_list_train, doc_class_list_test, doc_terms_list_train,fs_method, fs_num):
    #文件夹下有多个子文件夹，每个文件夹名为类别并存放属于该类的txt
    #fs_method是特征选取的方式
    #fs_num是在进行特征选取后，按照特征重要度排序后得到的前fs_num个特征
    print('Loading dataset, 80% for training, 20% for testing...')

    # print(doc_class_list_train)#一串1 0 组成的列表
    print('Feature selection...')
    print('fs method:' + fs_method, 'fs num:' + str(fs_num))

    # at first i encounter an encoding problem
    #here i fix the problem with the decode_error parameter = u'ignore' while default setting is u'strict'
    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html


    # print((word_tokenizer(('i am new student\nyou good'))))#return a list of word_tokenizer
    # print(list(jieba.cut('我是撒谎比\n你是\t杀吧', cut_all = False )))

    # doc_terms_list_train = [word_tokenizer(str(doc_str, encoding = 'utf-8', errors = 'ignore')) for doc_str in doc_str_list_train]

    selectedFeatures = feature_selection.feature_selection(doc_terms_list_train, doc_class_list_train, fs_method)

    print(len(selectedFeatures))
    term_set_fs = selectedFeatures[:fs_num]
    # print('-----------',len(term_set_fs))

    term_dict = dict(zip(term_set_fs, range(len(term_set_fs))))
    vectorizer.fixed_vocabulary = True
    vectorizer.vocabulary = term_dict
    doc_train_vec = vectorizer.fit_transform(doc_str_list_train)
    #print the number of features
    # print(doc_train_vec.shape)
    # print((doc_train_vec))
    #scipy.sparse.csr.csr_matrix
    # print(type(doc_train_vec))
    doc_test_vec= vectorizer.transform(doc_str_list_test)


    # print('Building Naive Beyas model...')
    # clf = MultinomialNB().fit(doc_train_vec, doc_class_list_train)  #调用MultinomialNB分类��?
    # doc_test_predicted = clf.predict(doc_test_vec)

    print('Building SVM model...')
    svclf = SVC(kernel = 'linear')#default with 'rbf'
    svclf.fit(doc_train_vec, doc_class_list_train)
    doc_test_predicted = svclf.predict(doc_test_vec)

    # print('Building KNN model...')
    # knnclf = KNeighborsClassifier()
    # knnclf.fit(doc_train_vec, doc_class_list_train)
    # doc_test_predicted = knnclf.predict(doc_test_vec)

    #here i can't use RBM, there is not prediction attribute
    # from sklearn.neural_network import BernoulliRBM
    # RBMclf = BernoulliRBM().fit(doc_train_vec, doc_class_list_train)
    # doc_test_predicted = RBMclf.predict(doc_test_vec)


    # print('Building Multilayer perception classifier model...')
    # mlpclf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (50,20), random_state = 1)
    # mlpclf.fit(doc_train_vec, doc_class_list_train)
    # doc_test_predicted = mlpclf.predict(doc_test_vec)




    #print accuracy
    acc = np.mean(doc_test_predicted == doc_class_list_test)
    print('Accuracy:', acc)
    return acc

    #print f1
    f1_micro = metrics.f1_score(doc_class_list_test, doc_test_predicted, average = 'micro')
    f1_macro = metrics.f1_score(doc_class_list_test, doc_test_predicted, average = 'macro')
    #average = macro , 没有考虑不平衡数据集，本次实验数据集中literacture类的数目少，预测结构差
    print('f1_micro:\t', f1_micro, '\tf1_macro:\t', f1_macro)
    print(metrics.f1_score(doc_class_list_test, doc_test_predicted, average = None))

    print(metrics.classification_report(doc_class_list_test, doc_test_predicted, target_names = ['sport','economy','computer']))
    print(metrics.confusion_matrix(doc_class_list_test, doc_test_predicted))
    return f1_micro, acc

if __name__ == '__main__':

    dataset_dir_name = sys.argv[1]
    fs_method_list = ['IG', 'MI', 'WLLR']
    # fs_num_list = [100,200,400,800,1000,2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,20000,40000]
    fs_num_list = [100,200]
    acc_dict = {}
    vectorizer, doc_str_list_train, doc_str_list_test,doc_class_list_train, doc_class_list_test, doc_terms_list_train = getDatas(dataset_dir_name)

    for fs_method in fs_method_list:
        acc_list = []
        f1_list = []
        for fs_num in fs_num_list:
            f1,acc = text_classifly_twang(vectorizer, doc_str_list_train, doc_str_list_test,doc_class_list_train, doc_class_list_test, doc_terms_list_train, fs_method, fs_num)
            acc_list.append(acc)
        acc_dict[fs_method] = acc_list
        print('fs method:', acc_dict[fs_method])

    for fs_method in fs_method_list:
        plt.plot(fs_num_list, acc_dict[fs_method],  '--^',  label = fs_method)
        plt.title('feature  selection')
        plt.xlabel('fs num')
        plt.ylabel('accuracy')
        plt.ylim((0.5, 1))

    plt.legend( loc='upper right', numpoints = 1)
    plt.show()
