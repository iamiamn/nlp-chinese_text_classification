#!/usr/bin/env python
# coding=utf-8

##this python file can use Information gain, mutal information and Weighted Log Likelihood Ration for feature selection


import os
import sys
import jieba

import numpy as np

import matplotlib.pyplot as plt
import nltk.stem
s=nltk.stem.SnowballStemmer('english')
s.stem('imaging')


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
    ## 传入一个文档的二进制代码，返回中文分词后的结果，用空格把中文分词的词

    inputStr = str(docStrByte, encoding = 'gbk', errors = 'ignore').lower()#二进制转为字符串，英文字母转为小写
    strList = ''.join(inputStr.split('\n'))#删去换行符，连接每行成为一个段落
    rawTokens = list(jieba.tokenize(strList))#中文分词

    #stopWord 是 一个字典，每个key 是一个停用词，value都是None
    fSW = open('stopwords.txt', 'r', encoding = 'utf-8', errors = 'ignore').read()
    stopWord = {}.fromkeys(fSW.split('\n'))
    stopWord[''] = None

    final = []
    s = nltk.stem.SnowballStemmer('english')
    for seg in rawTokens:
        # print(seg[0].strip())
        rawWord = seg[0].strip()#strip()函数，去除字符串前后的空格
        if (rawWord.isalpha()):#如果是英文单词，则提取词干
            word = s.stem(rawWord)
        else:
            word = rawWord

        if  word not in stopWord:#去除停用词
            final.append(word)#最后返回list
    return final

def getDatas(dataset_dir_name):
    movie_reviews = load_files(dataset_dir_name)

    doc_str_list_train, doc_str_list_test, doc_class_list_train, doc_class_list_test = train_test_split(movie_reviews.data, movie_reviews.target, test_size = 0.2, random_state = 0)

    #word_tokenizer 应用于英文文档时，按照空格将词分开，每篇文档转变为一个词向量，用于构建词频矩阵。所以应用于中文分词时
    vectorizer = CountVectorizer(binary = True, decode_error = u'ignore')
    word_tokenizer = vectorizer.build_tokenizer()


    #每个文档是一个词list
    doc_terms_list_train = list(getChList(doc_str) for doc_str in doc_str_list_train)
    doc_terms_list_test = list(getChList(doc_str) for doc_str in doc_str_list_test)


    return vectorizer, doc_str_list_train, doc_str_list_test,doc_class_list_train, doc_class_list_test, doc_terms_list_train


def text_classifly_twang(vectorizer, doc_str_list_train, doc_str_list_test,doc_class_list_train, doc_class_list_test, doc_terms_list_train,fs_method, fs_num, cf_method):
    print('Loading dataset, 80% for training, 20% for testing...')
    print('Feature selection...')
    print('fs method:' + fs_method, 'fs num:' + str(fs_num))

    selectedFeatures = feature_selection.feature_selection(doc_terms_list_train, doc_class_list_train, fs_method)
    print('-------select feature_selection')
    numShow = 500
    count = 0

    # 打印特证词
    # for i in selectedFeatures[1:numShow]:
    #     print(count, ' \t',i)
    #     count += 1

    print('特征词的数量:')
    print(len(selectedFeatures))
    term_set_fs = selectedFeatures[:fs_num]#选择前fs_num个特征词
    term_dict = dict(zip(term_set_fs, range(len(term_set_fs))))#建立单词字典，key是特证词，value是索引号

    #词频矩阵
    vectorizer.fixed_vocabulary = True#固定单词库，这样词频矩阵只会统计单词库内的单词出现的频次
    vectorizer.vocabulary = term_dict
    doc_train_vec = vectorizer.fit_transform(doc_str_list_train)
    doc_test_vec= vectorizer.transform(doc_str_list_test)

    if(cf_method == 'nb'):
        print('Building Naive Beyas model...')
        clf = MultinomialNB().fit(doc_train_vec, doc_class_list_train)  #调用MultinomialNB分类��?
        doc_test_predicted = clf.predict(doc_test_vec)
    elif(cf_method == 'svm'):
        print('Building SVM model...')
        svclf = SVC(kernel = 'linear')#default with 'rbf'
        svclf.fit(doc_train_vec, doc_class_list_train)
        doc_test_predicted = svclf.predict(doc_test_vec)
    elif(cf_method == 'knn'):
        print('Building KNN model...')
        knnclf = KNeighborsClassifier(5)#括号内传入k值
        knnclf.fit(doc_train_vec, doc_class_list_train)
        doc_test_predicted = knnclf.predict(doc_test_vec)
    elif(cf_method == 'bp'):
        print('Building Multilayer perception classifier model...')
        mlpclf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (50,20), random_state = 1)
        mlpclf.fit(doc_train_vec, doc_class_list_train)
        doc_test_predicted = mlpclf.predict(doc_test_vec)

    #here i can't use RBM, there is not prediction attribute
    # from sklearn.neural_network import BernoulliRBM
    # RBMclf = BernoulliRBM().fit(doc_train_vec, doc_class_list_train)
    # doc_test_predicted = RBMclf.predict(doc_test_vec)

    #打印准确度
    acc = np.mean(doc_test_predicted == doc_class_list_test)
    printOption = True
    if (printOption):
        print('Accuracy:', acc)

    f1_micro = metrics.f1_score(doc_class_list_test, doc_test_predicted, average = 'micro')
    f1_macro = metrics.f1_score(doc_class_list_test, doc_test_predicted, average = 'macro')
    #average = macro , 没有考虑不平衡数据集。本次实验使用的三个类样本数目相近，因此f1_macro直接计算分类器在三个类别的f1值的均值，结果可以接受
    if (printOption):
        print('f1_micro:\t', f1_micro, '\tf1_macro:\t', f1_macro)
        print(metrics.f1_score(doc_class_list_test, doc_test_predicted, average = None))
    #只选取了三个类的文章
        print(metrics.classification_report(doc_class_list_test, doc_test_predicted, target_names = ['sport','economy','computer']))
        print(metrics.confusion_matrix(doc_class_list_test, doc_test_predicted))
    else:
        f1_macro = 0
        acc = 0
    return f1_macro, acc

def printing(cf_method, fs_num_list, fs_method_list,acc_dict, f1_dict):
    #画图
    for fs_method in fs_method_list:
        plt.plot(fs_num_list, acc_dict[fs_method],  '--^',  label = fs_method)
        plt.title(cf_method + 'accuracy --- feature  selection')
        plt.xlabel('fs num')
        plt.ylabel('accuracy')
        plt.ylim((0.5, 1))

    plt.legend( loc='upper right', numpoints = 1)
    plt.show()

    for fs_method in fs_method_list:
        plt.plot(fs_num_list, f1_dict[fs_method],  '--^',  label = fs_method)
        plt.title(cf_method + 'f1 score --- feature  selection')
        plt.xlabel('fs num')
        plt.ylabel('f1')
        plt.ylim((0.85, 1))

    plt.legend( loc='upper right', numpoints = 1)
    plt.show()

if __name__ == '__main__':

    dataset_dir_name = sys.argv[1]
    fs_method_list = ['IG', 'MI', 'WLLR']
    fs_num_list = [100,200,300,500,750,1000,2000, 3000, 4000, 5000, 10000,20000]
    cf_method_list = ['nb','svm','knn','bp']
    # fs_num_list = [100]

    cf_acc_dict = {}
    cf_f1_dict = {}
    vectorizer, doc_str_list_train, doc_str_list_test,doc_class_list_train, doc_class_list_test, doc_terms_list_train = getDatas(dataset_dir_name)

    for cf_method in cf_method_list:
        fOutAcc = open(cf_method + 'acc_list.csv', 'w')
        fOutF1 = open(cf_method + 'f1_list.csv', 'w')
        fOutF1.write('fs_method,' + ','.join(str(i) for i in fs_num_list) +'\n')
        fOutAcc.write('fs_method,' +','.join(str(i) for i in fs_num_list) + '\n')
        acc_dict = {}
        f1_dict = {}
        for fs_method in fs_method_list:
            acc_list = []
            f1_list = []
            for fs_num in fs_num_list:
                f1,acc = text_classifly_twang(vectorizer, doc_str_list_train, doc_str_list_test,doc_class_list_train, doc_class_list_test, doc_terms_list_train, fs_method, fs_num, cf_method)
                acc_list.append(acc)
                f1_list.append(f1)
            acc_dict[fs_method] = acc_list
            f1_dict[fs_method] = f1_list
            print('fs method:', acc_dict[fs_method])
            fOutAcc.write(fs_method+ ',' + ','.join(str(i) for i in acc_dict[fs_method]) + '\n')
            fOutF1.write(fs_method + ',' + ','.join(str(i) for i in f1_dict[fs_method]) + '\n')
        fOutF1.close()
        fOutAcc.close()
        cf_acc_dict[cf_method] = acc_dict
        cf_f1_dict[cf_method] = f1_dict
    for cf_method in cf_method_list:
        printing(cf_method, fs_num_list, fs_method_list, cf_acc_dict[cf_method],cf_f1_dict[cf_method])
