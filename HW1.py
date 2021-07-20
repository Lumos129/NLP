
import numpy as np
import pandas as pd
import jieba.analyse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class HW1(object):
    def __init__(self):
        dataset_txt = pd.read_csv('hw1-dataset.txt',
        #dataset_txt = pd.read_csv('https://raw.githubusercontent.com/cjwu/cjwu.github.io/master/courses/nlp/hw1-dataset.txt',
                                  encoding='utf-8',
                                  header=None, sep='\r\n',
                                  engine='python',
                                  error_bad_lines=False)
        print(dataset_txt)
        self.dataset_values = dataset_txt.values.tolist()
        #print(self.dataset_values[0:2])

        #self.dataset_values = self.dataset_values[0:20]

        self.stop_word = [' ', ',', '.', '?', '\t', '..'] + [str(d) for d in range(10)]

    def parse(self):
        for idx in range(len(self.dataset_values)):
            string_line = self.dataset_values[idx][0]
            #print(string_line)
            #item_list = jieba.analyse.extract_tags(string_line)
            item_list = jieba.lcut(string_line, cut_all=False, HMM=True)
            #print(item_list)

            self.dataset_values[idx] = ' '.join(item_list)
            #print(self.dataset_values[idx])

        print(self.dataset_values[-10:])

    def cut(self):
        jieba.set_dictionary('dict.txt.big')
        for idx in range(len(self.dataset_values)):
            string_line = self.dataset_values[idx][0]
            items_list = jieba.lcut(string_line, cut_all=False, HMM=True)
            #print(items_list)

            #self.dataset_values[idx] = [item for item in items_list if item not in self.stop_word]
            self.dataset_values[idx] = [item for item in items_list if len(item) > 1]
            #print(self.dataset_values[idx])

        print(self.dataset_values[0:10])

    def cut2(self):
        self.dataset_cut = []
        for line in self.dataset_values[0:10]:
            string_line = line[0]
            items_list = jieba.lcut(string_line, cut_all=False, HMM=True)

            self.dataset_cut.append([item for item in items_list if item not in self.stop_word])

        print(self.dataset_cut[0:10])

    def tf(self):
        # 构建词汇表
        self.vocabulary = {}
        for paper in self.dataset_values:
            for term in paper:
                if term in self.vocabulary:
                    self.vocabulary[term] += 1
                else:
                    self.vocabulary[term] = 1

        print(self.vocabulary.items())

        # 排序
        self.top100 = sorted(self.vocabulary.items(), key=lambda k: (k[1]), reverse=True)[0:100]
        print(self.top100)


if __name__ == '__main__':
    hw = HW1()
    hw.cut()
    hw.tf()


    exit(0)




mydata_txt = pd.read_csv('https://raw.githubusercontent.com/cjwu/cjwu.github.io/master/courses/nlp/hw1-dataset.txt',
                         encoding='utf-8',
                         error_bad_lines=False)

#mydata_txt = pd.read_csv('hw1-dataset.txt', encoding='utf-8', header=None)
print(mydata_txt)

#with open('https://raw.githubusercontent.com/cjwu/cjwu.github.io/master/courses/nlp/hw1-dataset.txt', 'rt', ) as f:
#    print(f.readlines())
