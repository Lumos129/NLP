
import numpy as np
import pandas as pd
import jieba.analyse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from matplotlib import pyplot as plt


class HW1(object):
    def __init__(self):
        #dataset_txt = pd.read_csv('hw1-dataset.txt',
        dataset_txt = pd.read_csv('https://raw.githubusercontent.com/cjwu/cjwu.github.io/master/courses/nlp/hw1-dataset.txt',
                                  encoding='utf-8',
                                  header=None, sep='\r\t',
                                  engine='python',
                                  error_bad_lines=False)
        print(dataset_txt)
        self.dataset_values = dataset_txt.values.tolist()
        #print(self.dataset_values[0:2])

        #self.dataset_values = self.read_txt('hw1-dataset.txt')

        self.stop_words = [' ', ',', '.', '?', '\t', '..', '？', '!', '∷', '？'] + [str(d) for d in range(10)]

    def read_txt(self, file_name):
        with open(file_name, 'rt', encoding='utf-8') as f:
            lines = f.readlines()

        lines = [[line] for line in lines]
        print(lines)
        return lines

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
        jieba.set_dictionary('dict.txt')
        for idx in range(len(self.dataset_values)):
            string_line = self.dataset_values[idx][0]
            items_list = jieba.lcut(string_line, cut_all=False, HMM=True)
            #print(items_list)

            self.dataset_values[idx] = [item for item in items_list if item not in self.stop_words]
            #print(self.dataset_values[idx])

        print(self.dataset_values[0:10])

    def tf(self):
        """
        Term Frequency
        :return:
        """
        # 詞彙表
        self.vocabulary = {}
        for paper in self.dataset_values:
            for term in paper:
                if term in self.vocabulary:
                    self.vocabulary[term] += 1
                else:
                    self.vocabulary[term] = 1

        #print(self.vocabulary.items())

        # Sort
        self.top100 = sorted(self.vocabulary.items(), key=lambda k: (k[1]), reverse=True)[0:100]
        print(self.top100)
        #k, v = zip(*self.top100)
        #print(k, v)

    def tf2(self):
        """
        Term Frequency
        :return:
        """
        # 詞彙表
        self.vocabulary = {}
        self.tf = {}
        term_idx = 0
        for paper in self.dataset_values:
            terms_paper = {}
            for term in paper:
                if term not in self.vocabulary:
                    self.vocabulary[term] = term_idx
                    term_idx += 1

                if term in terms_paper:
                    terms_paper[term] += 1
                else:
                    terms_paper[term] = 1

            for key in terms_paper:
                if key in self.tf:
                    self.tf[key].append(terms_paper[key])
                else:
                    self.tf[key] = [terms_paper[key]]

        # self.tf的value由list转为np.array
        for key in self.tf:
            self.tf[key] = np.array(self.tf[key])

        # 詞頻
        freq = {key: np.sum(self.tf[key]) for key in self.tf}

        # Sort
        self.freq_sort = sorted(freq.items(), key=lambda k: (k[1]), reverse=True)
        self.freq_top100 = self.freq_sort[0:100]
        print(self.freq_top100)
        #k, v = zip(*self.freq_top100)
        #print(k, v)

    def tf3(self):
        """
        Term Frequency
        :return:
        """
        # 詞彙表
        self.vocabulary = {}
        self.tf = {}
        term_idx = 0
        for paper in self.dataset_values:
            terms_paper = {}
            lenn = len(paper)
            #print(lenn)
            for term in paper:
                if term not in self.vocabulary:
                    self.vocabulary[term] = term_idx
                    term_idx += 1

                if term in terms_paper:
                    terms_paper[term] += 1
                else:
                    terms_paper[term] = 1

            for key in terms_paper:
                if key in self.tf:
                    self.tf[key].append(terms_paper[key]/lenn)
                else:
                    self.tf[key] = [terms_paper[key]/lenn]

        # self.tf的value由list转为np.array
        for key in self.tf:
            self.tf[key] = np.array(self.tf[key])

        # 詞頻
        freq = {key: np.max(self.tf[key]) for key in self.tf}
        #print(freq)

        # Sort
        self.freq_sort = sorted(freq.items(), key=lambda k: (k[1]), reverse=True)
        self.freq_top100 = self.freq_sort[0:100]

        print(self.freq_top100)
        #k, v = zip(*self.freq_top100)
        #print(k, v)

    def tf_idf(self):
        n = len(self.dataset_values)
        
        df = {key: value.shape[0] for key, value in self.tf.items()}
        idf = {key: np.log10(n/value) for key, value in df.items()}

        self.tf_idf = {key: self.tf[key]*idf[key] for key in self.tf}
        tf_idf_max = {key: np.max(value) for key, value in self.tf_idf.items()}

        # Sort
        tf_idf_sort = sorted(tf_idf_max.items(), key=lambda k: (-k[1]))
        self.tf_idf_top100 = tf_idf_sort[0:100]
        print(self.tf_idf_top100)

    def plot(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']   
        plt.rcParams['axes.unicode_minus'] = False      

        plt.figure()
        plt.title('Frequency')

        terms, freq = zip(*self.freq_top100)
        terms_id = [self.vocabulary[term] for term in terms]

        plt.plot(range(100), freq)

        plt.xticks(range(100), terms_id)
        plt.xticks(rotation=90)

        plt.xlabel('編號')
        plt.ylabel("Frequency")

        # TF-DIF
        plt.figure()
        plt.title('TF-IDF')

        terms, tf_idf = zip(*self.tf_idf_top100)
        terms_id = [self.vocabulary[term] for term in terms]

        plt.plot(range(100), tf_idf)

        plt.xticks(range(100), terms_id)
        plt.xticks(rotation=90)

        plt.xlabel('編號')
        plt.ylabel("TF-IDF")


if __name__ == '__main__':
    hw = HW1()
    hw.cut()
	#hw.tf2()
    hw.tf3()
    hw.tf_idf()

    hw.plot()
    plt.show()
    exit(0)