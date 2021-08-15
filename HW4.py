
import os
import json

import word2vec     # v0.9.4

"""
1. 将opencc文件夹复制到D盘根目录
2. 注意, 在类的__init__中, 正确设置wiki_zh_2019文件夹的路径
"""


class HW4(object):
    def __init__(self, corpus_path: object = 'D:\我的資料\wiki_zh_2019') -> object:
        self.corpus_path = corpus_path

        self.corpus_raw = 'hw4_raw.txt'
        self.corpus_tw = 'hw4_tw.txt'
        self.corpus_cut = 'hw4_cut.txt'

        self.model_file = 'hw4_model.bin'
        self.clusters_file = 'hw4-clusters.txt'

        self.stop_words = ['\n', '\t', '\r', '，', '；',  '、', '。']

    def prepare_corpus(self):
        # 加载合并语料库
        self.load_corpus()

        # 简繁转化
        os.system('openCC.bat')

        # 分词
        self.cut()

    def load_corpus(self):
        idx=0
        with open(self.corpus_raw, 'w', encoding='utf-8') as fw:
            for mdir, _, files in os.walk(self.corpus_path):
                print(mdir)
                for file in files:
                    file_path_name = os.path.join(mdir, file)
                    print(file_path_name)
                    with open(file_path_name, 'r', encoding='utf-8') as fr:
                        lines = fr.readlines()
                        for line in lines:
                            line_text = json.loads(line)['text']
                            fw.write(line_text)
                            idx += 1
                            print(idx)

    def cut(self):
        import jieba.analyse

        print('jieba cutting.....')

        jieba.setLogLevel(jieba.logging.INFO)   # 引入jieba模块后加入该代码，代码即可不报警
        jieba.set_dictionary('dict.txt')
        idx=0
        with open(self.corpus_cut, 'w', encoding='utf-8') as fw:
            with open(self.corpus_tw, 'r', encoding='utf-8') as fr:
                lines_rd = fr.readlines()
                for line_rd in lines_rd:
                    terms_list = jieba.lcut(line_rd, cut_all=False, HMM=True)
                    #terms_list = [term for term in terms_list if term not in self.stop_words]
                    line_wr = ' '.join(terms_list) + ' '
                    fw.write(line_wr)

                    idx += 1
                    if idx % 1000 == 0:
                        print(len(lines_rd)-idx)

                """
                while True:
                    line_rd = fr.readline()
                    if not line_rd:
                        break
                    terms_list = jieba.lcut(line_rd, cut_all=False, HMM=True)
                    terms_list = [term for term in terms_list if term not in self.stop_words]
                    line_wr = ' '.join(terms_list) + ' '
                    fw.write(line_wr)
                """

    def train(self):
        word2vec.word2vec(self.corpus_cut, self.model_file, size=100, verbose=True, window=5)
        word2vec.word2clusters(self.corpus_cut, self.clusters_file, 100, verbose=True)

    def predict(self, term='數學'):
        print('predict:')
        model = word2vec.load(self.model_file)
        print(model.vocab)              # 词汇表
        print(model.vectors.shape)
        print(model.vectors)            # 词向量
        print(model[term].shape)
        print(model[term])              # term的词向量

    def test(self, term='數學'):
        print('test:')
        model = word2vec.load(self.model_file)
        indexes, metrics = model.cosine(term, n=20)     # 輸出前二十個相關詞之索引及相关度
        print(indexes, '\n', metrics)
        print(model.vocab[indexes])
        print(model.generate_response(indexes, metrics).tolist())


if __name__ == '__main__':
    hw4 = HW4()
    #hw4.prepare_corpus()
    #hw4.train()
    hw4.predict(term='數學')
    hw4.test(term='知')
