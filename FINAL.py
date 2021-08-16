
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import joblib

"""
WSDM - Fake News Classification
xxxxxx https://www.kaggle.com/wsdmcup/wsdm-fake-news-classification?select=train.csv
https://www.kaggle.com/c/fake-news-pair-classification-challenge/data?select=test.csv
"""


class FakeNews(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

        self.train_csv = './WSDM_FakeNews/train.csv'
        self.test_csv = './WSDM_FakeNews/test.csv'
        self.solution_file = 'gong1.csv'

        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)#, weights='distance')

        self.load_csv(self.train_csv)

        self.vectorizer0 = CountVectorizer(max_features=1000)   # 内存不够. 应为None
        self.tf_idf_transformer0 = TfidfTransformer()

        self.vectorizer1 = CountVectorizer(max_features=1000)
        self.tf_idf_transformer1 = TfidfTransformer()

        self.fit_tf_idf()

    def fit_tf_idf(self):
        self.vectorizer0.fit(self.fake_news)
        self.tf_idf_transformer0.fit(self.vectorizer0.transform(self.fake_news))

        self.vectorizer1.fit(self.true_news)
        self.tf_idf_transformer1.fit(self.vectorizer0.transform(self.true_news))

    def news_to_vector(self, x_f, x_t):
        x0 = self.tf_idf_transformer0.transform(self.vectorizer0.transform(x_f)).toarray().astype('float32')
        x1 = self.tf_idf_transformer1.transform(self.vectorizer1.transform(x_t)).toarray().astype('float32')
        return np.concatenate((x0, x1), axis=1)

    def cut(self, docs):
        import jieba.analyse

        print('jieba cutting.....')

        jieba.setLogLevel(jieba.logging.INFO)   # 引入jieba模块后加入该代码，代码即可不报警
        #jieba.set_dictionary('dict.txt.big')

        docs_list = []
        for story in docs:
            if not isinstance(story, str):
                story = '，'
            terms_list = jieba.lcut(story, cut_all=False, HMM=True)
            docs_list.append(' '.join(terms_list))

        return docs_list

    def knn_train(self):
        print('knn training ...')
        x = self.news_to_vector(self.fake_news, self.true_news)
        self.knn.fit(x, self.y)

        joblib.dump(self.knn, 'HWf3.m')

    def knn_predict(self):
        import pandas as pd

        self.load_test(self.test_csv)
        x = self.news_to_vector(self.fake_test, self.true_test)

        print('knn predicting ...')
        knn = joblib.load('HWf3.m')

        predictions = []
        batch_size = x.shape[0]//100
        print(batch_size, x.shape[0])
        for b in range(0, x.shape[0], batch_size):
            batch = x[b:min(b+batch_size, x.shape[0]), :]
            print(b, batch.shape)
            prediction = list(knn.predict(batch))
            predictions += prediction

        predictions_names = [self.y_cat.cat.categories[n] for n in predictions]
        data_frame = pd.DataFrame({'Id': self.id_test, 'Category': predictions_names})
        data_frame.to_csv(self.solution_file, index=False, sep=',')     # 将DataFrame存储为csv,index表示是否显示行名，default=True

    def load_csv(self, csv_file):
        import pandas as pd

        data_csv = pd.read_csv(csv_file, encoding='utf-8')

        self.fake_news = self.cut(data_csv['title1_zh'].values)
        self.true_news = self.cut(data_csv['title2_zh'].values)

        y_csv = data_csv['label']

        # 文本类别-转-数字类别
        self.y_cat = y_csv.astype('category')
        self.num_category = len(set(self.y_cat.values))

        print('\ny_cat.values: ', set(self.y_cat.values))
        print('category numbers: ', self.num_category, '\n')

        # 数字类别
        self.y = self.y_cat.cat.codes.values

    def load_test(self, csv_file):
        import pandas as pd

        data_csv = pd.read_csv(csv_file, encoding='utf-8')

        self.fake_test = self.cut(data_csv['title1_zh'].values)
        self.true_test = self.cut(data_csv['title2_zh'].values)
        self.id_test = data_csv['id'].values


if __name__ == '__main__':
    fn = FakeNews(n_neighbors=15)
    fn.knn_train()
    fn.knn_predict()
