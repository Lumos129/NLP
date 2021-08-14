
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class MoviesClassifierKnn(object):
    def __init__(self, csv_file_name='movies_mt2_sort.csv', n_neighbors=5):
        self.n_neighbors = n_neighbors

        self.load_csv(csv_file_name)

        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')

        # 初始化(实例化)
        self.vectorizer = CountVectorizer(max_features=None)
        self.tf_idf_transformer = TfidfTransformer()

        self.story_to_vector()

    def story_to_vector(self):
        x_cut = self.cut(self.training_x)

        self.vectorizer.fit(x_cut)
        vector_training_x = self.vectorizer.transform(x_cut)

        self.tf_idf_transformer.fit(vector_training_x)
        tf_idf_training_x = self.tf_idf_transformer.transform(vector_training_x)
        self.training_x_vec = tf_idf_training_x.toarray()
        print(self.training_x_vec.shape)

        xt_cut = self.cut(self.testing_x)
        tf_idf_testing_x = self.tf_idf_transformer.transform(self.vectorizer.transform(xt_cut))
        self.testing_x_vec = tf_idf_testing_x.toarray()
        print(self.testing_x_vec.shape)

    def text_to_vector(self):
        x_cut = self.cut(self.training_x)
        #print(x_cut)

        vector_training_x = self.vectorizer.fit_transform(x_cut)
        self.vocabulary = self.vectorizer.vocabulary_
        #print(self.vocabulary)
        #print(vector_training_x.toarray())

        tf_idf_training_x = self.tf_idf_transformer.fit_transform(vector_training_x)
        self.training_x_vec = tf_idf_training_x.toarray()
        print(self.training_x_vec.shape)

        xt_cut = self.cut(self.testing_x)
        tf_idf_testing_x = self.tf_idf_transformer.transform(self.vectorizer.transform(xt_cut))
        self.testing_x_vec = tf_idf_testing_x.toarray()
        print(self.testing_x_vec.shape)

    def cut(self, docs):
        """
        Cut with jieba
        :return:
        """
        import jieba.analyse

        print('jieba cutting.....')

        jieba.setLogLevel(jieba.logging.INFO)   # 引入jieba模块后加入该代码，代码即可不报警
        jieba.set_dictionary('dict.txt')

        docs_list = []
        for story in docs:
            terms_list = jieba.lcut(story, cut_all=False, HMM=True)
            #terms_list = [term for term in terms_list if len(term) > 1]
            docs_list.append(' '.join(terms_list))

        return docs_list

    # kNN Training
    def knn_train(self):
        print('knn training ...')
        self.knn.fit(self.training_x_vec, self.training_y)

    def knn_test(self, training_data=None):
        print('knn testing ...')
        predictions = self.knn.predict(self.training_x_vec)
        self.performance(self.training_y, predictions)
        return predictions

    # kNN Algorithm
    def knn_predict(self, testing_data=None):
        print('knn predicting ...')
        predictions = self.knn.predict(self.testing_x_vec)
        #print(np.mean(predictions == self.testing_y))

        self.performance(self.testing_y, predictions)
        return predictions

    def performance(self, y_true, y_predict):
        from sklearn.metrics import classification_report, confusion_matrix
        import pandas as pd

        #print(classification_report(y_true, y_predict, zero_division=0))

        target_names = [self.y_cat.cat.categories[n] for n in set(y_true)]     # set(y_true)????
        print(classification_report(y_true, y_predict, zero_division=0, target_names=target_names))

    def load_csv(self, csv_file):
        from sklearn.model_selection import train_test_split
        import pandas as pd

        data_csv = pd.read_csv(csv_file, encoding='utf-8')
        #print(data_csv.head())
        #print(data_csv.info())

        # 清理数量过少的类型
        data_csv = self.clean_csv(data_csv, key='Type', n_min=50)

        # 统计影片类型分布
        movie_type_set = set(data_csv['Type'].values)
        for movie_type in movie_type_set:
            print(movie_type, data_csv[data_csv['Type'] == movie_type].values.shape[0])

        X = data_csv['Story']
        y = data_csv['Type']

        # 文本类别-转-数字类别
        self.y_cat = y.astype('category')
        self.num_category = len(set(self.y_cat.values))

        print('\ny_cat.values: ', set(self.y_cat.values))
        print('category numbers: ', self.num_category, '\n')

        # 数字类别
        self.y_num = self.y_cat.cat.codes.values
        #print(type(self.y_num), self.y_num, '\n')
        #print(yd.cat.categories[0], '\n')
        #self.target_names = [self.yd.cat.categories[n] for n in range(self.num_category)]

        self.training_x, self.testing_x, self.training_y, self.testing_y = train_test_split(X, self.y_num,
                                                                                            #random_state=200,
                                                                                            test_size=500)   # seed固定, 实验结果可再现

    def clean_csv(self, csv_data, key=None, n_min=1):
        """
        清理数量过少的类型, 清理特定类型影片
        :param csv_data:
        :param key:
        :param n_min:
        :return:
        """
        # 清理数量过少的类型
        for item in set(csv_data[key].values):
            if len(csv_data[csv_data[key] == item].values) < n_min:
                csv_data = csv_data[~csv_data[key].str.contains(item, regex=False)]

        # 清理特定类型影片
        #csv_data = csv_data[~csv_data['Type'].str.contains('影展', regex=False)]
        return csv_data


if __name__ == '__main__':
    mc = MoviesClassifierKnn(n_neighbors=35)
    mc.knn_train()
    #mc.knn_test()
    mc.knn_predict()