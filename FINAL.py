
import numpy as np
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd


class HWFinalSVM(object):
    def __init__(self):
        self.train_file = './data/train.csv'
        self.trainLabels_file = './data/trainLabels.csv'
        self.test_file = './data/test.csv'
        self.solution_file = 'testLabels.csv'

        self.clf = SVC(C=1.5, class_weight='balanced', probability=True)

        self.x = self.load_csv(self.train_file)
        self.y = self.load_csv(self.trainLabels_file).ravel()

        # 检查训练样本均衡度
        #print(np.mean(self.y == 1))

        self.scaler = StandardScaler()
        self.scaler.fit(self.x)
        self.x = self.scaler.transform(self.x)

    def load_csv(self, csv_file, header=None):
        import os
        import pandas as pd

        # 检查文件扩展名
        name, ext = os.path.splitext(csv_file)
        if ext != '.csv':
            print(csv_file, 'is not csv!')
        else:
            print('Loading', csv_file+'...')

        data_csv = pd.read_csv(csv_file, header=header, encoding='utf-8')
        print(data_csv.head())
        #print(data_csv.info())

        #field = list(data_csv.columns.values)
        #print(field)

        data_values = data_csv.values

        return data_values

    def standardize(self, data):
        return self.scaler.transform(data)

    def fit(self, data_set=None):
        print("SVM is training...")

        if data_set is None:
            x, y = self.x, self.y
        else:
            x, y = data_set

        self.clf.fit(x, y)

        joblib.dump(self.clf, 'HWf.m')  # 模型存盘
        print("SVM training mean accuracy: ", self.clf.score(x, y))     

    def predict(self, test_file_path=None):
        if test_file_path is None:
            test_file_path = self.test_file

        xt = self.load_csv(test_file_path)
        xt = self.standardize(xt)

        self.clf = joblib.load('Hwf.m')

        predictions = self.clf.predict(xt)

        Id = [n+1 for n in range(predictions.shape[0])]

        data_frame = pd.DataFrame({'Id': Id, 'Solution': predictions})

        #将DataFrame存储为csv,index表示是否显示行名，default=True
        data_frame.to_csv(self.solution_file, index=False, sep=',')

        #print(self.clf.predict_proba(xt))

        return predictions


if __name__ == '__main__':
    # SVM
    hw = HWFinalSVM()
    hw.fit()
    hw.predict()

    # check test result
    #hw.load_csv(hw.solution_file, header="infer")
    exit(0)
