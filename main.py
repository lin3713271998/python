import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# 预测结果文件：src/step1/ground_truth/test_prediction.csv
def getPrediction():
    #********* Begin *********#
    train = pd.read_csv("src/step1/input/train.csv")
    test = pd.read_csv("src/step1/input/test.csv")
    y_train = train['TARGET']
    train.drop('TARGET', axis=1, inplace=True)
    train.drop('ID', axis=1, inplace=True)
    id_test = test['ID']
    submit = pd.DataFrame({'ID':[], 'TARGET': []})
    test.drop('ID', axis=1, inplace=True)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train, y_train)
    pred = clf.predict(test)
    submit['ID'] = id_test
    submit['TARGET'] = pred
    submit.to_csv("src/step1/ground_truth/test_prediction.csv", index=False)



    #********* End *********#