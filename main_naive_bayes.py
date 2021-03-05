from supervised_learning.ct_processing.ct_learn import naive_bayes

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--path_data_set', default='E:\\google_drive\\data_sets\\balance_scale\\balance-scale.data',
                    help='path data set')

    args = vars(ap.parse_args())

    data_set = pd.read_csv(args['path_data_set'])

    # data_set.loc[data_set['class'] == 'L', ['class']] = 2
    # data_set.loc[data_set['class'] == 'B', ['class']] = 0
    # data_set.loc[data_set['class'] == 'R', ['class']] = 1

    #train and test
    values_train, values_test, class_train, class_test = train_test_split(data_set.values[:,1:], \
                                                                data_set["class"], test_size=0.3)

    predict_class = naive_bayes(values_train, values_test, class_train)

    print("acurácia: ", accuracy_score(predict_class, class_test))
    print(classification_report(predict_class, class_test))


if __name__ == "__main__":
    main()