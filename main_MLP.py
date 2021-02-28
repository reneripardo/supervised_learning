# from supervised_learning.ct_processing.ct_cluster import
# from supervised_learning.ct_processing.ct_io import

import argparse
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--path_data_set', default='E:\\google_drive\\data_sets\\iris\\iris.data',
                    help='path data set')

    args = vars(ap.parse_args())

    data_set = pd.read_csv(args['path_data_set'])

    min_max_scaler = preprocessing.MinMaxScaler() # init 0 and 1
    array_data_set_scale = min_max_scaler.fit_transform(data_set.values[:,:4])# 0 to 1
    data_set.loc[:,:4] = array_data_set_scale

    data_set.loc[data_set['class'] == 'Iris-setosa', ['class']] = 2
    data_set.loc[data_set['class'] == 'Iris-versicolor', ['class']] = 0
    data_set.loc[data_set['class'] == 'Iris-virginica', ['class']] = 1

    #train and test
    values_train, values_test, class_train, class_test = train_test_split(data_set.values[:,:4], \
                                                                data_set["class"], test_size=0.3)


if __name__ == "__main__":
    main()