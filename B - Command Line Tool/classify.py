import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import features

def loadClassifier():
    classifier_f = open("classifier.pkl", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()

    return classifier

def loadFeatureLevel():
    featureLevel_f = open("featureLevel.pkl", "rb")
    featureLevel = pickle.load(featureLevel_f)
    featureLevel_f.close()
    print("Feature level is {}".format(featureLevel))

    return featureLevel

def loadData(filepath):
    print("\nLoading file {}".format(filepath))
    data = pd.read_pickle(filepath)
    data.reset_index(inplace=True)
    return data

def addFeatures(filepath):
    data = loadData(filepath)
    print("Adding Required Features to data")
    featureLevel = int(loadFeatureLevel())

    if featureLevel >= 1:
        print("Adding level 1")
        features.add_mean_var_variables(data)
    if featureLevel >= 2:
        print("Adding level 2")
        data = features.add_reciprocal_column(data, "acc_x", "acc_y", log=True)
        data = features.add_reciprocal_column(data, "acc_x", "acc_z", log=True)
        data = features.add_reciprocal_column(data, "acc_y", "acc_z", log=True)
    if featureLevel >= 3:
        print("Adding level 3")
        labels_of_interest = ['jumping', 'galloping', 'sitting', 'barking', 'trotting', 'walking', 'laying', 'sniffing', 'playing', 'standing', 'eating', 'sleeping', 'hunting']
        value = "acc_y/acc_z"
        do_log=False
        freq_limit = 15
        useful_frequencies = {'acc_x': [0, 2, 3, 4, 5, 7], 'acc_y': [0, 4, 5], 'acc_z': [0, 2, 3, 4, 5], 'gyro_x': [5], 'gyro_y': [3, 4], 'acc_x/acc_y': [0, 2, 3, 4], 'acc_x/acc_z': [5], 'acc_y/acc_z': [0, 2, 3]}
        features.add_fourier_variables(useful_frequencies, data, 80)
    if featureLevel >= 4:
        print("Adding level 4")
        useful_wavelet_frequencies = {'acc_x': [14, 43, 62], 'acc_y': [48, 15], 'acc_z': [45], 'gyro_x': [12, 45, 62], 'gyro_y': [18], 'gyro_z': [48]}
        features.add_wavelet_variables(useful_wavelet_frequencies, data, 80)
    data.to_pickle("{}_featured.pkl".format(filepath[:-4]))
    print("Saved as {}_featured.pkl".format(filepath[:-4]))

def classifyData(filepath):
    classifier = loadClassifier()
    data_original = loadData(filepath)
    if 'level_0' in data_original.columns:
        data = data_original.drop('level_0', axis=1)
    data = data.drop(['index', 'dt','ms_counter', 'seq_counter'], axis=1).to_numpy()
    print("Classifying Data...")
    predictions = classifier.predict(data)

    data_original["Activity"] = predictions

    data_csv = data_original[['dt', 'Activity']]

    data_csv.to_csv('{}.csv'.format(filepath[:-4]))




