import os
import sys
import glob
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from features import loadFeatured

#Setup global path
os_system = sys.platform
if os_system.startswith('win'):
    os_system = 'windows'

if os_system == 'linux':
    filepath = './Data'
    pathstyle = '/'
elif os_system == 'windows':
    filepath = 'Data'
    pathstyle = '\\'

folderPath = os.listdir(filepath)
folderPath = [ filepath+pathstyle+i for i in folderPath]

def split_test_train(data, training_variables, excluded_activities=[], test_ratio=0.25, perform_one_hot=False, undersample=True, oversample=False):
    excluded_activities.append('None')
    new_data = data[~data['activity'].isin(excluded_activities)]
    activities = new_data['activity'].unique()
    print("Activity Data Count:")
    print(new_data['activity'].value_counts())
    print()
    if(undersample):
        print("Undersampling to: " + str(np.min(new_data['activity'].value_counts())) + " data points per activity" )
    elif(oversample):
        print("Oversampling to: " + str(np.max(new_data['activity'].value_counts())) + " data points per activity" )
    independent_data = new_data[training_variables]
    
    one_hot=new_data['activity'].to_numpy()
    print(one_hot)
    if(perform_one_hot):
        one_hot = pd.get_dummies(new_data['activity'])
        if(undersample):
            rus = RandomUnderSampler()
            X_resampled, y_resampled = rus.fit_resample(independent_data.to_numpy(), one_hot.to_numpy())
            #Returns X_train,X_test,y_train,y_test, independent_data.columns, one_hot.columns
            return (train_test_split(X_resampled,y_resampled,test_size=test_ratio), independent_data.columns, one_hot.columns)
        elif(oversample):
            ros = RandomOverSampler()
            X_resampled, y_resampled = ros.fit_resample(independent_data.to_numpy(), one_hot)
            #Returns X_train,X_test,y_train,y_test, independent_data.columns, one_hot.columns
            return (train_test_split(X_resampled,y_resampled,test_size=test_ratio), independent_data.columns, one_hot.columns)
        else:
            return (train_test_split(independent_data.to_numpy(),one_hot.to_numpy(),test_size=test_ratio), independent_data.columns, one_hot.columns)
    else:
        one_hot=new_data['activity'].to_numpy()
        if(undersample):
            rus = RandomUnderSampler()
            X_resampled, y_resampled = rus.fit_resample(independent_data.to_numpy(), one_hot)
            #Returns X_train,X_test,y_train,y_test, independent_data.columns, one_hot.columns
            return (train_test_split(X_resampled,y_resampled,test_size=test_ratio), independent_data.columns, [])
        elif(oversample):
            ros = RandomOverSampler()
            X_resampled, y_resampled = ros.fit_resample(independent_data.to_numpy(), one_hot)
            #Returns X_train,X_test,y_train,y_test, independent_data.columns, one_hot.columns
            return (train_test_split(X_resampled,y_resampled,test_size=test_ratio), independent_data.columns, [])
        else:
            return (train_test_split(independent_data.to_numpy(),one_hot,test_size=test_ratio), independent_data.columns, [])

def train_classifier(classifier, data, columns_to_use, exclude=[], name = 'Classifier', undersample_train=False, oversample_train = False):

    (X_train,X_test,y_train,y_test),variable_columns, one_hot_columns = split_test_train(data, columns_to_use, excluded_activities=exclude, test_ratio=0.01, perform_one_hot=False, undersample=undersample_train, oversample=oversample_train)
    
    # Train classifier
    print("Training " + name + "...")
    start_time = time.time()
    classifier.fit(X_train,y_train)
    end_time = time.time()
    print(name + " Training Complete in " + str(end_time-start_time) + " seconds")
    print()
    
    return one_hot_columns

def trainAndSave(classifierType):

    activities_to_exclude = ['barking', 'jumping', 'playing']

    dfList = []

    for i, folder in enumerate(folderPath):
            data = loadFeatured(folder)
            if data is not None:
                dfList.append(data)

    useful_values = [x for x in list(dfList[0].columns) if x not in ['level_0','index','dt','ms_counter','seq_counter','deltaMillis','activity']]

    if classifierType == "ExtraTrees":
        classifier = ExtraTreesClassifier(n_estimators=300, criterion='entropy')
    elif classifierType == "RandomForest":
        classifier = RandomForestClassifier(n_estimators=10, verbose=0)
    elif classifierType == "AdaBoost":
        classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators = 100)

    train_classifier(classifier, pd.concat(dfList), useful_values, exclude=activities_to_exclude, name=classifierType, undersample_train=True)
    save_classifier = open("classifier.pkl","wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

