import os
import sys
import pandas as pd
import pickle

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


#Functions to label the data
def loadRaw(folderPath):
    for filename in os.listdir(folderPath):
        if filename == "raw.pkl":
            data = pd.read_pickle(folderPath+pathstyle+ filename)
            data.reset_index(inplace=True)
            return data

def loadLabels(folderPath):
    for filename in os.listdir(folderPath):
        if filename == "labels.csv":
            labels = pd.read_csv(folderPath+pathstyle+filename)
            labels.reset_index(inplace=True)
            return labels

def labelData(data, labels):
    if('activity' in data.columns):
        data.drop('activity', 1, inplace=True)

    #Trim data to labelled space
    start = pd.Timestamp.to_pydatetime(pd.to_datetime(labels['start_time'].iloc[0], utc=True)).time()
    end = pd.Timestamp.to_pydatetime(pd.to_datetime(labels['end_time'].iloc[-1], utc=True)).time()
    data = data.drop(['mag_x', 'mag_y', 'mag_z', 'pressure'], axis=1)
    data = data.loc[(data['dt'].dt.time > start) & (data['dt'].dt.time < end)]
    data.reset_index(inplace=True)
    
    labelsList = []
    intervalIndex = 0
    intervalBegin = pd.to_datetime(labels['start_time'][intervalIndex], utc=True)
    intervalEnd = pd.to_datetime(labels['end_time'][intervalIndex], utc=True)
    for i in range(0, len(data)):
        time = pd.to_datetime(data['dt'][i])
        while(time > intervalEnd):
            if(intervalIndex + 1 >= len(labels)):
                break
            intervalIndex += 1
            intervalBegin = pd.to_datetime(labels['start_time'][intervalIndex], utc=True)
            intervalEnd = pd.to_datetime(labels['end_time'][intervalIndex], utc=True)
        if(time > intervalEnd or time < intervalBegin):
            labelsList.append("None")
        else:
            labelsList.append(labels["activity"][intervalIndex])
    data["activity"] = labelsList

    return data

def loadAndLabel():

    for subdir, dirs, files in os.walk(filepath):
        if subdir != "Data":
            data = loadRaw(subdir)
            labels = loadLabels(subdir)
            if (data is not None) & (labels is not None):
                print("Labelled Data from {}".format((str(pd.to_datetime(data['dt'][0], utc=True))[0:10])))
                labelledData = labelData(data, labels)

                labelledData.to_pickle(subdir+pathstyle+"/labelled.pkl")