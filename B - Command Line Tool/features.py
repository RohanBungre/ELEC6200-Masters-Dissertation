import os
import sys
import glob
import pandas as pd
import pickle
#from IPython import display
import numpy as np
from skimage import util
import pywt

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


#Functions to add features
def loadLabelled(folderPath):
    for filename in os.listdir(folderPath):
        if filename == "labelled.pkl":
            print("\nLoading file from {}".format(folderPath))
            data = pd.read_pickle(folderPath+pathstyle+ filename)
            data.reset_index(inplace=True)
            return data

def loadFeatured(folderPath):
    for filename in os.listdir(folderPath):
        if filename == "dataFeatured.pkl":
            print("\nLoading file from {}".format(folderPath))
            data = pd.read_pickle(folderPath+pathstyle+ filename)
            return data

def add_mean_var_variables(data, windowSize=10, show_progress=False):
    query_variables = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    composite_variables = []
    values = {}
    for variable in query_variables:
            values[variable + "_MEAN"] = []
            values[variable + "_VAR"] = []
    for i in range(0, len(data.index)):
        minIndex = 0
        maxIndex = len(data.index)
        if(i >= windowSize):
            minIndex = i - windowSize
        if(i <= len(data.index)-1-windowSize):
            maxIndex = i+windowSize
        slice = data.iloc[minIndex:maxIndex+1]
        for variable in query_variables:
            values[variable + "_MEAN"].append(slice[variable].mean())
            values[variable + "_VAR"].append(slice[variable].var())
        if(show_progress):
            display.clear_output()
            print("Processing Dataframe: " + str(dataIndex))
            print("Percentage Finished: " + str(float(i)*100/len(data.index)) + "%")
    for variable in query_variables:
            data[variable + "_MEAN"] = values[variable + "_MEAN"]
            data[variable + "_VAR"] = values[variable + "_VAR"]
            if(not (variable + "_MEAN") in composite_variables):
                composite_variables.append((variable + "_MEAN"))
            if(not (variable + "_VAR") in composite_variables):
                composite_variables.append((variable + "_VAR"))
    return composite_variables

def add_reciprocal_column(df, column1, column2, log=False, verbose=False):
    count = 0
    if(column1 in df.columns and column2 in df.columns):
        print("Forming reciprocal of " + column1 + "/" + column2)
        reciprocal_vals = []
        if((column1 + "/" + column2) in df.columns):
            df.drop(column1 + "/" + column2, 1, inplace=True)
        
        new_df = df
        for i in range(len(df)):
            x = df[column1][i]
            y = df[column2][i]
            if(x == 0):
                x = 0.00001
            if(y == 0):
                y = 0.00001
            
            result = x/y
            if(np.isnan(result) or np.isinf(result)):
                count += 1
                result = 0.00001
            if(log):
                result = np.log10(np.abs(result))
            reciprocal_vals.append(result)
            
            if(verbose):
                display.clear_output()
                print("Forming reciprocal of " + column1 + "/" + column2 + " for Dataframe")
                print("Percentage Finished: " + str(float(i)*100/len(df)) + "%")
                print("NaN or Inf results: " + str(count))
        new_df[column1 + "/" + column2] = reciprocal_vals
        
        return new_df
    else:
        print("Columns not present in Dataframe. Cannot form Reciprocal!")

# M = Window Size, Rate in Hz
def calculate_fourier(column, df, rate, label, use_label=False, M = 64, freq_limit=-1):
    if(column in df.columns):
        data = np.array(df[column])
        if(use_label):
            data = np.array(df.loc[df['activity'] == label][column])
        N = data.shape[0]
        if(N <= M):
            if(use_label):
                print("Insufficient Data for activity: " + label)
            else:
                print("Insufficient Complete Data")
            return []
        L = N / rate
        slices = util.view_as_windows(data, window_shape=(M,), step=1)
        slices = slices * np.hanning(M + 1)[:-1]
        slices = slices.T
        spectrum = np.fft.fft(slices, axis=0)[:M // 2 + 1:-1]
        spectrum = np.abs(spectrum)
        

        S = np.abs(spectrum)
        #S = 20 * np.log10(S / np.max(S))
        S = S / np.max(S)

        return S

def add_fourier_variables(variable_frequencies, data, window_size):
    fourier_variables = []
    for variable in variable_frequencies:
        frequency_indices = variable_frequencies[variable]
        S = calculate_fourier(variable, data, 50, '', use_label=False, M=window_size)
        for frequency_index in frequency_indices:
            new_column = []
            new_column.extend(np.zeros(window_size-1))
            new_column.extend(S[frequency_index])
            #print(variable + " at frequency index: " + str(frequency_index))
            new_var_name = variable + "_freq_" + str(frequency_index)
            if(new_var_name in data.columns):
                data.drop(new_var_name, 1, inplace=True)
            data[new_var_name] = new_column
            if(not new_var_name in fourier_variables):
                fourier_variables.append(new_var_name)
    return fourier_variables

def sampling(data, N=80):
    # Every N data points (1.6s)
    
    list_np_X = list()
    list_np_Y = list()
        
    IMU_data = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    Y_N = np.array(data['acc_x'])
    Y_N = Y_N.reshape(len(Y_N), 1)
    print(Y_N.shape)

    #Values of ind. variables for all data points
    X_N = data[IMU_data].to_numpy()

    XY_N = np.concatenate((X_N, Y_N), axis=1)

    window_X = util.view_as_windows(XY_N, window_shape=(N,7), step=1)
    window_X = np.array([x[0] for x in window_X])


    X_N = window_X[:,:,np.arange(len(IMU_data))]
    print(X_N.shape)
    Y_N = window_X[:,:,-1]
    print(Y_N.shape)
    Y_N = np.array([i[0] for i in Y_N])
    
    list_np_X.append(X_N)
    list_np_Y.append(Y_N)
    
    dataX = np.concatenate(list_np_X, axis=0)
    dataY = np.concatenate(list_np_Y, axis=0)
    
    return dataX, dataY

# Wavelet Transforms 
def cwt_coeffs(X, column_indices=[], n_scales=64, wavelet_name='morl'):
    cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    scales = np.arange(1, n_scales + 1)
    
    X_cwt = []
    variable_columns = []
    if(column_indices == []):
        variable_columns = range(X.shape[2])
    else:
        variable_columns = column_indices
    for signal in variable_columns:
        print('Column: ', cols[signal])
        
        coeffs_list = list()
        percentCompleted = 0
        for sample in range(X.shape[0]):
            if(sample*100/X.shape[0] > percentCompleted + 10):
                print(str(percentCompleted+10) + "% Completed")
                percentCompleted += 10
            coeffs, freqs = pywt.cwt(X[sample, :, signal], scales, wavelet_name)
            #Coeffs = Coefficients of 80 datapoints for each scale (Shape=(64,80))
            coeffs_mean = [ np.mean(i) for i in coeffs ]
            coeffs_list.append(coeffs_mean)
                     
        if X_cwt == []:
            coeffs_list = np.array(coeffs_list)#.reshape(len(coeffs_list), 1)
            print(coeffs_list.shape)
            X_cwt.append(coeffs_list)
        else:
            coeffs_list = np.array(coeffs_list)#.reshape(len(coeffs_list), 1)
            X_cwt.append(coeffs_list)
            
    return X_cwt

def add_wavelet_variables(variable_frequencies, data, window_size):
    cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    wavelet_variables = []
    dataX, dataY = sampling(data, N=window_size)
    cwt_X = cwt_coeffs(dataX, n_scales=64, wavelet_name='morl')
    
    for variable in variable_frequencies:
        frequency_indices = variable_frequencies[variable]
        column_cwt = cwt_X[cols.index(variable)]
        for frequency_index in frequency_indices:
            new_column = []
            new_column.extend(np.zeros(window_size-1))
            new_column.extend(np.array(column_cwt).T[frequency_index])
            print(variable + " wavelet at frequency index: " + str(frequency_index))
            new_var_name = variable + "_wavfreq_" + str(frequency_index)
            if(new_var_name in data.columns):
                data.drop(new_var_name, 1, inplace=True)
            data[new_var_name] = new_column
            if(not new_var_name in wavelet_variables):
                wavelet_variables.append(new_var_name)

    return wavelet_variables

def addMeanVar():
    for i, folder in enumerate(folderPath):
        data = loadLabelled(folder)
        if data is not None:
            print("Adding Mean and Variance...")
            add_mean_var_variables(data)
            data.to_pickle(folder+"/dataFeatured.pkl")
    save_featureLevel = open("featureLevel.pkl","wb")
    pickle.dump("1", save_featureLevel)
    save_featureLevel.close()

def addReciprocal():
    for i, folder in enumerate(folderPath):
        data = loadFeatured(folder)
        if data is not None:
            print("Adding Reciprocals...")
            data = add_reciprocal_column(data, "acc_x", "acc_y", log=True)
            data = add_reciprocal_column(data, "acc_x", "acc_z", log=True)
            data = add_reciprocal_column(data, "acc_y", "acc_z", log=True)
            data.to_pickle(folder+"/dataFeatured.pkl")
    save_featureLevel = open("featureLevel.pkl","wb")
    pickle.dump("2", save_featureLevel)
    save_featureLevel.close()

def addFourier():
    for i, folder in enumerate(folderPath):
        data = loadFeatured(folder)
        if data is not None:
            print("Adding Fourier...")
            labels_of_interest = ['jumping', 'galloping', 'sitting', 'barking', 'trotting', 'walking', 'laying', 'sniffing', 'playing', 'standing', 'eating', 'sleeping', 'hunting']
            value = "acc_y/acc_z"
            do_log=False
            freq_limit = 15
            useful_frequencies = {'acc_x': [0, 2, 3, 4, 5, 7], 'acc_y': [0, 4, 5], 'acc_z': [0, 2, 3, 4, 5], 'gyro_x': [5], 'gyro_y': [3, 4], 'acc_x/acc_y': [0, 2, 3, 4], 'acc_x/acc_z': [5], 'acc_y/acc_z': [0, 2, 3]}
            add_fourier_variables(useful_frequencies, data, 80)
            data.to_pickle(folder+"/dataFeatured.pkl")
    save_featureLevel = open("featureLevel.pkl","wb")
    pickle.dump("3", save_featureLevel)
    save_featureLevel.close()

def addWavelet():
    useful_wavelet_frequencies = {'acc_x': [14, 43, 62], 'acc_y': [48, 15], 'acc_z': [45], 'gyro_x': [12, 45, 62], 'gyro_y': [18], 'gyro_z': [48]}
    for i, folder in enumerate(folderPath):
        data = loadFeatured(folder)
        if data is not None:
            print("Adding Wavelet...")
            add_wavelet_variables(useful_wavelet_frequencies, data, 80)
            print(data.head())
            data.to_pickle(folder+"/dataFeatured.pkl")
    save_featureLevel = open("featureLevel.pkl","wb")
    pickle.dump("4", save_featureLevel)
    save_featureLevel.close()

