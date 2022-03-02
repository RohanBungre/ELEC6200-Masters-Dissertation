import sys
import os
import argparse

import label
import features
import train
import classify

import time

start = time.time()

load_and_label = False 
add_Mean_Var = False
add_Reciprocal = False
add_Fourier = False
add_Wavelet = False
train_RandomForest = False
train_ExtraTrees = False
train_AdaBoost = False
train_DecisionTree = False
classifyData = False
processData = False

# Create the parser
my_parser = argparse.ArgumentParser(description='GDP Group Wild Dogs Tool')

# Add the arguments
my_parser.add_argument( '-l',
                        '--label',
                        action='store_true',
                        help='Labels the data found in the defined file structure')

my_parser.add_argument( '-f',
                        '--features',
                        action='store',
                        type=int,
                        choices=[1,2,3,4],
                        help='Calculates and adds features to the data')

my_parser.add_argument( '-t',
                        '--train',
                        action='store',
                        type=str,
                        choices=['RandomForest', 'ExtraTrees', 'AdaBoost'],
                        help='Trains the classifier on the current data')

my_parser.add_argument( '-p',
                        '--process',
                        action='store',
                        type=str,
                        metavar='filepath',
                        help='Processes the .pkl file specified ready for classification')

my_parser.add_argument( '-c',
                        '--classify',
                        action='store',
                        metavar='"_featured" filepath',
                        type=str,
                        help='Classifies the .pkl file specified')

my_parser.print_help()

# Execute the parse_args() method
args = my_parser.parse_args()

try:
    if args.label == True:
        print("Labelling data in directory...")
        load_and_label = True

    if args.features == 1:
        print("Adding mean/var features to labelled data in directory...")
        add_Mean_Var = True

    elif args.features == 2:
        print("Adding mean/var + reciprocal features to labelled data in directory...")
        add_Mean_Var = True
        add_Reciprocal = True

    elif args.features == 3:
        print("Adding mean/var + reciprocal + fourier features to labelled data in directory...")
        add_Mean_Var = True
        add_Reciprocal = True
        add_Fourier = True

    elif args.features == 4:
        print("Adding mean/var + reciprocal + fourier + wavelet features to labelled data in directory...")
        add_Mean_Var = True
        add_Reciprocal = True
        add_Fourier = True
        add_Wavelet = True

    if args.train == "RandomForest":
        print("Training Random Forest Classifier on current data...")
        train_RandomForest = True

    elif args.train == "ExtraTrees":
        print("Training Extra Trees Classifier on current data...")
        train_ExtraTrees = True

    elif args.train == "AdaBoost":
        print("Training Ada Boost Classifier on current data...")
        train_AdaBoost = True

    if args.process is not None:
        print("Processing {}".format(args.process))
        processData = True

    if args.classify is not None:
        print("Classifying {}".format(args.classify))
        classifyData = True

except TypeError as e:
    print('Error: Argument(s) not recognized')
    my_parser.print_help()
    sys.exit(1)

    

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


if __name__ == '__main__':

    if load_and_label: 
        label.loadAndLabel()

    if add_Mean_Var:
        features.addMeanVar()
    
    if add_Reciprocal or add_Fourier or add_Wavelet:
        if add_Reciprocal:
            features.addReciprocal()

        if add_Fourier:
            features.addFourier()

        if add_Wavelet:
            features.addWavelet()

    if train_RandomForest or train_ExtraTrees or train_AdaBoost:
        if train_RandomForest:
            train.trainAndSave("RandomForest")
        elif train_ExtraTrees:
            train.trainAndSave("ExtraTrees")
        elif train_AdaBoost:
            train.trainAndSave("AdaBoost")

    if processData:
        classify.addFeatures(args.process)

    if classifyData:
        classify.classifyData(args.classify)

    end = time.time()
    print('Time taken: ', end-start)