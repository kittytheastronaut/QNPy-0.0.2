import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import collections
import numpy as np
import math

import random
import csv
from datetime import datetime

from cycler import cycler

import os
import glob

from tqdm import tqdm

import json

import pandas as pd

from sklearn import svm, datasets
import scipy.stats as ss

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error

from QNPy.CNP_ARCHITECTURE import DeterministicModel
from QNPy.CNP_METRICS import LogProbLoss, MSELoss
from QNPy.CNP_DATASETCLASS import LighCurvesDataset, collate_lcs

def create_prediction_folders(base_dir='./output/predictions'):
    sets = ['train', 'test', 'val']
    subfolders = ['plots', 'data']

    for set_folder in sets:
        set_path = os.path.join(base_dir, set_folder)
        if not os.path.exists(set_path):
            os.makedirs(set_path)
            print(f"Created folder: {set_path}")
        else:
            print(f"Folder already exists: {set_path}")

        for subfolder in subfolders:
            subfolder_path = os.path.join(set_path, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                print(f"Created folder: {subfolder_path}")
            else:
                print(f"Folder already exists: {subfolder_path}")

def prepare_output_dir(OUTPUT_PATH):
    """ -- the function prepare_output_dir takes the `OUTPUT_PATH` as an argument and removes all files in the output directory using os.walk method.


    Args:
    :param str OUTPUT_PATH: path to output folder

    How to use: prepare_output_dir(OUTPUT_PATH)
    """
    for root, dirs, files in os.walk(OUTPUT_PATH):
        for name in files:
            os.remove(os.path.join(root, name))
            


def load_trained_model(MODEL_PATH, device):
    """--Uploading trained model


    agrs:
    :param str MODEL_PATH = path to model directorium
    :param device = torch device CPU or CUDA

    How to use: model=load_trained_model(MODEL_PATH, device)
    """
    model = DeterministicModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    model.eval()
    
    return model



def get_criteria():
    """-- Gives the criterion and mse_metric

    How to use: criterion, mseMetric=get_criteria()
    """
    criterion = LogProbLoss()
    mseMetric = MSELoss()
    
    return criterion, mseMetric


import os
import pandas as pd

def remove_padded_values_and_filter(folder_path):
    """-- Preparing data for plotting. It'll remove the padded values from lc and it'll delete artifitially added lc with plus and minus errors. If your lc are not padded it'll only delete additional curves


    Args:
    :param str folder_path: Path to folder where the curves are. In this case it'll be './dataset/test' or './dataset/train' or './dataset/val'

    How to use: 
    if __name__ == "__main__":
    folder_path = "./dataset/test"  # Change this to your dataset folder

    remove_padded_values_and_filter(folder_path)
    """
    # Get the list of CSV files in the input folder
    csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.csv')]

    # Iterate over the CSV files
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)

        # Check if the filename contains "minus" or "plus"
        if "minus" in filename or "plus" in filename:
            # Delete the CSV file
            os.remove(file_path)
            print(f"Deleted file with 'minus' or 'plus' in the name: {filename}")
        elif "original" not in filename:
            # Delete the CSV file if it doesn't contain "original" in the name
            os.remove(file_path)
            print(f"Deleted file without 'original' in the name: {filename}")
        else:
            # Remove padded values from the curve and keep the original
            try:
                # Read the CSV file and load the data into a pandas DataFrame
                data = pd.read_csv(file_path)

                # Check if the DataFrame has more than 1 row
                if len(data) > 1:
                    # Check if 'mag' and 'magerr' columns are the same as the last observation
                    last_row = data.iloc[-1]
                    mag_values = data['cont']
                    magerr_values = data['conterr']
                    if not (mag_values == last_row['cont']).all() or not (magerr_values == last_row['conterr']).all():
                        # Keep rows where 'mag' and 'magerr' are not the same as the last observation
                        data = data[(mag_values != last_row['cont']) | (magerr_values != last_row['conterr'])]

                        # Overwrite the original CSV file with the modified DataFrame
                        data.to_csv(file_path, index=False)
                        print(f"Removed padding in file: {filename}")
                    else:
                        print(f"No padding removed for file: {filename}")
                else:
                    print(f"No padding removed for file: {filename}")

            except pd.errors.EmptyDataError:
                print(f"Error: Empty file encountered: {filename}")



OUTPUT_PATH="./output/predictions/"


def plot_function(target_x, target_y, context_x, context_y, pred_y, var, target_test_x, lcName, save=False, flagval=0, isTrainData=None, notTrainData=None):
    """-- Defines the plots of the light curve data and predicted mean and variance, and it should be imported separately
         

    Args:
    :param context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
    :param context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
    :param target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
    :param target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.
    :param target_test_x: Array of shape BATCH_SIZE x 400 that contains uniformly spread points across in [-2, 2] range.
    :param pred_y: Array of shape BATCH_SIZE x 400 that contains predictions across [-2, 2] range.
    :param var: An array of shape BATCH_SIZE x 400  that contains the variance of predictions at target_test_x points.
    """
    
    # Move to cpu
    target_x, target_y, context_x, context_y, pred_y, var = target_x.cpu(), target_y.cpu(), \
                                                              context_x.cpu(), context_y.cpu(), \
                                                              pred_y.cpu(), var.cpu()

    target_test_x = target_test_x.cpu()

    # Plot everything
    plt.plot(target_test_x[0], pred_y[0], 'b-', linewidth=1.5, label='mean model')
    plt.plot(target_x[0], target_y[0], linestyle='', linewidth=1.3, color='k')
    plt.plot(context_x[0], context_y[0], marker='|', linestyle='', linewidth=1.3, color='k', label='observations')

    plt.fill_between(
        target_test_x[0, :],
        pred_y[0, :] - var[0, :],
        pred_y[0, :] + var[0, :],
        alpha=0.2,
        facecolor='#ff9999',
        interpolate=True)

    plt.legend()

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid('off')
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.title(lcName)

    if save:
        if isTrainData and flagval == 0:
            savePath = os.path.join(OUTPUT_PATH, 'train')
        elif notTrainData and flagval == 1:
            savePath = os.path.join(OUTPUT_PATH, 'val')
        else:
            savePath = os.path.join(OUTPUT_PATH, 'test')

        lcName = lcName.split(',')[0]
        pltPath = os.path.join(savePath, 'plots', lcName + '.png')
        csvpath = os.path.join(savePath, 'data', lcName + '_predictions.csv')

        if not os.path.exists(os.path.join(savePath, 'plots')):
            os.makedirs(os.path.join(savePath, 'plots'))

        if not os.path.exists(os.path.join(savePath, 'data')):
            os.makedirs(os.path.join(savePath, 'data'))

        plt.savefig(pltPath, bbox_inches='tight')
        plt.clf()

        # Create dataframe with predictions and save csv
        d = {'mjd': target_test_x[0], 
             'mag': pred_y[0],
             'magerr': var[0]}
        
        df = pd.DataFrame(data=d)
        
        df.to_csv(csvpath, index=False)

        if not os.path.exists(pltPath):
            print(pltPath)
    else:
        plt.show()


def load_test_data(data_path):
    """-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

    Args:
    :param str data_path: path to Test data

    How to use: testLoader=load_test_data(DATA_PATH_TEST)
    """
    testSet = LighCurvesDataset(root_dir = data_path, status = 'test')
    testLoader = DataLoader(testSet,
                             num_workers = 0,
                             batch_size  = 1,      # must remain 1
                             shuffle=True,
                             pin_memory  = True)
    
    return testLoader


def load_train_data(data_path):
    """-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

    Args:
    :param str data_path: path to train data

    How to use: trainLoader=load_train_data(DATA_PATH_TRAIN)
    """
    train_set = LighCurvesDataset(root_dir=data_path, status='test')
    train_loader = DataLoader(train_set, 
                              num_workers=0, 
                              batch_size=1, 
                              shuffle=True, 
                              pin_memory=True)
    return train_loader

def load_val_data(data_path):
    """-- takes data_path as an argument, creates a LighCurvesDataset and returns a PyTorch DataLoader for the test set. The DataLoader is used to load and preprocess the test data in batches

    Args:
    :param str data_path: path to VAL data

    How to use: valLoader=load_val_data(DATA_PATH_VAL)
    """
    valSet = LighCurvesDataset(root_dir = data_path, status = 'test')
    valLoader = DataLoader(valSet,
                           num_workers = 0,
                           batch_size  = 1, 
                           pin_memory  = True)
    return valLoader

def plot_light_curves_from_test_set(model, testLoader, criterion, mseMetric, plot_function, device):
    testMetrics = {}

    """-- Ploting the transformed light curves from test set


    Args:
    :param model: Deterministic model
    :param testLoader: Uploaded test data
    :param criterion: criterion
    :param mseMetric: Mse Metric
    :param plot_function: plot function defined above
    :param device: torch device CPU or CUDA

    How to use: testMetrics = plot_light_curves_from_test_set(model, testLoader, criterion, mseMetric, plot_function, device)
    """
    with torch.no_grad():
        for data in tqdm(testLoader):
            # Unpack data
            lcName, context_x, context_y, target_x, target_y, target_test_x, measurement_error = data['lcName'], data['context_x'], \
                                                                              data['context_y'], data['target_x'], \
                                                                              data['target_y'], data['target_test_x'], data['measurement_error']

            # Move to gpu
            context_x, context_y, target_x, target_y, target_test_x, measurement_error = context_x.to(device), context_y.to(device), \
                                                                      target_x.to(device), target_y.to(device), \
                                                                      target_test_x.to(device), measurement_error.to(device)

            # Forward pass
            dist, mu, sigma = model(context_x, context_y, target_x)

            # Calculate loss
            loss = criterion(dist, target_y)
            loss = loss.item()

            # Calculate MSE metric
            mseLoss = mseMetric(target_y, mu, measurement_error)

            # Discard .csv part of LC name
            lcName = lcName[0].split('.')[0]

            # Add metrics to map
            testMetrics[lcName] = {'log_prob:': str(loss),
                                      'mse': str(mseLoss)}

            # Add loss value to LC name
            lcName = lcName + ", Log prob loss: " + str(loss) + ", MSE: " + str(mseLoss)

            # Predict and plot
            dist, mu, sigma = model(context_x, context_y, target_test_x)
            plot_function(target_x, target_y, context_x, context_y, mu, sigma, target_test_x, lcName, save = True, isTrainData = False, flagval =0)
    
    return testMetrics


def save_test_metrics(OUTPUT_PATH, testMetrics):
    """-- saving the test metrics as json file


    Args:
    :param str OUTPUT_PATH: path to output folder
    :param testMetrics: returned data from ploting function

    How to use: save_test_metrics(OUTPUT_PATH, testMetrics)
    """
    with open(OUTPUT_PATH + 'test/testMetrics.json', 'w') as fp:
        json.dump(testMetrics, fp, indent=4)

def plot_light_curves_from_train_set(trainLoader, model, criterion, mseMetric, plot_function, device):

    """-- Ploting the transformed light curves from train set


    Args:
    :param model: Deterministic model
    :param trainLoader: Uploaded trained data
    :param criterion: criterion
    :param mseMetric: Mse Metric
    :param plot_function: plot function defined above
    :param device: torch device CPU or CUDA

    How to use: trainMetrics = plot_light_curves_from_train_set(model, trainLoader, criterion, mseMetric, plot_function, device) 
    """
    # Plots all light curves from train set

    trainMetrics = {}

    counter = 0

    with torch.no_grad():
        for data in tqdm(trainLoader):
            # End after predicting given number of LCs
 #           if counter > 100:
 #               break
 #           counter += 1

            try:
                lcName, context_x, context_y, target_x, target_y, target_test_x, measurement_error = data['lcName'], data['context_x'], \
                                                                          data['context_y'], data['target_x'], data['target_y'], \
                                                                          data['target_test_x'], data['measurement_error']

                # Move to gpu
                context_x, context_y, target_x, target_y, target_test_x, measurement_error = context_x.to(device), context_y.to(device), \
                                                                          target_x.to(device), target_y.to(device), \
                                                                          target_test_x.to(device), measurement_error.to(device)

                # Forward pass
                dist, mu, sigma = model(context_x, context_y, target_x)

                # Calculate loss
                loss = criterion(dist, target_y)
                loss = loss.item()

                # Calculate MSE metric
                mseLoss = mseMetric(target_y, mu, measurement_error)

                # Discard .csv part of LC name
                lcName = lcName[0].split('.')[0]

                # Add metrics to map
                trainMetrics[lcName] = {'log_prob': str(loss),
                                        'mse': str(mseLoss)}

                # Add loss value to LC name
                lcName = lcName + ", Log prob loss: " + str(loss) + ", MSE: " + str(mseLoss)

                # Predict and plot
                dist, mu, sigma = model(context_x, context_y, target_test_x)
                plot_function(target_x, target_y, context_x, context_y, mu, sigma, target_test_x, lcName, save = True, isTrainData = True)
            except Exception as e:
                print(f'Error in: {lcName}')
    return trainMetrics


def save_train_metrics(OUTPUT_PATH, trainMetrics):
    """-- saving the train metrics as json file


    Args:
    :param str OUTPUT_PATH: path to output folder
    :param trainMetrics: returned data from ploting function

    How to use: save_train_metrics(OUTPUT_PATH, trainMetrics)
    """
    with open(OUTPUT_PATH + 'train/trainMetrics.json', 'w') as fp:
        json.dump(trainMetrics, fp, indent=4)

def plot_light_curves_from_val_set(model, valLoader, criterion, mseMetric, plot_function, device):

    """-- Ploting the transformed light curves from validation set


    Args:
    :param model: Deterministic model
    :param valLoader: Uploaded val data
    :param criterion: criterion
    :param mseMetric: Mse Metric
    :param plot_function: plot function defined above
    :param device: torch device CPU or CUDA

    How to use: valMetrics = plot_light_curves_from_val_set(model, valLoader, criterion, mseMetric, plot_function, device)
    """
    valMetrics = {}

    counter = 0

    with torch.no_grad():
        for data in tqdm(valLoader):
            # End after predicting given number of LCs
            if counter > 100:
                break
            counter += 1

            try:
                lcName, context_x, context_y, target_x, target_y, target_test_x, measurement_error = data['lcName'], data['context_x'], \
                                                                              data['context_y'], data['target_x'], data['target_y'], \
                                                                              data['target_test_x'], data['measurement_error']

                # Move to GPU
                context_x, context_y, target_x, target_y, target_test_x, measurement_error = context_x.to(device), context_y.to(device), \
                                                                              target_x.to(device), target_y.to(device), \
                                                                              target_test_x.to(device), measurement_error.to(device)

                # Forward pass
                dist, mu, sigma = model(context_x, context_y, target_x)

                # Calculate loss
                loss = criterion(dist, target_y)
                loss = loss.item()

                # Calculate MSE metric
                mseLoss = mseMetric(target_y, mu, measurement_error)

                # Discard .csv part of LC name
                lcName = lcName[0].split('.')[0]

                # Add metrics to map
                valMetrics[lcName] = {'log_prob': str(loss),
                                        'mse': str(mseLoss)}

                # Add loss value to LC name
                lcName = lcName + ", Log prob loss: " + str(loss) + ", MSE: " + str(mseLoss)

                # Predict and plot
                dist, mu, sigma = model(context_x, context_y, target_test_x)
                plot_function(target_x, target_y, context_x, context_y, mu, sigma, target_test_x, lcName, save=True, isTrainData=False, flagval=1, notTrainData=True)
            except Exception as e:
                print(f'Error in: {lcName}')
    return valMetrics


def save_val_metrics(OUTPUT_PATH, valMetrics):
    """-- saving the validation metrics as json file

    Args:
    :param str OUTPUT_PATH: path to output folder
    :param valMetrics: returned data from ploting function

    How to use: save_val_metrics(OUTPUT_PATH, valMetrics)
    """
    with open(OUTPUT_PATH + 'val/valMetrics.json', 'w') as fp:
        json.dump(valMetrics, fp, indent=4)
