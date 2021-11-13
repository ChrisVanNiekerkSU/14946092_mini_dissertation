import os
from re import I
import pandas as pd
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import exposure
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import mapping
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from deepforest import main
import deepforest
from deepforest import preprocess
import slidingwindow
from shapely.geometry import Point
import torch

import json
from os import listdir
from os.path import isfile, join
import zipfile
import datetime
from deepforest import get_data
from deepforest import visualize
from deepforest import utilities
import sys
import gc
from pytorch_lightning.loggers import CSVLogger

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn import svm
import xgboost as xgb
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pickle

import time

def plot_predictions_from_df(df, img, colour = (255, 255, 0)):

    """
	Function to plot prediction on orthomosaic
	
	Parameters:
	df (dataframe): Dataframe containing all prediction bounding box coordinates
	img (array): Image in array format 
    colour: Colour of the bounding box as a tuple of BGR color.
	
	Returns:
    predicted_raster_image (array): Orthomosaic in array format containing with bounding boxes overlaid
	"""    

    # Draw predictions on BGR using DF .plot_predictions() method. 
    # https://deepforest.readthedocs.io/en/latest/source/deepforest.html#module-deepforest.visualize
    # https://github.com/weecology/DeepForest/blob/main/deepforest/visualize.py
    image = img[:,:,::-1]
    predicted_raster_image = visualize.plot_predictions(image, df, color=colour)

    return predicted_raster_image

def nearest_neighbor(tree, tree_locations_df):

    """
	Function to assign a predicted tree position to the nearest actual tree 
	
	Parameters:
	tree (tuple): X and Y coordinates of predicted tree location
	tree_locations_df (dataframe): dataframe containing all actual tree positions
	
	Returns:
	distance_to_nn (float): Euclidean distance from predicted position to the 
	                        nearest actual tree
	distance_to_nn_squared (float): distance_to_nn squared
	tree_id (str): Tree ID of nearest actual tree
	"""
    # Create array of all tree positions
    trees = np.asarray(tree_locations_df[['X', 'Y']])

    # Create array of all tree position under consideration
    tree = np.array(tree).reshape(-1, 1)

    # Calculate distances between tree in question and all other trees using Scikit-learn euclidean_distances() function
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
    # https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/metrics/pairwise.py#L226
    distances = euclidean_distances(tree.T, trees)

    # Find the nearest tree
    nn_idx = distances.argmin()
    distance_to_nn = distances.T[nn_idx][0]

    # Calculate squared distance
    distance_to_nn_squared = (distances.T[nn_idx][0])**2

    # Allocate nearest tree
    tree_id = tree_locations_df.loc[nn_idx, 'tree_id']

    return distance_to_nn, distance_to_nn_squared, tree_id

def actual_tree_data(tree_actual_df_path, sheet_name):

    """
	Function to load and format project spronsor-supplied data depending on dataset specified
	
	Parameters:
	tree_actual_df_path (str): Path to actual tree data file location
	sheet_name (str): Specification of dataset ('TRAIN', 'VAL', 'TEST)
	
	Returns:
    tree_actual_df_full (dataframe): Dataframe containing all data but with residual data after last tree removed
    tree_actual_df (dataframe): Same as tree_actual_df_full but with reduced columns
    tree_actual_df_no_dead (dataframe): Same as tree_actual_df_full but with dead trees removed
    min_height (int): Minimum measured tree height in dataset
	"""    

    # Load actual data
    tree_actual_df_full = pd.read_excel(open(tree_actual_df_path, 'rb'), sheet_name=sheet_name)

    # Determine last entry in dataset and filter out residual data
    last_valid_entry = tree_actual_df_full['Plot'].last_valid_index()
    tree_actual_df_full = tree_actual_df_full.loc[0:last_valid_entry]

    # Format data types for later use
    tree_actual_df_full = tree_actual_df_full.astype({'Plot':'int','Rep':'int','Tree no':'int'})
    tree_actual_df_full['tree_id'] = tree_actual_df_full['Plot'].astype('str') + '_' + tree_actual_df_full['Tree no'].astype('str')
    tree_actual_df_full['Hgt22Rod'] = pd.to_numeric(tree_actual_df_full['Hgt22Rod'], errors='coerce').fillna(0)

    # Reduce main dataframe
    tree_actual_df = tree_actual_df_full[['tree_id', 'Plot', 'Rep', 'Tree no', 'Hgt22Rod','24_Def', 'Hgt22Drone']]

    # Filter out dead trees for tree_actual_df_no_dead
    tree_actual_df_no_dead = tree_actual_df[tree_actual_df['24_Def'] != 'D']

    # Determine minimum tree height in dataset (excluding dead trees)
    min_height = tree_actual_df_no_dead['Hgt22Rod'].min()

    return tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height

def dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df, save_crops=False):

    """
	Function to split datareturn dead and non-dead tree positions and classify dead trees as 0 and non-dead trees as 1.
	
	Parameters:
	tree_point_calc_shifted_df (dataframe): Dataframe of ground truth (corrected) tree positions
	tree_actual_df (dataframe): Same as tree_actual_df_full but with reduced columns
	
	Returns:
    tree_positions_dead_filtered (dataframe): Dataframe of all DF predictions with dead trees filtered out
    tree_positions_not_dead_filtered (dataframe): Dataframe of all DF predictions with dead trees not filtered out
	"""  

    # Filter DeepForest datatframe for tree points only
    tree_positions = tree_point_calc_shifted_df[['X', 'Y', 'tree_id']]

    #  Merge with actual tree data from project sponsor
    tree_positions = tree_positions.merge(tree_actual_df, on='tree_id', how='inner')

    # Isolate non-dead trees and classify as 1
    tree_positions_not_dead = tree_positions[tree_positions['24_Def'] != 'D'].reset_index(drop=True)
    tree_positions_not_dead_filtered = tree_positions_not_dead[['X', 'Y']]
    tree_positions_not_dead_filtered['class'] = 1

    # Isolate dead trees and classify as 0
    tree_positions_dead = tree_positions[tree_positions['24_Def'] == 'D'].reset_index(drop=True)
    tree_positions_dead_filtered = tree_positions_dead[['X', 'Y']]
    tree_positions_dead_filtered['class'] = 0

    return tree_positions_dead_filtered, tree_positions_not_dead_filtered

def boxes_from_points(tree_positions_for_classification_all, expansion_size):

    """
	Function to create boxes from points based on a specified expansion size in metres
	
	Parameters:
    classification_all (dataframe): Dataframe of all actual trees with classifications
	expansion_size (int): Expansion size in metres to be used for cropping.
	
	Returns:
    tree_positions_for_classification_all (dataframe): Dataframe of all actual trees with classifications including box coordinates 
	"""  

    # Grow boxes from tree positions
    tree_positions_for_classification_all['xmin'] = tree_positions_for_classification_all['X'] - expansion_size / 2
    tree_positions_for_classification_all['ymin'] = tree_positions_for_classification_all['Y'] - expansion_size / 2
    tree_positions_for_classification_all['xmax'] = tree_positions_for_classification_all['X'] + expansion_size / 2
    tree_positions_for_classification_all['ymax'] = tree_positions_for_classification_all['Y'] + expansion_size / 2

    # Create polygons from expanded boxes
    geoms = []
    for idx in tree_positions_for_classification_all.index:

        xmin = tree_positions_for_classification_all.loc[idx,'xmin']
        xmax = tree_positions_for_classification_all.loc[idx,'xmax']
        ymin = tree_positions_for_classification_all.loc[idx,'ymin']
        ymax = tree_positions_for_classification_all.loc[idx,'ymax']    

        geom = Polygon([[xmin, ymin], [xmin,ymax], [xmax,ymax], [xmax,ymin]])
        geoms.append(geom)

    tree_positions_for_classification_all['geometry'] = geoms

    return tree_positions_for_classification_all

def crop_pixels_to_df(ortho_cropped_for_cropping, classification_all, expansion_size = 36, save_crops = False, train=True):

    """
	Function to generate dataset for model training (image flattening)
	
	Parameters:
	ortho_cropped_for_cropping (array): Orthomosaic in Numpy array format to be used for cropping
	classification_all (dataframe): Dataframe of all actual trees with classifications
    expansion_size (int): Expansion size in pixels to be used for cropping. Default = 36
	save_crops (bool): Option to save crops. Default=False
    train (bool): Option to be set during training to include classifications for model training. Default=True
	
	Returns:
    crop_df (dataframe): Dataframe including all flattened images. Shape (m x n) = (# of samples, expansion_size**2*3). Add 1 to n if train==True.
	"""    
    # Create empty array to store all image data
    images_array = np.empty((classification_all.shape[0], expansion_size*expansion_size*3))

    # Create dataframe to store all data
    crop_df = pd.DataFrame()
    col_counter = 0 
    for idx in classification_all.index:

        # Assign tree position to geo
        geo = classification_all.loc[idx,'geometry']

        # Clip (crop) orthomosaic
        clip = ortho_cropped_for_cropping.rio.clip([geo])
        clip = np.array(clip)
        clip = clip.astype('uint8')
        clip_rgb = np.moveaxis(clip, 0, 2).copy()

        y = clip_rgb.shape[1]
        x = clip_rgb.shape[0]
        startx = x//2-(expansion_size//2)
        starty = y//2-(expansion_size//2)    
        clip_rgb_cropped = clip_rgb[starty:starty+expansion_size,startx:startx+expansion_size]
        clip_rgb_cropped = clip_rgb_cropped.astype('uint8')

        # Flatten crop
        clip_rgb_flat = clip_rgb_cropped.flatten().T
        images_array[idx] = clip_rgb_flat

        # If save option is selected, save crops for visual analysis
        if save_crops == True:
            tree_id_for_save = classification_all.loc[idx,'tree_id_pred']
            model_for_save = classification_all.loc[0,'model']
            tree_class_path = 'tree_classifier_crops/' + str(tree_id_for_save) + '_' + str(model_for_save) + '.png'
            plt.imsave(tree_class_path,arr=clip_rgb)

    # Create dataframe of image pixel data and assign class is train == True
    crop_df = pd.DataFrame(data=images_array)
    if train == True:
        crop_df['class'] = classification_all['class']

    return crop_df

def classification_scores(y_true, y_pred, y_pred_prob, model, features=None, feature_list_id=None):

    """
	Function to generate scores for a classification model output
	
	Parameters:
	y_true (array): Ground truth target variable 
	y_pred (array): Predicted target variable 
    y_pred_prob (array): Prediction probability of positive class
	model (str): Model utilised
    features (list): List of features used (for testing only). Default=None
    feature_list_id (int): Identifier for feature set (for testing only). Default=None
	
	Returns:
    score_df (dataframe): Dataframe with accuracy, f1, precision, recall, auc and confusion matrix
	"""    

    # Create scores dataframe
    score_df = pd.DataFrame(columns=['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'features', 'feature_list_idx'])

    # Determine evaluation metrics
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    score_df['confusion'] = score_df['confusion'].astype(object)
    score_df['features'] = score_df['features'].astype(object)
    score_df.loc[0,'model'] = model
    score_df.loc[0,'accuracy'] = accuracy_score(y_true, y_pred)
    score_df.loc[0,'f1'] = f1_score(y_true, y_pred)
    score_df.loc[0,'precision'] = precision_score(y_true, y_pred)
    score_df.loc[0,'recall'] = recall_score(y_true, y_pred)
    score_df.loc[0,'confusion'] = [confusion_matrix(y_true, y_pred)]
    score_df.loc[0,'auc'] = roc_auc_score(y_true, y_pred_prob) 
    
    if features != None or feature_list_id != None:
        score_df.loc[0,'feature_list_idx'] = feature_list_id
        score_df.loc[0,'features'] = features

    return score_df

################################################
#         BUILD DEAD TREE CLASSIFIER           #
################################################

# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'

# Clipped CHM paths
chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
chm_train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
chm_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_val_clip.tif")

# Load clipped ortho path
ortho_name_train = 'ortho_train_clip.tif'
ortho_name_val = 'ortho_val_clip.tif'
ortho_name_test = 'ortho_test_clip.tif'
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
ortho_clipped_path_train = ortho_clipped_root + '/' + ortho_name_train
ortho_clipped_path_val = ortho_clipped_root + '/' + ortho_name_val
ortho_clipped_path_test = ortho_clipped_root + '/' + ortho_name_test

# Import actual tree data from Sappi
tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df_train, tree_actual_df_full_train, tree_actual_df_no_dead_train, min_height_train = actual_tree_data(tree_actual_df_path, 'TRAIN')
tree_actual_df_val, tree_actual_df_full_val, tree_actual_df_no_dead_val, min_height_val = actual_tree_data(tree_actual_df_path, 'VAL')
tree_actual_df_test, tree_actual_df_full_test, tree_actual_df_no_dead_test, min_height_test = actual_tree_data(tree_actual_df_path, 'TEST')

# Get tree positions from shapfiles and csv files and convert to geoPandas geometry
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y

# Set tree position to geometry type for GIS applications
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y

# Set expansion size for boxes in pixels and convert to metres as per GSD and add 10% buffer
expansion_size = 36
expansion_size_metres = expansion_size*0.0317*1.1

# Get dead and non-dead tree datasets from the train, validation and test datasets
dead_trees_classified_train, non_dead_tree_classified_train  = dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df_train, save_crops=False)
classification_all_train = pd.concat([dead_trees_classified_train[['X', 'Y', 'class']], non_dead_tree_classified_train[['X', 'Y', 'class']]], ignore_index=True)

dead_trees_classified_val, non_dead_tree_classified_val  = dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df_val, save_crops=False)
classification_all_val = pd.concat([dead_trees_classified_val[['X', 'Y', 'class']], non_dead_tree_classified_val[['X', 'Y', 'class']]], ignore_index=True)

dead_trees_classified_test, non_dead_tree_classified_test  = dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df_test, save_crops=False)
classification_all_test = pd.concat([dead_trees_classified_test[['X', 'Y', 'class']], non_dead_tree_classified_test[['X', 'Y', 'class']]], ignore_index=True)

# Open all orthomosaics (train, validation and test)
ortho_cropped_for_cropping_train = rxr.open_rasterio(ortho_clipped_path_train, masked=True).squeeze()
ortho_cropped_for_cropping_val = rxr.open_rasterio(ortho_clipped_path_val, masked=True).squeeze()
ortho_cropped_for_cropping_test = rxr.open_rasterio(ortho_clipped_path_test, masked=True).squeeze()

# Generate boxes for cropping according to expansion size in metres
classification_all_train = boxes_from_points(classification_all_train, expansion_size_metres)
classification_all_val = boxes_from_points(classification_all_val, expansion_size_metres)
classification_all_test = boxes_from_points(classification_all_test, expansion_size_metres)

# Create dataset for model training
crop_df_train = crop_pixels_to_df(ortho_cropped_for_cropping_train, classification_all_train, expansion_size = expansion_size, save_crops = False)
crop_df_val = crop_pixels_to_df(ortho_cropped_for_cropping_val, classification_all_val, expansion_size = expansion_size, save_crops = False)
crop_df_test = crop_pixels_to_df(ortho_cropped_for_cropping_test, classification_all_test, expansion_size = expansion_size, save_crops = False)

# Separate training data (X) and target (Y)
x_train = crop_df_train.drop(columns=['class'],axis=1)
x_val = crop_df_val.drop(columns=['class'],axis=1)
x_test = crop_df_test.drop(columns=['class'],axis=1)
y_train = crop_df_train['class']
y_val = crop_df_val['class']
y_test = crop_df_test['class']
all_x = pd.concat([x_test,x_train])

# Scale image data to [0,1] using max pixel value
x_train = x_train/255
x_test = x_train/255

# Create dataframe to store scores
svm_scores = pd.DataFrame()

# Set parameters
c = 0.5
kernel = 'rbf'
gamma = 'scale'

# Train SVM model
svm_model = svm.SVC(C=c, kernel=kernel, gamma=gamma, probability=True).fit(x_train, np.ravel(y_train))

# Generate predictions using x_train, x_val, and x_test
predictions_svm_train = svm_model.predict(x_train)
predictions_svm_train_prob = svm_model.predict_proba(x_train)

predictions_svm_val = svm_model.predict(x_val)
predictions_svm_val_prob = svm_model.predict_proba(x_val)

predictions_svm_test = svm_model.predict(x_test)
predictions_svm_test_prob = svm_model.predict_proba(x_test)

# Calculate scores and save to dataframe
score_df_svm_val = classification_scores(y_true=y_test, y_pred=predictions_svm_val, y_pred_prob=predictions_svm_val_prob[:, 1], model='svm')
score_df_svm_train = classification_scores(y_true=y_train, y_pred=predictions_svm_train, y_pred_prob=predictions_svm_train_prob[:, 1], model='svm')
score_df_svm_test_test= classification_scores(y_true=y_test_test, y_pred=predictions_svm_test, y_pred_prob=predictions_svm_test_prob[:, 1], model='svm')
svm_scores.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_svm_train.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]
svm_scores.loc[1,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_svm_val.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]
svm_scores.loc[2,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_svm_test_test.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]

# Write scores to CSV
svm_parameter_search.to_csv('tree_classifier_crops/classifier_scores/svm_scores_final.csv')

# Save model
svm_model_filename = 'tree_classifier_crops/saved_models/svm_v2.sav'
pickle.dump(svm_model, open(svm_model_filename, 'wb'))


#####################################################################
#    MODEL TESTING AND PARAMETER SEARCH (FOR INFORMATION ONLY!!)    #
#           INCOMPLETE IN CURRENT STATE DUE TO TESTING              #
#####################################################################
test_counter = 0
num_of_tests_for_avg = 10

for i in range(num_of_tests_for_avg):
    
    rf = RandomForestClassifier(n_estimators = 100, max_depth=None).fit(x_train, np.ravel(y_train))
    ab = AdaBoostClassifier(n_estimators = 50, learning_rate = 1).fit(x_train, np.ravel(y_train))
    svm_model = svm.SVC(probability=True).fit(x_train, np.ravel(y_train))
    xgb_model = xgb.XGBClassifier(n_estimators = 100, objective="reg:squarederror", use_label_encoder=False).fit(x_train, np.ravel(y_train))
    nn_model = MLPClassifier(hidden_layer_sizes=(64,64,64), max_iter=1500).fit(x_train, np.ravel(y_train))

    # Make predictions
    predictions_rf = rf.predict(x_test)
    predictions_ab = ab.predict(x_test)
    predictions_xgb = xgb_model.predict(x_test)
    predictions_svm = svm_model.predict(x_test)
    predictions_nn = nn_model.predict(x_test)

    predictions_rf_prob = rf.predict_proba(x_test)
    predictions_ab_prob = ab.predict_proba(x_test)
    predictions_svm_prob = svm_model.predict_proba(x_test)
    predictions_nn_prob = nn_model.predict_proba(x_test)

    score_df_rf = classification_scores(y_true=y_test, y_pred=predictions_rf, y_pred_prob=predictions_rf_prob[:, 1], model='random forest')
    score_df_ab = classification_scores(y_true=y_test, y_pred=predictions_ab, y_pred_prob=predictions_ab_prob[:, 1], model='adaboost')
    score_df_xgb = classification_scores(y_true=y_test, y_pred=np.round(predictions_xgb), y_pred_prob=predictions_xgb, model='xgboost')
    score_df_svm = classification_scores(y_true=y_test, y_pred=predictions_svm, y_pred_prob=predictions_svm_prob[:, 1], model='svm')
    score_df_nn = classification_scores(y_true=y_test, y_pred=predictions_nn, y_pred_prob=predictions_nn_prob[:, 1], model='nn')

    scores_df_all_models = pd.concat([score_df_rf,score_df_ab,score_df_xgb,score_df_svm, score_df_nn]).reset_index(drop=True)

    scores_df_all_tests = pd.concat([scores_df_all_tests, scores_df_all_models]).reset_index(drop=True)

    test_counter += 1

    if test_counter % 5 == 0:

        print(test_counter, ' tests of ', num_of_tests_for_avg*2, 'completed') 

scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc']] = scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc']].astype(float)
scores_df_all_tests_red = scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc', 'model']]
scores_df_all_tests_red = scores_df_all_tests_red.reset_index(drop=True)
scores_df_all_tests_avg = scores_df_all_tests_red.groupby(['model'], as_index=False).mean()

scores_df_all_tests.to_csv('tree_classifier_crops/classifier_scores/model_tests_avg_std.csv')
scores_df_all_tests_avg.to_csv('tree_classifier_crops/classifier_scores/model_tests_average_score_avg_std.csv')

cs = [0.5, 0.7, 1, 1.5]
kernels = ['rbf', 'linear','poly', 'sigmoid']
degrees = [1,5,7]
gammas = ['scale', 'auto']

grid_search_idx = 0
svm_parameter_search = pd.DataFrame(columns=['C', 'kernel', 'gamma', 'accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc'])
    
for c in cs:
    for kernel in kernels:
            for gamma in gammas:

            # if kernel in (['linear', 'rbf', 'sigmoid']) and degree>1: continue

            svm_model = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma, probability=True).fit(x_train, np.ravel(y_train))
            predictions_svm = svm_model.predict(x_test)
            predictions_svm_prob = svm_model.predict_proba(x_test)
            score_df_svm = classification_scores(y_true=y_test, y_pred=predictions_svm, y_pred_prob=predictions_svm_prob[:, 1], model='svm')
            svm_parameter_search.loc[grid_search_idx,'C'] = c
            svm_parameter_search.loc[grid_search_idx,'kernel'] = kernel
            # svm_parameter_search.loc[grid_search_idx,'degree'] = degree
            svm_parameter_search.loc[grid_search_idx,'gamma'] = gamma
            svm_parameter_search.loc[grid_search_idx,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_svm.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]

            grid_search_idx += 1 