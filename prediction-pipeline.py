import os
import time
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

def find_duplicates(tree_locations_pred_df):

    """
	Function to find duplicate tree positions allocated to a actual tree positions
	
	Parameters:
	tree_locations_pred_df (dataframe): Dataframe including all estimated tree positions and allocated actual trees
	
	Returns:
    ids_to_remove_pred_id (list): List of suggested tree positions to remove based on unallocated trees
    tree_locations_pred_df (dataframe): Geopandas dataframe of all tree positions estimated by LM, allocated trees, distances to nearest neighbours and heights but including additional data after checking for duplicate allocations
	"""

    # Find actual tree positions assigned more than one estimation
    duplicates_series = tree_locations_pred_df.duplicated(subset=['tree_id_pred'], keep=False)
    duplicates_list_idx = list(duplicates_series[duplicates_series == True].index)
    duplicates_list = list(tree_locations_pred_df.loc[duplicates_list_idx, 'tree_id_pred'])
    duplicates_list = list(set(duplicates_list))

    # Loop through all duplicates to determine which is closest to the actual tree
    ids_to_remove_pred_id = []
    for duplicate in duplicates_list:

        # Locate assigned tree with highest nearest neighbour distance
        duplicate_idx = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'] == duplicate]['tree_id_pred_nn_dist'].idxmax()

        # Append suggested removal to list
        ids_to_remove_pred_id.append(duplicate_idx)

    # Assign NaN to duplicate trees that did not win
    tree_locations_pred_df.loc[ids_to_remove_pred_id,'tree_id_pred'] = np.nan

    return ids_to_remove_pred_id, tree_locations_pred_df

def local_maxima_func(chm_clipped_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df, window_size, min_height=0, shape_fig_name=None, save_shape_file=False):
    
    """
	Function to run LM algorithm and determine scores
	
	Parameters:
	chm_clipped_path (str): Folder path to CHM
	tree_point_calc_shifted_csv_path (str): Folder path to ground truth (corrected) tree positions
    tree_pos_calc_df (dataframe): Dataframe of calculated tree points (used for tree labels only)
    tree_actual_df (dataframe): Dataframe of actual tree data from project sponsor
    window_size (int): min_distance value under consideration
    min_height: Minimum height used to filter low height artifacts (optional). Default=0
    shape_fig_name (str): Shapefile name (optional). Default=None
    save_shape_file (bool): Option to save shapefile of all predictions (optional). Default=None
	
	Returns:
    df_global_gp (gp dataframe): Geopandas dataframe of all tree positions estimated by LM, allocated trees, distances to nearest neighbours and heights
    tree_positions_from_lm (dataframe): Dataframe containing all tree positions (only coordinates)
    results_df_2 (dataframe): Results dataframe containing all scores
    tree_locations_pred_df (dataframe): Same as df_global_gp but including additional data after checking for duplicate allocations
	"""    
    
    # Load CHM for local maxima
    with rasterio.open(chm_clipped_path) as source:
        chm_img = source.read(1) # Read raster band 1 as a numpy array
        affine = source.transform

    # Load CHM for height sampling
    src = rasterio.open(chm_clipped_path)

    # Find local maxima using Scikit-image peak_local_max() function. 
    # https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max 
    # https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/peak.py#L119-L326
    coordinates = peak_local_max(chm_img, min_distance=window_size, threshold_abs=min_height)
    X=coordinates[:, 1]
    y=coordinates[:, 0]

    # Transform from local (image-based) coordinates to CHM CRS coordinates via affine set when opening CHM
    xs, ys = affine * (X, y)
    df_global = pd.DataFrame({'X':xs, 'Y':ys})
    df_local = pd.DataFrame({'X':X, 'Y':y})

    # Create geomtry type column for use with GeoPandas and shapefiles
    df_global['geometry'] = df_global.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)
    df_global_gp = gpd.GeoDataFrame(df_global, geometry='geometry')

    # Save shapefiles depending on options
    if save_shape_file == True:
        if shape_fig_name == None:
            shape_file_name = 'lm_shape_files/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_lm_tree_points_' + str(window_size) + '.shp'
        else:
            shape_file_name = 'lm_shape_files/' + shape_fig_name + '.shp'
        df_global_gp.to_file(shape_file_name, driver='ESRI Shapefile')

    # Sample CHM to obtain predicted heights
    df_global_gp = pred_heights(df_global_gp, raster=src)

    # Filter trees < min height
    df_global_gp = df_global_gp[df_global_gp['height_pred'] >= min_height].reset_index(drop=True)
    tree_positions_from_lm = df_global_gp[['X','Y']]

    # Allocate predictions to actual trees
    tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
    tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
    tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y
    tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')

    # Allocate estimated postions to actual positions
    for idx in range(len(df_global_gp)):

        current_tree = (df_global_gp.loc[idx, 'X'], df_global_gp.loc[idx, 'Y'])
        distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
        df_global_gp.loc[idx, 'tree_id_pred'] = tree_id
        df_global_gp.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
        df_global_gp.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

    # Find actual tree positions that have duplicates allocated
    ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(df_global_gp)

    # Merge with actual data to determine number of dead trees predicted
    tree_actual_df_no_dead = tree_actual_df[tree_actual_df['24_Def'] != 'D']
    tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')

    # Merge with actual data to determine number of dead trees predicted
   
    tree_locations_pred_df_no_unal = tree_locations_pred_df[tree_locations_pred_df['tree_id'].isna()==False]
    results_idx_2 = 0
    results_df_2 = pd.DataFrame()
    results_df_2.loc[results_idx_2, 'window_size'] = window_size
    results_df_2.loc[results_idx_2, 'number_trees_pred'] = tree_locations_pred_df.shape[0]
    results_df_2.loc[results_idx_2, 'number_unallocated'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]
    results_df_2.loc[results_idx_2, 'number_dead_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
    results_df_2.loc[results_idx_2, 'perc_trees_pred'] = tree_locations_pred_df.shape[0] / tree_actual_df_no_dead.shape[0]
    results_df_2.loc[results_idx_2, 'MAE_position'] = tree_locations_pred_df_no_unal['tree_id_pred_nn_dist'].mean()
    results_df_2.loc[results_idx_2, 'MSE_position'] = tree_locations_pred_df_no_unal['tree_id_pred_nn_dist_squared'].mean()
    results_df_2.loc[results_idx_2, 'max_dist'] = tree_locations_pred_df_no_unal[tree_locations_pred_df_no_unal['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].max()
    results_df_2.loc[results_idx_2, 'min_dist'] = tree_locations_pred_df_no_unal[tree_locations_pred_df_no_unal['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].min()
    results_df_2.loc[results_idx_2, 'MAE_height'] = np.abs(tree_locations_pred_df_no_unal['Hgt22Rod'] - tree_locations_pred_df_no_unal['height_pred']).mean()
    results_df_2.loc[results_idx_2, 'MSE_height'] = np.square(np.abs(tree_locations_pred_df_no_unal['Hgt22Rod'] - tree_locations_pred_df_no_unal['height_pred'])).mean()
    results_df_2.loc[results_idx_2, 'RMSE_height'] = np.sqrt(results_df_2.loc[results_idx_2, 'MSE_height'])
    results_df_2.loc[results_idx_2, 'R2'] = r2_score(tree_locations_pred_df_no_unal['Hgt22Rod'], tree_locations_pred_df_no_unal['height_pred'])
    results_df_2.loc[results_idx_2, 'max_height_pred'] = tree_locations_pred_df_no_unal['height_pred'].max()
    results_df_2.loc[results_idx_2, 'min_height_pred'] = tree_locations_pred_df_no_unal['height_pred'].min()

    return df_global_gp, tree_positions_from_lm, results_df_2, tree_locations_pred_df

def deep_forest_pred(ortho_name, ortho_path, ortho_clipped_root, tree_point_calc_csv_path, tree_point_calc_shifted_csv_path, tree_actual_df, tree_actual_df_no_dead, patch_size, patch_overlap=0.15, thresh, iou_threshold=0.5, shape_fig_name=None, save_fig = False, save_shape = False):

    """
	Function to run DF model and determine scores
	
	Parameters:
	ortho_name (str): File name of orthomosaic to be predicted on
	ortho_path (str): Folder path to orthomosaic to be predicted on (including ortho_name)
    ortho_clipped_root (str): Root directory of orthomosaic to be predicted on
	tree_point_calc_shifted_csv_path (str): Folder path to ground truth (corrected) tree positions
    tree_pos_calc_df (dataframe): Dataframe of calculated tree points (used for tree labels only)
    tree_actual_df (dataframe): Dataframe of actual tree data from project sponsor
    tree_actual_df_no_dead (dataframe): Same as tree_actual_df_full but with dead trees removed
    patch_size (int): Patch size to be used by DF model for predictions
    patch_overlap (float): Patch overlap (Default=0.15)
    iou_threshold (float): Minimum iou overlap among predictions between windows to be suppressed. Default=0.5
    thresh (float): Score threshold for filtering low-confidence predictions 
    shape_fig_name (str): shapefile and image name for writing. Default=None 
    save_fig (bool) = Option to save orthomosaic including predcited bounding boxes. Default=False
    save_shape = Option to save shapefile of bounding box centroids. Default=FalseFalse
	
	Returns:
    predictions_df (dataframe): Dataframe including all predictions and relevant data (in local image coordinates)
    predictions_df_transform (dataframe): Dataframe including all predictions and relevant data (in transformed orthomosaic CRS coordinates)
    results_df (dataframe): Results dataframe containing all scores
    predicted_raster_image (array): Orthomosaic including all predictions
	"""    

    # Open orthomosaic as Numpy array and record CRS and transform data
    with rasterio.open(ortho_path) as source:
        img = source.read() # Read raster bands as a numpy array
        transform_crs = source.transform
        crs = source.crs

    # Convert to RGB and BGR (OpenCV standard)
    img = img.astype('float32')
    img_rgb = np.moveaxis(img, 0, 2).copy()
    img_bgr = img_rgb[...,::-1].copy()

    # Read in dataframe of all trees
    tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

    # Read in dataframe of all trees (shifted)
    tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
    tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
    tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y
    tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')

    # Create results dataframe and set start index
    results_df = pd.DataFrame()
    results_idx = 0

    # Create predictions using DF .predict_tile() method (https://github.com/weecology/DeepForest/blob/main/deepforest/main.py)
    predictions_df = model.predict_tile(image=img_bgr, patch_size=patch_size, patch_overlap=patch_overlap, iou_threshold=iou_threshold)

    # Filter predictions using score threshold and print number of predictions remaining
    predictions_df = predictions_df[predictions_df['score'] > thresh]
    print(f"{predictions_df.shape[0]} predictions kept after applying threshold")

    # Create predictions image using DF 
    predicted_raster_image = plot_predictions_from_df(predictions_df, img_bgr)

    # Save image if option selected
    if save_fig == True: 
        if shape_fig_name == None:
            df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + '_thresh-' + str(thresh)  + '_iou-' + str(iou_threshold) + '.png'
        else:
            df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + shape_fig_name + '.png'
        plt.imsave(df_image_save_path,arr=predicted_raster_image)

    # Transform predictions to original CRS
    predictions_df_transform = predictions_df.copy()
    predictions_df_transform['image_path'] = ortho_name
    predictions_df_transform = predictions_df_transform[['xmin', 'ymin', 'xmax', 'ymax','image_path']]
    predictions_df_transform = utilities.project_boxes(predictions_df_transform, root_dir=ortho_clipped_root)
    predictions_df_transform[['xmin','ymin','xmax', 'ymax']] = predictions_df_transform.geometry.apply(lambda x: x.bounds).tolist()
    predictions_df_transform['centroid'] = predictions_df_transform['geometry'].centroid
    predictions_df_transform['X'] = gpd.GeoSeries(predictions_df_transform['centroid']).x
    predictions_df_transform['Y'] = gpd.GeoSeries(predictions_df_transform['centroid']).y
    predictions_df_transform['geometry'] = predictions_df_transform.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)

    # Save shapefile if option selected
    if save_shape == True:
        if shape_fig_name == None:
            shape_file_name = 'deepforest_predictions/shapefiles/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + 'thresh' + str(thresh)  + 'iou' + str(iou_threshold) + '.shp'
        else:
            shape_file_name = 'deepforest_predictions/shapefiles/' + shape_fig_name + '.shp'
        predictions_df_transform['geometry'].to_file(shape_file_name, driver='ESRI Shapefile')

    # For each predcition, find actual nearest neighbour and allocate to prediction
    for idx in range(len(predictions_df_transform)):

        current_tree = (predictions_df_transform.loc[idx, 'X'], predictions_df_transform.loc[idx, 'Y'])
        distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
        predictions_df_transform.loc[idx, 'tree_id_pred'] = tree_id
        predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
        predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared
    
    # Add scores to transformed df
    predictions_df_transform['score'] = predictions_df['score']

    # Allocate predictions to actual trees
    ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_df_transform)

    # Merge with actual data to determine number of dead trees predicted
    tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')

    # Save results
    results_df.loc[results_idx, 'patch_size'] = patch_size
    results_df.loc[results_idx, 'patch_overlap'] = patch_overlap
    results_df.loc[results_idx, 'thresh'] = thresh
    results_df.loc[results_idx, 'iou_threshold'] = iou_threshold
    results_df.loc[results_idx, 'number_trees_pred'] = tree_locations_pred_df.shape[0]
    results_df.loc[results_idx, 'number_unallocated'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]
    results_df.loc[results_idx, 'number_dead_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
    results_df.loc[results_idx, 'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
    results_df.loc[results_idx, 'MAE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist'].mean()
    results_df.loc[results_idx, 'MSE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist_squared'].mean()
    results_df.loc[results_idx, 'max_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].max()
    results_df.loc[results_idx, 'min_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].min()

    results_idx += 1

    return predictions_df, predictions_df_transform, results_df, predicted_raster_image


    ################################################

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

def tree_neighbour_stats(tree_pred_df):

    """
	Function to calculate tree neighbour statistics
	
	Parameters:
	tree_pred_df (dataframe): Dataframe containing all tree position predictions

	
	Returns:
    max_dist (float): maximum distance between any two trees
    min_dist (float): minimum distance between any two trees
    avg_dist (float): average distance between all two trees
    sd_dist (float): standard deviation of distance between all two trees
	""" 

    nn_list = []
    # Loop over all tree predictions
    for idx in range(len(tree_pred_df)):

        # Create arravy of all trees
        trees = np.asarray(tree_pred_df[['X', 'Y']])

        # Isolate tree under consideration
        current_tree = (trees[idx][0], trees[idx][1])
        current_tree = np.array(current_tree).reshape(-1, 1)

        # Calculate distance between tree in question and all other trees
        distances = euclidean_distances(current_tree.T, trees)

        # Remove distance from tree in question to itself (0)
        distances = np.delete(distances, np.where(distances == 0))

        # Determine closest tree
        nn_idx = distances.argmin()

        # Determine distance to closest tree
        distance_to_nn = distances.T[nn_idx]
        nn_list.append(distance_to_nn)

        # Store max, min, avg and std. dev.
        max_dist = max(nn_list)
        min_dist = min(nn_list)
        avg_dist = sum(nn_list) / len(nn_list)
        sd_dist = np.std(nn_list)

    return max_dist, min_dist, avg_dist, sd_dist

def tree_point_quality(tree_pred_df, max_dist, min_dist, dec_func, min_dist_coef = 1, max_dist_coef = 1):

    """
	Function to calculate tree position quality
	
	Parameters:
	tree_pred_df (dataframe): Dataframe containing all tree position predictions
    min_dist (float): minimum distance between any two trees
    max_dist (float): maximum distance between any two trees
    dec_func: The threshold is used to determine whether a tree position that has been deemed invalid by the minimum distance coefficient should be disregarded.
    min_dist_coef (float):  The coefficient by which the minimum nearest neighbour distance is multiplied to 
                            determine whether a tree position or its nearest neighbour, is an invalid tree position. Default = 1
    max_dist_coef (float):  The coefficient by which the maximum nearest neighbour distance is multiplied to 
                            determine whether a neighbouring tree position is, in fact,a legitimate neighbouring tree position. Default = 1

	Returns:
    tree_pred_df (dataframe): Dataframe containing all tree position predictions with ill-quality trees removed
    num_trees_removed (int): Number of trees removed
    perc_trees_removed (float): % of trees removed
	""" 

    # Store model name
    model = tree_pred_df.iloc[0]['model']

    # Determine number of trees before algorithm
    num_trees_before = tree_pred_df.shape[0]

    # Loop over all predictions
    for i, idx in enumerate(tree_pred_df.index):

        # Create array of all prediction
        trees = np.asarray(tree_pred_df[['X', 'Y']])

        # Isolate prediction under consideration
        current_tree = (trees[i][0], trees[i][1])
        current_tree = np.array(current_tree).reshape(-1, 1)

        # Calculate distances between prediction under consideration and all other predictions
        distances = euclidean_distances(current_tree.T, trees)

        # Remove prediction under consideration (distance=0)
        distances = np.delete(distances, np.where(distances == 0))

        # Sort distances
        distances.sort()

        # Isolate nearest 4 predictions
        distances = distances[0:4]

        # Filter predictions > max_dist*max_dist_coef
        distances = distances[distances <= (max_dist*max_dist_coef)]

        # Store minimum distance
        tree_pred_df.loc[idx,'nn_min'] = min(distances)

    # Filter predictions with min_nn >= min_dist*min_dist_coef and dec_func < dec_func
    tree_pred_df = tree_pred_df[~((tree_pred_df['nn_min'] < min_dist*min_dist_coef) & (tree_pred_df['dec_func'] < dec_func))]

    # Calculate number and percentage of trees removed
    num_trees_removed = num_trees_before - tree_pred_df.shape[0]
    perc_trees_removed = num_trees_removed / num_trees_before

    return tree_pred_df, num_trees_removed, perc_trees_removed

def tree_selection(all_trees_pred_after_removal_LM_2, all_trees_pred_after_removal_DF_2, dec_func_diff_thresh):

    """
	Function to determine which tree position to retain when an LM prediction falls within a DF bouding box
	
	Parameters:
	all_trees_pred_after_removal_LM_2 (dataframe): All remaining LM predictions
    all_trees_pred_after_removal_DF_2 (dataframe): All remaining DF predictions
    dec_func_diff_thresh:   The difference threshold is considered when an LM tree position falls within a DF bounding box
                            and must be compared. If the difference between the LM and DF tree position decision function
                            values is greater than this parameter, the tree position with the highest decision function value
                            is retained, otherwise, the DF tree position is retained.
	
	Returns:
    DF_predictions_final (dataframe): All remaining DF predictions after selection
    LM_predictions_final (dataframe): All remaining LM predictions after selection
    total_trees (int): Total number of trees remaining 
	""" 

    # Declare lists for positions to keep and remove
    df_idx_to_keep = []
    lm_idx_to_remove = []

    # Loop over all DF bounding boxes
    for idx_df in all_trees_pred_after_removal_DF_2.index:
        box = all_trees_pred_after_removal_DF_2.loc[idx_df,'geometry']
        point_in_box_flag = False

        # Loop ove all LM tree positions
        for idx_lm in all_trees_pred_after_removal_LM_2.index: 

            X_point = all_trees_pred_after_removal_LM_2.loc[idx_lm, 'X']
            Y_point = all_trees_pred_after_removal_LM_2.loc[idx_lm, 'Y']
            point = Point((float(X_point), float(Y_point)))
            
            # Check of LM point is in DF bounding box
            if box.contains(point) == True:
                point_in_box_flag = True

                # If difference in decision function values exceeds dec_func_diff_thresh, check which decision function value is greater
                if np.abs(all_trees_pred_after_removal_DF_2.loc[idx_df, 'dec_func'] - all_trees_pred_after_removal_LM_2.loc[idx_lm, 'dec_func']) > dec_func_diff_thresh:

                    if all_trees_pred_after_removal_DF_2.loc[idx_df, 'dec_func'] > all_trees_pred_after_removal_LM_2.loc[idx_lm, 'dec_func']:
                        df_idx_to_keep.append(idx_df)
                        lm_idx_to_remove.append(idx_lm)

                # If difference in decision function values does not exceed dec_func_diff_thresh, retain DF prediction
                else: 
                    df_idx_to_keep.append(idx_df)
                    lm_idx_to_remove.append(idx_lm)
        
        # If no point in box, retain DF prediction
        if point_in_box_flag == False: 
            df_idx_to_keep.append(idx_df)

    # Determine LM predictions to keep
    lm_idx_to_keep = [x for x in list(all_trees_pred_after_removal_LM_2.index) if x not in lm_idx_to_remove]

    # Remove duplicated boxes that contained more than one point
    df_idx_to_keep = list(set(df_idx_to_keep))

    # Remove unwanted points in DeepForest and Local Maxima predictions
    DF_predictions_final = all_trees_pred_after_removal_DF_2.loc[df_idx_to_keep]
    LM_predictions_final = all_trees_pred_after_removal_LM_2.loc[lm_idx_to_keep]
    total_trees = DF_predictions_final.shape[0] + LM_predictions_final.shape[0]

    return DF_predictions_final, LM_predictions_final, total_trees

def regression_scores(y_true, y_pred, model, features=None, feature_list_id=None):

    """
	Function to generate scores for a regression model output
	
	Parameters:
	y_true (array): Ground truth target variable 
	y_pred (array): Predicted target variable 
	model (str): Model utilised
    features (list): List of features used (for testing only). Default=None
    feature_list_id (int): Identifier for feature set (for testing only). Default=None
	
	Returns:
    score_df (dataframe): Dataframe with MAE, RMSE, MAPE, and R2
	"""    

    # Create dataframe to store scores
    score_df = pd.DataFrame(columns=['model', 'mae', 'rmse', 'mape', 'r2', 'features', 'feature_list_idx'])

    # Calculate scores 
    # https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics
    score_df['features'] = score_df['features'].astype(object)
    score_df.loc[0,'model'] = model
    score_df.loc[0,'mae'] = mean_absolute_error(y_true, y_pred)
    score_df.loc[0,'rmse'] = mean_squared_error(y_true, y_pred, squared=False)
    score_df.loc[0,'mape'] = mean_absolute_percentage_error(y_true, y_pred)
    score_df.loc[0,'r2'] = r2_score(y_true, y_pred)
    
    if features != None or feature_list_id != None:
        score_df.loc[0,'feature_list_idx'] = feature_list_id
        score_df.loc[0,'features'] = features

    return score_df

################################################
#            PREDICTION PIPELINE               #
################################################

# Import actual tree data from project sponsor 
tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(tree_actual_df_path, 'TEST')
    
# Clipped CHM path
chm_test_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")

# Clipped ortho file name, path and root dir
ortho_name = 'ortho_test_clip.tif'
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
ortho_clipped_path = ortho_clipped_root + '/' + ortho_name

# Load actual tree positions
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y

# Create dataframe to save results
final_results_df = pd.DataFrame(columns=['patch_size', 'patch_overlap', 'iou_threshold', 'thresh', 'min_distance', 'dec_func',\
                                         'min_dist_coef', 'dec_func_diff_thresh','total_trees_lm','total_trees_df', 'unalloc_lm', 'unalloc_df', 'dead_lm', \
                                         'dead_df','max_dist_coef', 'rem_class_lm', 'rem_class_df','rem_qual_lm', 'rem_qual_df',\
                                         'mae_pos_lm', 'mae_pos_df','mae_pos_final', 'mae_pos_perc_orig',\
                                         'mae_height_lm', 'mae_height_df','mae_height_final', 'mae_height_perc_orig',\
                                         'rmse_height_lm', 'rmse_height_df','rmse_height_final', 'rmse_height_perc_orig',\
                                         'r2_height_lm', 'r2_height_df','r2_height_final', 'r2_height_perc_orig', 'final_trees_pred', 'perc_trees_pred',\
                                         'final_dead', 'final_unalloc','duration'])

patch_size=850
window_size = 24
patch_overlap = 0.4
iou_threshold = 0.5
thresh = 0.3
dec_func = 0.5
dec_func_diff_thresh = 0.3
min_dist_coef = 1.4
max_dist_coef = 1
pipe_idx= 0 

# Start Timer
start = time.time()

final_results_df.loc[pipe_idx, 'patch_size'] = patch_size
final_results_df.loc[pipe_idx, 'patch_overlap'] = patch_overlap
final_results_df.loc[pipe_idx, 'iou_threshold'] = iou_threshold
final_results_df.loc[pipe_idx, 'thresh'] = thresh
final_results_df.loc[pipe_idx, 'min_distance'] = window_size
final_results_df.loc[pipe_idx, 'dec_func'] = dec_func
final_results_df.loc[pipe_idx, 'min_dist_coef'] = min_dist_coef
final_results_df.loc[pipe_idx, 'max_dist_coef'] = max_dist_coef
final_results_df.loc[pipe_idx, 'dec_func_diff_thresh'] = dec_func_diff_thresh

# Instantiate model and set to GPU
model = main.deepforest()
model.use_release()
model.to("cuda")
model.config["gpus"] = 1

# Set NMS threshold
model.config["nms_thresh"] = iou_threshold

# Load model
model_path = 'df_models/final_model.pt'
model.model.load_state_dict(torch.load(model_path))

# Get tree positions from DF
predictions_df, predictions_df_transform, results_df_df, predicted_raster_image = deep_forest_pred(ortho_name, ortho_clipped_path, ortho_clipped_root, tree_point_calc_csv_path=tree_point_calc_csv_path, tree_point_calc_shifted_csv_path=tree_point_calc_shifted_csv_path, tree_actual_df=tree_actual_df, tree_actual_df_no_dead = tree_actual_df_no_dead, patch_size=patch_size, patch_overlap=patch_overlap, thresh=thresh, iou_threshold=iou_threshold,shape_fig_name=test_name, save_fig = True, save_shape = True)

# Get tree positions from LM
df_global_gp, tree_positions_from_lm, results_df_2_lm, tree_locations_pred_df_lm = local_maxima_func(chm_test_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df, window_size=window_size, min_height=min_height, shape_fig_name=test_name, save_shape_file=True)

# Store scores in scores dataframe
total_trees_LM = results_df_2_lm.loc[0,'number_trees_pred']
total_trees_DF = results_df_df.loc[0,'number_trees_pred']
total_dead_trees_LM = results_df_2_lm.loc[0,'number_dead_pred']
total_dead_trees_DF = results_df_df.loc[0,'number_dead_pred']
total_unallocated_LM = results_df_2_lm.loc[0,'number_unallocated']
total_unallocated_DF = results_df_df.loc[0,'number_unallocated']
final_results_df.loc[pipe_idx,'total_trees_lm'] = total_trees_LM
final_results_df.loc[pipe_idx,'total_trees_df'] = total_trees_DF
final_results_df.loc[pipe_idx,'dead_lm'] = total_dead_trees_LM
final_results_df.loc[pipe_idx,'dead_df'] = total_dead_trees_DF
final_results_df.loc[pipe_idx,'unalloc_lm'] = total_unallocated_LM
final_results_df.loc[pipe_idx,'unalloc_df'] = total_unallocated_DF

# Create one dataframe of all predictions
df_global_gp[['xmin_box', 'ymin_box', 'xmax_box', 'ymax_box']] = 0
df_global_gp_red = df_global_gp[['X', 'Y', 'tree_id_pred','xmin_box', 'ymin_box', 'xmax_box', 'ymax_box', 'height_pred']]
df_global_gp_red.loc[:,'model'] = 'LM'
predictions_df_transform = predictions_df_transform.rename(columns={'xmin':'xmin_box', 'ymin':'ymin_box', 'xmax':'xmax_box', 'ymax':'ymax_box'})
predictions_df_transform_red = predictions_df_transform[['X', 'Y', 'tree_id_pred', 'xmin_box', 'ymin_box', 'xmax_box', 'ymax_box']]
predictions_df_transform_red.loc[:,'model'] = 'DF'
all_trees_pred = pd.concat([df_global_gp_red, predictions_df_transform_red]).reset_index(drop=True)

# Set expansion size
expansion_size = 25
expansion_size_metres = expansion_size*0.0317*1.1

# Generate boxes from tree positions
all_trees_pred_boxes = boxes_from_points(all_trees_pred, expansion_size_metres)

# Load orthomosaic, crop all tree positions and create dataset on which classifications are to be carried out on
ortho_cropped_for_cropping = rxr.open_rasterio(ortho_clipped_path, masked=True).squeeze()
crop_df = crop_pixels_to_df(ortho_cropped_for_cropping, all_trees_pred_boxes, expansion_size = expansion_size, save_crops = False, train=False)

# Run classifier and filter trees
model_path = 'tree_classifier_crops/saved_models/svm_v2.sav'
all_trees_pred_after_removal_LM, all_trees_pred_after_removal_DF, removed_LM, removed_DF = tree_classifier(all_trees_pred, crop_df, model_path)
final_results_df.loc[pipe_idx,'rem_class_lm'] = removed_LM.shape[0]
final_results_df.loc[pipe_idx,'rem_class_df'] = removed_DF.shape[0]

# Obtain tree neighbour stats for LM and DF predictions/estimations
max_dist_lm, min_dist_lm, avg_dist_lm, sd_dist_lm = tree_neighbour_stats(all_trees_pred_after_removal_LM)
max_dist_df, min_dist_df, avg_dist_df, sd_dist_df = tree_neighbour_stats(all_trees_pred_after_removal_DF)

# Obtain tree point quality and filter
all_trees_pred_after_removal_LM_2, num_trees_removed_LM, perc_trees_removed_LM = tree_point_quality(all_trees_pred_after_removal_LM, max_dist_lm, min_dist_lm, dec_func=dec_func, min_dist_coef=min_dist_coef, max_dist_coef=max_dist_coef)
all_trees_pred_after_removal_DF_2, num_trees_removed_DF, perc_trees_removed_DF = tree_point_quality(all_trees_pred_after_removal_DF, max_dist_df, min_dist_df, dec_func=dec_func, min_dist_coef=min_dist_coef, max_dist_coef=max_dist_coef)

# Store number of trees removed
final_results_df.loc[pipe_idx,'rem_qual_lm'] = num_trees_removed_LM
final_results_df.loc[pipe_idx,'rem_qual_df'] = num_trees_removed_LM

# Generate box polygon geometries
geoms = [] 
for idx in all_trees_pred_after_removal_DF_2.index:

    box_list = list(all_trees_pred_after_removal_DF_2.loc[idx,['xmin_box', 'ymin_box', 'xmax_box','ymax_box']])
    geom = Polygon([[box_list[0], box_list[1]], [box_list[0],box_list[3]], [box_list[2],box_list[3]], [box_list[2],box_list[1]]])
    geoms.append(geom)

all_trees_pred_after_removal_DF_2['geometry'] = geoms

# Select which tree positions to retain
DF_predictions_final, LM_predictions_final, total_trees = tree_selection(all_trees_pred_after_removal_LM_2, all_trees_pred_after_removal_DF_2, dec_func_diff_thresh = dec_func_diff_thresh)

# Create dataframe of all predictions
all_predictions_final = pd.concat([DF_predictions_final[['X', 'Y', 'tree_id_pred']], LM_predictions_final[['X', 'Y', 'tree_id_pred']]])
all_predictions_final_gp = all_predictions_final.copy()
all_predictions_final_gp['geometry'] = all_predictions_final_gp.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)
all_predictions_final_gp = gpd.GeoDataFrame(all_predictions_final_gp, geometry='geometry').reset_index(drop=True)

# Save final tree positions to a shapefile
shape_file_path= 'results/shape_files/' + test_name + 'final_tree_positions.shp'
all_predictions_final_gp.to_file(shape_file_path, driver='ESRI Shapefile')

# Assign trees to actual trees for evaluation 
nn_list_final = []
for idx in all_predictions_final_gp.index:
    current_tree = (all_predictions_final_gp.loc[idx, 'X'], all_predictions_final_gp.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    nn_list_final.append(distance_to_nn)

# Load test set CHM
chm_test = rxr.open_rasterio(chm_test_clip_path, masked=True).squeeze()

# Calculate the DF perimeter and area of bounding boxes, including CHM statistics for 
for idx in DF_predictions_final.index:

    width = np.abs(DF_predictions_final.loc[idx,'xmax_box'] - DF_predictions_final.loc[idx,'xmin_box'])
    height = np.abs(DF_predictions_final.loc[idx,'ymax_box'] - DF_predictions_final.loc[idx,'ymin_box'])
    perimeter = 2 * width + 2 * height
    area = width * height 
    DF_predictions_final.loc[idx,'perimeter'] = perimeter
    DF_predictions_final.loc[idx,'area'] = area

    geo = DF_predictions_final.loc[idx,'geometry']
    clip = chm_test.rio.clip([geo])

    DF_predictions_final.loc[idx,'avg_chm'] = clip.mean().values
    DF_predictions_final.loc[idx,'max_chm'] = clip.max().values
    DF_predictions_final.loc[idx,'min_chm'] = clip.min().values
    DF_predictions_final.loc[idx,'std_chm'] = clip.std().values

# Transform area from mitre to pixel values
DF_predictions_final['area'] = (np.sqrt(np.array(DF_predictions_final['area']))/0.0317)**2

# Feature names
features_names = ['area', 'avg_chm','max_chm', 'std_chm']

# Load DF scaler and scale crop data
scaler = pickle.load(open('height_predictor/saved_models/scaler.pkl', 'rb'))
X_scaled = pd.DataFrame()
X_scaled[features_names] = scaler.transform(DF_predictions_final[features_names])

# Load LM scaler
scaler_lm = pickle.load(open('height_predictor/saved_models/scaler_lm.pkl', 'rb'))
X_scaled_lm = scaler_lm.transform(np.array(LM_predictions_final['height_pred']).reshape(-1, 1))

# Load models
height_predictor = pickle.load(open('height_predictor/saved_models/nn_7.sav',"rb"))
height_predictor_lm = pickle.load(open('height_predictor/saved_models/nn_lm.sav',"rb"))

# Make height predictions and assign to dataframes
predictions_df = height_predictor.predict(X_scaled)
predictions_lm = height_predictor_lm.predict(X_scaled_lm)

DF_predictions_final_incl_actual = DF_predictions_final.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
DF_predictions_final_incl_actual['height_pred_model'] = predictions_df

LM_predictions_final_incl_actual = LM_predictions_final.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
LM_predictions_final_incl_actual['height_pred_model'] = predictions_lm

# Reduce dataframes retaining only necessary columns
columns_to_keep = ['X', 'Y', 'tree_id_pred', 'height_pred', 'height_pred_model','Hgt22Rod']
predictions_final = pd.concat([DF_predictions_final_incl_actual[columns_to_keep], LM_predictions_final_incl_actual[columns_to_keep]]).reset_index(drop=True)

# Create dataframes for scoring
DF_score_df = DF_predictions_final_incl_actual[(DF_predictions_final_incl_actual['Hgt22Rod'].isna()!=True) & (DF_predictions_final_incl_actual['height_pred_model'].isna()!=True)]
LM_score_df = LM_predictions_final_incl_actual[(LM_predictions_final_incl_actual['Hgt22Rod'].isna() != True) & (LM_predictions_final_incl_actual['height_pred_model'].isna() != True)]

# Concatenate all predictions for scoring
final_score_df = pd.concat([DF_score_df[columns_to_keep], LM_score_df[columns_to_keep]]).reset_index(drop=True)

# Score height predictions
score_df_DF = regression_scores(DF_score_df['Hgt22Rod'], DF_score_df['height_pred_model'], model='height_predictor')
score_df_final = regression_scores(final_score_df['Hgt22Rod'], final_score_df['height_pred_model'], model='height_predictor_lm')

# Calculate position MAE in relation to LM baseline scores
lm_mae_pos = 0.3515
df_mae_pos = np.round(results_df_df.loc[0,'MAE_position'],4)
final_mae_pos = np.round(sum(nn_list_final)/len(nn_list_final),4)
final_mae_pos_perc = np.round(((lm_mae_pos-final_mae_pos)/lm_mae_pos)*100,2)

# Calculate height MAE in relation to LM baseline scores
lm_mae_height = 0.4949
df_mae_height = np.round(score_df_DF.loc[0,'mae'],4)
final_mae_height = np.round(score_df_final.loc[0,'mae'],4)
final_mae_height_perc = np.round(((lm_mae_height-final_mae_height)/lm_mae_height)*100,2)

# Calculate height RMSE in relation to LM baseline scores
lm_rmse_height = 0.6435
df_rmse_height = np.round(score_df_DF.loc[0,'rmse'],4)
final_rmse_height = np.round(score_df_final.loc[0,'rmse'],4)
final_rmse_height_perc = np.round(((lm_rmse_height-final_rmse_height)/lm_rmse_height)*100,2)

# Calculate height R2 in relation to LM baseline scores
lm_r2_height = 0.6662
df_r2_height = np.round(score_df_DF.loc[0,'r2'],4)
final_r2_height = np.round(score_df_final.loc[0,'r2'],4)
final_r2_height_perc = np.round(((lm_r2_height-final_r2_height)/lm_r2_height)*100,2)

# Store results in results dataframe
final_results_df.loc[pipe_idx,'mae_pos_lm'] = lm_mae_pos
final_results_df.loc[pipe_idx,'mae_pos_df'] = df_mae_pos
final_results_df.loc[pipe_idx,'mae_pos_final'] = final_mae_pos
final_results_df.loc[pipe_idx,'mae_pos_perc_orig'] = final_mae_pos_perc

final_results_df.loc[pipe_idx,'mae_height_lm'] = lm_mae_height
final_results_df.loc[pipe_idx,'mae_height_df'] = df_mae_height
final_results_df.loc[pipe_idx,'mae_height_final'] = final_mae_height
final_results_df.loc[pipe_idx,'mae_height_perc_orig'] = final_mae_height_perc

final_results_df.loc[pipe_idx,'rmse_height_lm'] = lm_rmse_height
final_results_df.loc[pipe_idx,'rmse_height_df'] = df_rmse_height
final_results_df.loc[pipe_idx,'rmse_height_final'] = final_rmse_height
final_results_df.loc[pipe_idx,'rmse_height_perc_orig'] = final_rmse_height_perc

final_results_df.loc[pipe_idx,'r2_height_lm'] = lm_r2_height
final_results_df.loc[pipe_idx,'r2_height_df'] = df_r2_height
final_results_df.loc[pipe_idx,'r2_height_final'] = final_r2_height
final_results_df.loc[pipe_idx,'r2_height_perc_orig'] = final_rmse_height_perc

# Allocate predictions to actual trees
for idx in range(len(predictions_final)):

    current_tree = (predictions_final.loc[idx, 'X'], predictions_final.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    predictions_final.loc[idx, 'tree_id_pred'] = tree_id
    predictions_final.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
    predictions_final.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_final)

# Merge with actual data to and determine number of trees predicted including dead and unallocated trees
tree_actual_df_no_dead = tree_actual_df[tree_actual_df['24_Def'] != 'D']
tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
results_idx = 0
results_df = pd.DataFrame()

final_results_df.loc[pipe_idx,'final_trees_pred'] = tree_locations_pred_df.shape[0]
final_results_df.loc[pipe_idx,'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
final_results_df.loc[pipe_idx,'final_dead'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
final_results_df.loc[pipe_idx,'final_unalloc'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]

# Store time taken to run prediction pipeline
end = time.time()
final_results_df.loc[pipe_idx,'duration'] = round((end - start)/60, 2)

# Write final results to CSV
csv_file_path = 'final_results/' + test_name + '.csv'
final_results_df.to_csv(csv_file_path)