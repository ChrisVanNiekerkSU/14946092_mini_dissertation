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
from shapely.geometry import Point

import json
from os import listdir
from os.path import isfile, join
import zipfile

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

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

###############################################
#      LM GRID SEARCH FOR WINDOW SIZE         #
###############################################

# Set file path to merged train and validation CHM and tree points (used for tree labels only)
chm_train_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_val_clip.tif")
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

# Import actual tree data from project sponsor (train & validation data)
tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df_val, tree_actual_df_full_val, tree_actual_df_no_dead_val, min_height_val = actual_tree_data(tree_actual_df_path, 'VAL')
tree_actual_df_train, tree_actual_df_full_train, tree_actual_df_no_dead_train, min_height_train = actual_tree_data(tree_actual_df_path, 'TRAIN')
tree_actual_df_train_val = pd.concat([tree_actual_df_train, tree_actual_df_val])

# Import corrected tree positions
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)

# Define dataset
dataset = "train_val"

# Set window sizes (min_distance) from 10 to 50 and create results dataframe
window_sizes = range(10,51)
results_grid_search = pd.DataFrame(columns=['window_size', 'number_trees_pred', 'number_unallocated','number_dead_pred', 'perc_trees_pred', 'MAE_position', 'MSE_position','max_dist', 'min_dist', 'MAE_height', 'MSE_height', 'RMSE_height','R2','max_height_pred', 'min_height_pred'])

# Test window sizes (min_distance) from 10 to 50
for i, window_size in enumerate(window_sizes):

    df_global_gp_train_val, tree_positions_from_lm_train_val, results_df_lm_train_val,results_df_lm_train_val_2, tree_locations_pred_df = local_maxima_func(chm_train_val_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_train_val, window_size=window_size, min_height=0, save_shape_file=True)
    results_grid_search = pd.concat([results_grid_search, results_df_lm_train_val_2])
    if i % 10 == 0: print(i, "tests of ", len(window_sizes), " completed")
        
results_grid_search.to_csv('height_predictor/window_size_test_results_v4' + dataset + '.csv', index=False)

# Check how selected window size performs on the test set
chm_test_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df_test, tree_actual_df_full_test, tree_actual_df_no_dead_test, min_height_test = actual_tree_data(tree_actual_df_path, 'TEST')

tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)

dataset = "test"
results_grid_search = pd.DataFrame(columns=['window_size', 'number_trees_pred', 'number_unallocated','number_dead_pred', 'perc_trees_pred', 'MAE_position', 'MSE_position','max_dist', 'min_dist', 'MAE_height', 'MSE_height', 'RMSE_height','R2','max_height_pred', 'min_height_pred'])

df_global_gp_test, tree_positions_from_lm_test,results_df_lm_test_2, tree_locations_pred_df = local_maxima_func(chm_test_val_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_test, window_size=24, min_height=0, save_shape_file=True)

tree_locations_pred_df.to_csv('../Main/images/working/lm_initial results.csv')