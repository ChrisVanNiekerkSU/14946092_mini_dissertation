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
import sys
import gc

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

def annotation_json_to_csv(folder_path, dataset, image_path=None):

    """
	Function to import CVAT annotations and transform into DataFrame
	
	Parameters:
	folder_path (str): Folder path to annotation zip files from CVAT
	dataset (str): Dataset in question ('TRAIN', 'VAL', or 'TEST')
    image_path (str): Image path to image(s) on which the annotations were made (optional). Default = None
	
	Returns:
    all_annotations_df (dataframe): : Geopandas dataframe of all tree positions estimated by LM, allocated trees, distances to nearest neighbours and heights
    annotations_csv_filename (str): File name of annotations csv file
    annotations_csv_filepath (str): File path of annotations csv file
	"""

    # Create dataframe to store data
    all_annotations_df = pd.DataFrame()

    # Create a list of all zip files in the specified folder
    zip_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    # Loop through all zip files
    for zip_file in zip_files:

        # Extract zip file
        zip_filename = zip_file
        extracted_folder_name = folder_path + zip_filename.replace('.zip','')
        zip_path = folder_path + zip_filename

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_name)
        
        if not [f for f in listdir(extracted_folder_name) if isfile(join(extracted_folder_name, f))]: continue
        
        # Extract JSON data from zip file
        with open(extracted_folder_name + '/annotations.json') as json_file:
            data = json.load(json_file)   

        # Write JSON data to dataframe
        annotations_df = pd.json_normalize(data[0]['shapes'])
        annotations_df = annotations_df[['points', 'label', 'frame']]
        annotations_df[['xmin', 'ymin', 'xmax', 'ymax']] = pd.DataFrame(annotations_df['points'].tolist())
        annotations_df['frame'] = annotations_df['frame'].apply(lambda x: str(x).zfill(2))
        
        # Create default image_path if not specified
        if image_path == None:
            annotations_df['image_path'] = dataset + '_' + annotations_df['frame'] + '.png'

        # Use specified image_path if given
        else: 
            annotations_df['image_path'] = image_path

        # Retain only necessary columns
        annotations_df = annotations_df[['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']]

        # Add current annotations to main annotations dataframe
        all_annotations_df = pd.concat([all_annotations_df, annotations_df], ignore_index=True)

    # Write all annotations to csv
    annotations_csv_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_annotations.csv'
    annotations_csv_filepath = folder_path + 'annotation csv files/' + annotations_csv_filename
    all_annotations_df.to_csv(annotations_csv_filepath, index=False)

    return all_annotations_df, annotations_csv_filename, annotations_csv_filepath

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

################################################
#         DF HEIGHT MODEL TRAINING             #
################################################

# Set folder path to train dataset and obtain all annotations from CVAT JSON zip files for train set
folder_path='df_crops_annotations/train/'
dataset='train'
annotations_df, annotations_csv_filename, annotations_csv_filepath = annotation_json_to_csv(folder_path, dataset)


# Set folder path to train dataset and obtain all annotations from CVAT JSON zip files for validation set
folder_path_val='df_crops_annotations/val/'
dataset_val='val'
annotations_df_val, annotations_csv_filename_val, annotations_csv_filepath_val = annotation_json_to_csv(folder_path_val, dataset_val, image_path = 'ortho_cropped/ortho_val_clip.tif')
annotations_df_val['image_path'] = 'ortho_val_clip.tif'
root_dir_val = folder_path_val + 'ortho_cropped'

# Set paths for orthomosaics (train, validation and test)
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
ortho_name_train = 'ortho_train_clip.tif'
ortho_clipped_path_train = ortho_clipped_root + '/' + ortho_name_train
ortho_for_cropping_train = rxr.open_rasterio(ortho_clipped_path_train, masked=True).squeeze()

ortho_name_val = 'ortho_val_clip.tif'
ortho_clipped_path_val = ortho_clipped_root + '/' + ortho_name_val
ortho_for_cropping_val = rxr.open_rasterio(ortho_clipped_path_val, masked=True).squeeze()

ortho_name_test = 'ortho_test_clip.tif'
ortho_clipped_path_test = ortho_clipped_root + '/' + ortho_name_test
ortho_for_cropping_test = rxr.open_rasterio(ortho_clipped_path_test, masked=True).squeeze()

# Open train orthomosaic as Numpy arrays and transform to BGR format
with rasterio.open(ortho_clipped_path_train) as source:
    img_train = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img_train = img_train.astype(np.uint8)
img_train_rgb = np.moveaxis(img_train, 0, 2).copy()
img_train_bgr = img_train_rgb[...,::-1].copy()

# Open validation orthomosaic as Numpy arrays and transform to BGR format
with rasterio.open(ortho_clipped_path_val) as source:
    img_val = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img_val = img_val.astype(np.uint8)
img_val_rgb = np.moveaxis(img_val, 0, 2).copy()
img_val_bgr = img_val_rgb[...,::-1].copy()

# Open test orthomosaic as Numpy arrays and transform to BGR format
with rasterio.open(ortho_clipped_path_test) as source:
    img_test = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img_test = img_test.astype(np.uint8)
img_test_rgb = np.moveaxis(img_test, 0, 2).copy()
img_test_bgr = img_test_rgb[...,::-1].copy()

# Join annotations to actual tree data as per project sponsor data
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y
tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')

# Reformat image_path details
annotations_df['ortho_path'] = annotations_df['image_path']
annotations_df['image_path'] = annotations_df['image_path'].replace('.png', '.tif', regex=True)
annotations_df['image_path'] = annotations_df['image_path'].replace('df_crops_annotations/train/crops/', '', regex=True)

# Declare root directory of train set orthomosaic
ortho_clipped_tranform_root = 'df_crops_annotations/train/cropped_ortho'

# Create dataframe for transformed annotations
annotations_df_transform = pd.DataFrame(columns = annotations_df.columns)

# Transform training annotations from image coordinates to CRS coordinates using DF project_boxes() method
# https://github.com/weecology/DeepForest/blob/main/deepforest/utilities.py
for image_path in annotations_df['image_path'].unique():

    working_df = annotations_df[annotations_df['image_path'] == image_path]
    working_df_transform = utilities.project_boxes(working_df, root_dir=ortho_clipped_tranform_root)

    annotations_df_transform = pd.concat([annotations_df_transform, working_df_transform])

# Convert to GeoPandas dataframe
annotations_df_transform_gp = gpd.GeoDataFrame(annotations_df_transform, geometry='geometry').reset_index(drop=True)
# shape_file_path= 'df_crops_annotations/train/annotation shape files/'  + datetime.datetime.now().strftime("%Y%m%d_%H%M") + 'training_annotations.shp'
# annotations_df_transform_gp.to_file(shape_file_path, driver='ESRI Shapefile')

# Load training CHM
chm_train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
chm_train = rxr.open_rasterio(chm_train_clip_path, masked=True)

# Calculate the perimeter and area of bounding boxes and CHM stats - TRAIN
for i,idx in enumerate(annotations_df_transform.index):

    width = np.abs(annotations_df_transform.loc[idx,'xmax'] - annotations_df_transform.loc[idx,'xmin'])
    height = np.abs(annotations_df_transform.loc[idx,'ymax'] - annotations_df_transform.loc[idx,'ymin'])
    perimeter = 2 * width + 2 * height
    area = width * height 
    annotations_df_transform.loc[idx,'perimeter'] = perimeter
    annotations_df_transform.loc[idx,'area'] = area

    geom = annotations_df_transform.loc[idx,'geometry']
    geom_df = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[geom]) 
    clip = chm_train.rio.clip(geom_df.geometry)

    annotations_df_transform.loc[idx,'avg_chm'] = clip.mean().values
    annotations_df_transform.loc[idx,'max_chm'] = clip.max().values
    annotations_df_transform.loc[idx,'min_chm'] = clip.min().values
    annotations_df_transform.loc[idx,'std_chm'] = clip.std().values

    if idx % 100 == 0: 
        print(idx)

# Calculate bounding box centroids - TRAIN
annotations_df_transform['X'] = annotations_df_transform['xmin'] + (annotations_df_transform['xmax'] - annotations_df_transform['xmin'])/2
annotations_df_transform['Y'] = annotations_df_transform['ymin'] + (annotations_df_transform['ymax'] - annotations_df_transform['ymin'])/2

# Transform validation annotations from image coordinates to CRS coordinates using DF project_boxes() method
# https://github.com/weecology/DeepForest/blob/main/deepforest/utilities.py
annotations_df_val_transform = utilities.project_boxes(annotations_df_val, root_dir=root_dir_val)

# Load validation CHM
chm_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_val_clip.tif")
chm_val = rxr.open_rasterio(chm_val_clip_path, masked=True)

# Calculate the perimeter and area of bounding boxes and CHM stats - VAL
for i,idx in enumerate(annotations_df_val_transform.index):

    width = np.abs(annotations_df_val_transform.loc[idx,'xmax'] - annotations_df_val_transform.loc[idx,'xmin'])
    height = np.abs(annotations_df_val_transform.loc[idx,'ymax'] - annotations_df_val_transform.loc[idx,'ymin'])
    perimeter = 2 * width + 2 * height
    area = width * height 
    annotations_df_val_transform.loc[idx,'perimeter'] = perimeter
    annotations_df_val_transform.loc[idx,'area'] = area

    geom = annotations_df_val_transform.loc[idx,'geometry']
    geom_df = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[geom]) 
    clip = chm_val.rio.clip(geom_df.geometry)

    annotations_df_val_transform.loc[idx,'avg_chm'] = clip.mean().values
    annotations_df_val_transform.loc[idx,'max_chm'] = clip.max().values
    annotations_df_val_transform.loc[idx,'min_chm'] = clip.min().values
    annotations_df_val_transform.loc[idx,'std_chm'] = clip.std().values

    if idx % 100 == 0: 
        print(idx)

# Calculate bounding box centroids - VAL
annotations_df_val_transform['X'] = annotations_df_val_transform['xmin'] + (annotations_df_val_transform['xmax'] - annotations_df_val_transform['xmin'])/2
annotations_df_val_transform['Y'] = annotations_df_val_transform['ymin'] + (annotations_df_val_transform['ymax'] - annotations_df_val_transform['ymin'])/2

# Convert to GeoPandas dataframe
annotations_df_val_transform_gp = gpd.GeoDataFrame(annotations_df_val_transform, geometry='geometry').reset_index(drop=True)

# Convert to GeoPandas dataframe
annotations_df_val_transform_gp = gpd.GeoDataFrame(annotations_df_val_transform, geometry='geometry').reset_index(drop=True)
annotations_df_transform_gp = gpd.GeoDataFrame(annotations_df_transform, geometry='geometry').reset_index(drop=True)

# Calculate centroids for train and validation sets
annotations_df_transform['centroid'] = annotations_df_transform_gp['geometry'].centroid
annotations_df_val_transform['centroid'] = annotations_df_val_transform_gp['geometry'].centroid
annotations_df_val_transform['X'] = gpd.GeoSeries(annotations_df_val_transform['centroid']).x
annotations_df_val_transform['Y'] = gpd.GeoSeries(annotations_df_val_transform['centroid']).y
annotations_df_transform['X'] = gpd.GeoSeries(annotations_df_transform['centroid']).x
annotations_df_transform['Y'] = gpd.GeoSeries(annotations_df_transform['centroid']).y

# Assign training annotations to actual trees as per data from project sponsor
for idx in range(len(annotations_df_transform)):

    current_tree = (annotations_df_transform.loc[idx, 'X'], annotations_df_transform.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    annotations_df_transform.loc[idx, 'tree_id_pred'] = tree_id
    annotations_df_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
    annotations_df_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

# Assign validation annotations to actual trees as per data from project sponsor
for idx in range(len(annotations_df_val_transform)):

    current_tree = (annotations_df_val_transform.loc[idx, 'X'], annotations_df_val_transform.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    annotations_df_val_transform.loc[idx, 'tree_id_pred'] = tree_id
    annotations_df_val_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
    annotations_df_val_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

# Remove training annotations not assigned to trees (unclassified)
remove_list_train = []
for tree_id in annotations_df_transform['tree_id_pred'].unique():

    working_df = annotations_df_transform[annotations_df_transform['tree_id_pred'] == tree_id].sort_values(by='tree_id_pred_nn_dist', ascending=True)

    if working_df.shape[0] == 1: continue

    else:
        remove_idxs = list(working_df.iloc[1::].index)
        for idx in remove_idxs:
            remove_list_train.append(idx)
    
annotations_df_transform_cleaned_train = annotations_df_transform[~annotations_df_transform.index.isin(remove_list_train)]

# Remove validation annotations not assigned to trees (unclassified)
remove_list_val = []
for tree_id in annotations_df_val_transform['tree_id_pred'].unique():

    working_df = annotations_df_val_transform[annotations_df_val_transform['tree_id_pred'] == tree_id].sort_values(by='tree_id_pred_nn_dist', ascending=True)

    if working_df.shape[0] == 1: continue

    else:
        remove_idxs = list(working_df.iloc[1::].index)
        for idx in remove_idxs:
            remove_list_val.append(idx)
    
annotations_df_val_transform_cleaned_val = annotations_df_val_transform[~annotations_df_val_transform.index.isin(remove_list_val)]

# Import actual tree data (training) from project sponsor
tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df_train, tree_actual_df_full_train, tree_actual_df_no_dead_train, min_height_train = actual_tree_data(tree_actual_df_path, 'TRAIN')

# Convert data to numeric
tree_actual_df_ground_train = tree_actual_df_train.copy()
tree_actual_df_ground_train['Hgt22Rod'] = pd.to_numeric(tree_actual_df_ground_train['Hgt22Rod'], errors='coerce')
tree_actual_df_ground_train['Hgt22Drone'] = pd.to_numeric(tree_actual_df_ground_train['Hgt22Drone'], errors='coerce')
tree_actual_df_ground_train = tree_actual_df_ground_train[(tree_actual_df_ground_train['Hgt22Drone'].isna() == False) & (tree_actual_df_ground_train['Hgt22Rod'].isna() == False)]

# Merge annotations and actual data
annotations_df_transform_cleaned_actual_train = annotations_df_transform_cleaned_train.merge(tree_actual_df_ground_train[['tree_id', 'Hgt22Rod']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
annotations_df_transform_cleaned_actual_train = annotations_df_transform_cleaned_actual_train[annotations_df_transform_cleaned_actual_train['tree_id'].isna() == False].reset_index(drop=True)

# Import actual tree data (validation) from project sponsor
tree_actual_df_val, tree_actual_df_full_val, tree_actual_df_no_dead_val, min_height_val = actual_tree_data(tree_actual_df_path, 'VAL')

# Convert data to numeric
tree_actual_df_ground_val = tree_actual_df_val.copy()
tree_actual_df_ground_val['Hgt22Rod'] = pd.to_numeric(tree_actual_df_ground_val['Hgt22Rod'], errors='coerce')
tree_actual_df_ground_val['Hgt22Drone'] = pd.to_numeric(tree_actual_df_ground_val['Hgt22Drone'], errors='coerce')
tree_actual_df_ground_val = tree_actual_df_ground_val[(tree_actual_df_ground_val['Hgt22Drone'].isna() == False) & (tree_actual_df_ground_val['Hgt22Rod'].isna() == False)]

# Merge annotations and actual data
annotations_df_transform_cleaned_actual_val = annotations_df_val_transform_cleaned_val.merge(tree_actual_df_ground_val[['tree_id', 'Hgt22Rod']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
annotations_df_transform_cleaned_actual_val = annotations_df_transform_cleaned_actual_val[annotations_df_transform_cleaned_actual_val['tree_id'].isna() == False].reset_index(drop=True)

# Determine features to be used
features_names = ['area', 'avg_chm','max_chm', 'std_chm']

# Create training dataset for model training and assign dataset identifier
X_train = annotations_df_transform_cleaned_actual_train[features_names]
X_train['set'] = 'train'
y_train = annotations_df_transform_cleaned_actual_train['Hgt22Rod']

# Create validation dataset for model training and assign dataset identifier
X_val = annotations_df_transform_cleaned_actual_val[features_names]
X_val['set'] = 'val'
y_val = annotations_df_transform_cleaned_actual_val['Hgt22Rod']

# Concatenate datasets to creat scaler
X = pd.concat([X_train, X_val]).reset_index(drop=True)

# Fit minmax scaler
scaler = MinMaxScaler()
X_scaled = pd.DataFrame()
X_scaled[features_names] = scaler.fit_transform(X[features_names])

# Save scaler for use with test set
pickle.dump(scaler, open('height_predictor/saved_models/scaler.pkl', 'wb'))

# Reassign dataset identifier for splitting the dataset
X_scaled['set'] = X['set']

# Split dataset after scaler fitting - train
x_train = X_scaled[X_scaled['set'] == 'train']
x_train = x_train.drop('set', axis=1).reset_index(drop=True)

# Split dataset after scaler fitting - validation
x_val = X_scaled[X_scaled['set'] == 'val']
x_val = x_val.drop('set', axis=1).reset_index(drop=True)

# Create dataframe for scoring
nn_model_search = pd.DataFrame(columns=['model_no', 'mae', 'rmse', 'mape', 'r2'])

# Train 50 models and save each
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
# https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/neural_network/_multilayer_perceptron.py#L1257
for i in range(50):
    
    # Train MLP
    nn_reg = MLPRegressor(hidden_layer_sizes=(20, 133, 205), solver='adam', activation='relu',learning_rate='invscaling', max_iter=3000, early_stopping=True).fit(x_train, np.ravel(y_train))
    
    # Generate predictions on validation set 
    predictions_nn_reg = nn_reg.predict(x_val)

    # Calculate scores for validation set
    score_df_nn = regression_scores(y_true=y_val, y_pred=predictions_nn_reg, model='NN')

    # Store scores for model in dataframe
    nn_model_search.loc[i,'model_no'] = i
    nn_model_search.loc[i,['mae', 'rmse', 'mape', 'r2']] = score_df_nn.loc[0,['mae', 'rmse', 'mape', 'r2']]

    # Save model
    nn_model_filename = 'height_predictor/saved_models/nn_' + str(i) + '.sav'
    pickle.dump(nn_reg, open(nn_model_filename, 'wb'))

# Write scores to CSV
nn_model_search.to_csv('height_predictor/scores/nn_height_model_search.csv')

################################################
#           LM HEIGHT MODEL TESTING            #
################################################

# Run LM algorithm on all dataset CHMs
df_global_gp_train, tree_positions_from_lm_train, results_df_lm_train, results_df_2_lm_train, tree_locations_pred_df_lm_train = local_maxima_func(chm_train_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_train, window_size=24, min_height=0, save_shape_file=False)
df_global_gp_val, tree_positions_from_lm_val, results_df_lm_val, results_df_2_lm_val, tree_locations_pred_df_lm_val = local_maxima_func(chm_val_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_val, window_size=24, min_height=0, save_shape_file=False)
df_global_gp_test, tree_positions_from_lm_test, results_df_lm_test, results_df_2_lm_test, tree_locations_pred_df_lm_test = local_maxima_func(chm_test_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_test, window_size=24, min_height=0, save_shape_file=False)

# Remove estimated tree positions without assigned actual trees (no actual data available fro training)
train_df = tree_locations_pred_df_lm_train[tree_locations_pred_df_lm_train['Hgt22Rod'].isna()!=True][['height_pred', 'Hgt22Rod']]
test_df = tree_locations_pred_df_lm_test[tree_locations_pred_df_lm_test['Hgt22Rod'].isna()!=True][['height_pred', 'Hgt22Rod']]

# Create target variables
y_lm = np.array(train_df['Hgt22Rod'])
x_lm = np.array(train_df['height_pred']).reshape(-1, 1)

# Fit scaler 
scaler = MinMaxScaler()
x_lm_scaled = scaler.fit_transform(x_lm)

# Save scaler
pickle.dump(scaler, open('height_predictor/saved_models/scaler_lm.pkl', 'wb'))

# Create test data and scale
x_test_scaled = scaler.transform(np.array(test_df['height_pred']).reshape(-1, 1))

# Train model
nn_reg = MLPRegressor(hidden_layer_sizes=(20, 133, 205), solver='adam', activation='relu',learning_rate='invscaling', max_iter=3000, early_stopping=True).fit(x_lm_scaled, np.ravel(y_lm))

# Generate predictions
nn_predictions = nn_reg.predict(x_test_scaled)

# Calculate scores
nn_result = regression_scores(np.array(test_df['Hgt22Rod']), nn_predictions,'nn')

# Save model
mlp_lm_model_filename = 'height_predictor/saved_models/nn_lm.sav'
pickle.dump(nn_reg, open('height_predictor/saved_models/nn_lm.sav', 'wb'))

#####################################################################
#    MODEL TESTING AND PARAMETER SEARCH (FOR INFORMATION ONLY!!)    #
#           INCOMPLETE IN CURRENT STATE DUE TO TESTING              #
#####################################################################

# Create dataframe for scoring
scores_df_all_tests = pd.DataFrame(columns=['mae', 'rmse','mape', 'r2', 'model'])
test_counter = 0

for i in range(20):

    rf_reg = RandomForestRegressor(n_estimators = 100, max_depth=None).fit(x_train, np.ravel(y_train))
    ab_reg = AdaBoostRegressor(n_estimators = 50, learning_rate = 1,loss='linear').fit(x_train, np.ravel(y_train))
    svm_reg = svm.SVR(kernel='rbf', degree=5, C=1).fit(x_train, np.ravel(y_train))
    xgb_reg = xgb.XGBRegressor(n_estimators = 100, objective="reg:squarederror", random_state=42, use_label_encoder=False).fit(x_train, np.ravel(y_train))
    nn_reg = MLPRegressor(hidden_layer_sizes=(50,200,200),activation ='relu', solver = 'adam', alpha = 0.0001, learning_rate = 'constant', max_iter=1000).fit(x_train, np.ravel(y_train))

    # Make predictions
    predictions_rf = rf_reg.predict(x_val)
    predictions_ab = ab_reg.predict(x_val)
    predictions_svm = svm_reg.predict(x_val)
    predictions_xgb = xgb_reg.predict(x_val)
    predictions_nn = nn_reg.predict(x_val)

    score_df_rf = regression_scores(y_true=y_val, y_pred=predictions_rf, model='Random Forest')
    score_df_ab = regression_scores(y_true=y_val, y_pred=predictions_ab, model='AdaBoost')
    score_df_svm = regression_scores(y_true=y_val, y_pred=predictions_svm, model='SVM')
    score_df_xgb = regression_scores(y_true=y_val, y_pred=predictions_xgb, model='XGBoost')
    score_df_nn = regression_scores(y_true=y_val, y_pred=predictions_nn, model='NN')

    scores_df_all_models = pd.concat([score_df_rf, score_df_ab, score_df_xgb, score_df_svm, score_df_nn]).reset_index(drop=True)
    # scores_df_all_models.loc[df_idx,'feature_list'] = features_test

    scores_df_all_tests = pd.concat([scores_df_all_tests, scores_df_all_models]).reset_index(drop=True)

    test_counter += 1

    if test_counter % 200 == 0:

        print(test_counter, ' tests of ', feature_lists_df.shape[0]*10*4, 'completed') 
    
scores_df_all_tests[['mae', 'rmse','mape', 'r2']] = scores_df_all_tests[['mae', 'rmse','mape', 'r2']].astype(float)
scores_df_all_tests_red = scores_df_all_tests[['mae', 'rmse','mape', 'r2', 'model']]
scores_df_all_tests_red = scores_df_all_tests_red.reset_index(drop=True)
scores_df_all_tests_avg = scores_df_all_tests_red.groupby(['model'], as_index=False).mean()

from sklearn.inspection import permutation_importance
r = permutation_importance(nn_reg, x_val, y_val,n_repeats=30, random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{features_names[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

############### NN Grid Search ###############
# solvers = ['adam', 'lbfgs', 'sgd']
# learning_rates = ['constant', 'invscaling', 'adaptive']
solver = 'adam'
learning_rate = 'invscaling'
hl1s = 20
hl2s = 133
hl3s = 205
activations = ['identity', 'logistic', 'tanh', 'relu']

# max_iters = [100, 200, 300, 500, 800, 1000, 2000]
# alphas = [0.0001, 0.005, 0.001, 0.05, 0.01]

grid_search_idx = 0
nn_parameter_search = pd.DataFrame(columns=['activation', 'mae', 'rmse', 'mape', 'r2'])
# nn_parameter_search['arch'] = nn_parameter_search['arch'].astype(object)

# for hl1 in hl1s:
#     for hl2 in hl2s:
#         for hl3 in hl3s:
#             for solver in solvers:
for activation in activations:
    hidden_layer_sizes = (hl1s, hl2s, hl3s)
    nn_reg = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver=solver, activation=activation, learning_rate=learning_rate, max_iter=10000, random_state = 42, early_stopping =False).fit(x_train, np.ravel(y_train))
    predictions_nn_reg = nn_reg.predict(x_val)
    score_df_nn = regression_scores(y_true=y_val, y_pred=predictions_nn_reg, model='NN')

    nn_parameter_search.loc[grid_search_idx,'activation'] = activation
    # nn_parameter_search.loc[grid_search_idx,'solver'] = solver
    # nn_parameter_search.loc[grid_search_idx,'learning_rate'] = learning_rate
    # nn_parameter_search.loc[grid_search_idx,'alpha'] = alpha
    # nn_parameter_search.loc[grid_search_idx,'activation'] = activation
    # nn_parameter_search.loc[grid_search_idx,'early_stopping'] = early_stopping
    # nn_parameter_search.loc[grid_search_idx,'max_iter'] = max_iter
    nn_parameter_search.loc[grid_search_idx,['mae', 'rmse', 'mape', 'r2']] = score_df_nn.loc[0,['mae', 'rmse', 'mape', 'r2']]

    grid_search_idx += 1

    if grid_search_idx % 100 == 0: 
        print(grid_search_idx, " of ", len(hl1s)*len(hl2s)*len(hl3s), "tests completed")

nn_parameter_search.to_csv('height_predictor/scores/nn_height_model_grid_search_final_v2.csv')