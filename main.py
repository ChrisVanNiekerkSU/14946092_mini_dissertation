# %%
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
# from PIL import Image
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
# from pympler.tracker import SummaryTracker
# from pympler import muppy, summary

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

pd.set_option('display.float_format', lambda x: '%.5f' % x)
# %%
def outer_trees(tree_pos_calc_df):

    # Identify outermost points
    south_side = pd.DataFrame(columns=['tree_id', 'X', 'Y'])
    north_side = pd.DataFrame(columns=['tree_id', 'X', 'Y'])
    west_side = pd.DataFrame(columns=['tree_id', 'X', 'Y'])
    east_side = pd.DataFrame(columns=['tree_id', 'X', 'Y'])
    for row in tree_pos_calc_df['Row'].unique():

        if row == tree_pos_calc_df['Row'].min():
            south_side = tree_pos_calc_df[tree_pos_calc_df['Row'] == row]
            # south_side = pd.concat([outerpoints_df, all_trees_in_row[['tree_id', 'tree_easting','tree_northing']]])
        elif row == tree_pos_calc_df['Row'].max():
            north_side = tree_pos_calc_df[tree_pos_calc_df['Row'] == row]
            north_side = north_side.iloc[::-1].reset_index(drop=True)
        else:
            all_trees_in_row = tree_pos_calc_df[tree_pos_calc_df['Row'] == row]

            east_plot = all_trees_in_row['Plot'].min()
            west_plot = all_trees_in_row['Plot'].max()

            east_tree = all_trees_in_row[(all_trees_in_row['Plot'] == east_plot) & (all_trees_in_row['Tree no'] == 1)][['tree_id', 'X','Y']]
            west_tree = all_trees_in_row[(all_trees_in_row['Plot'] == west_plot) & (all_trees_in_row['Tree no'] == 6)][['tree_id', 'X','Y']]
            west_side = pd.concat([west_side, west_tree])
            east_side = pd.concat([east_side, east_tree])
            
    east_side = east_side.iloc[::-1].reset_index(drop=True)
    outerpoints_df = pd.concat([south_side, west_side, north_side, east_side]).reset_index(drop=True)
    outerpoints_df['geometry'] = outerpoints_df.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)
    outerpoints_gp = gpd.GeoDataFrame(outerpoints_df, geometry='geometry')

    return outerpoints_gp, outerpoints_df

def tree_buffer_area(outer_trees_df, buffer_dist):

    # Create list of Easting and Northing points
    easting_point_list = list(outer_trees_df['geometry'].x) 
    northing_point_list = list(outer_trees_df['geometry'].y)

    # Create polygon from corner points and convert to gpd dataframe
    corner_trees_poly = Polygon(zip(easting_point_list, northing_point_list))
    corner_trees_poly_df = gpd.GeoDataFrame(index=[0], geometry=[corner_trees_poly])

    # Buffer polygon and convert to gpd dataframe
    corner_trees_poly_buffered = corner_trees_poly.buffer(buffer_dist) 

    buffered_poly_df = gpd.GeoDataFrame(index=[0], geometry=[corner_trees_poly_buffered])

    return corner_trees_poly_df, buffered_poly_df

def annotation_json_to_csv(folder_path, dataset, image_path=None):

    all_annotations_df = pd.DataFrame()
    zip_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    # print(zip_files)

    for zip_file in zip_files:

        zip_filename = zip_file
        extracted_folder_name = folder_path + zip_filename.replace('.zip','')
        zip_path = folder_path + zip_filename

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_name)
        
        if not [f for f in listdir(extracted_folder_name) if isfile(join(extracted_folder_name, f))]: continue
        
        with open(extracted_folder_name + '/annotations.json') as json_file:
            data = json.load(json_file)

        # img_name_list = [f for f in listdir(extracted_folder_name + '/data') if isfile(join(extracted_folder_name + '/data', f))]

        # for file in img_name_list:

        #     if '.png' in file: image_name = file      

        annotations_df = pd.json_normalize(data[0]['shapes'])
        annotations_df = annotations_df[['points', 'label', 'frame']]
        annotations_df[['xmin', 'ymin', 'xmax', 'ymax']] = pd.DataFrame(annotations_df['points'].tolist())
        annotations_df['frame'] = annotations_df['frame'].apply(lambda x: str(x).zfill(2))
        if image_path == None:
            # annotations_df['image_path'] = folder_path + 'crops/' + dataset + '_' + annotations_df['frame'] + '.png'
            annotations_df['image_path'] = dataset + '_' + annotations_df['frame'] + '.png'
        else: 
            annotations_df['image_path'] = image_path
        # image_path = test_df['col2'] = 'test_' + test_df['col1'].apply(lambda x: str(x).zfill(2)) + '.png'
        # annotations_df['image_path'] = image_path
        annotations_df = annotations_df[['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']]

        all_annotations_df = pd.concat([all_annotations_df, annotations_df], ignore_index=True)

    annotations_csv_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_annotations.csv'
    annotations_csv_filepath = folder_path + 'annotation csv files/' + annotations_csv_filename
    # annotations_csv_filepath = 'annotations/annotation csv files/' + annotations_csv_filename

    all_annotations_df.to_csv(annotations_csv_filepath, index=False)

    return all_annotations_df, annotations_csv_filename, annotations_csv_filepath

def plot_predictions_from_df(df, img, colour = (255, 255, 0)):

    # Draw predictions on BGR 
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

    trees = np.asarray(tree_locations_df[['X', 'Y']])
    # distances = cdist([current_tree], trees, 'euclidean').T
    tree = np.array(tree).reshape(-1, 1)
    distances = euclidean_distances(tree.T, trees)
    nn_idx = distances.argmin()
    distance_to_nn = distances.T[nn_idx][0]
    distance_to_nn_squared = (distances.T[nn_idx][0])**2
    tree_id = tree_locations_df.loc[nn_idx, 'tree_id']

    return distance_to_nn, distance_to_nn_squared, tree_id

def find_duplicates(tree_locations_pred_df):

    duplicates_series = tree_locations_pred_df.duplicated(subset=['tree_id_pred'], keep=False)
    duplicates_list_idx = list(duplicates_series[duplicates_series == True].index)
    duplicates_list = list(tree_locations_pred_df.loc[duplicates_list_idx, 'tree_id_pred'])
    duplicates_list = list(set(duplicates_list))

    ids_to_remove_pred_id = []
    for duplicate in duplicates_list:

        duplicate_idx = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'] == duplicate]['tree_id_pred_nn_dist'].idxmax()
        ids_to_remove_pred_id.append(duplicate_idx)

    tree_locations_pred_df.loc[ids_to_remove_pred_id,'tree_id_pred'] = np.nan

    return ids_to_remove_pred_id, tree_locations_pred_df

def actual_tree_data(tree_actual_df_path, sheet_name):

    # tree_actual_df_path = 'data/EG0181T Riverdale A9b MAIN.xlsx'
    tree_actual_df_full = pd.read_excel(open(tree_actual_df_path, 'rb'), sheet_name=sheet_name)
    last_valid_entry = tree_actual_df_full['Plot'].last_valid_index()
    tree_actual_df_full = tree_actual_df_full.loc[0:last_valid_entry]
    tree_actual_df_full = tree_actual_df_full.astype({'Plot':'int','Rep':'int','Tree no':'int'})
    tree_actual_df_full['tree_id'] = tree_actual_df_full['Plot'].astype('str') + '_' + tree_actual_df_full['Tree no'].astype('str')
    tree_actual_df_full['Hgt22Rod'] = pd.to_numeric(tree_actual_df_full['Hgt22Rod'], errors='coerce').fillna(0)
    tree_actual_df = tree_actual_df_full[['tree_id', 'Plot', 'Rep', 'Tree no', 'Hgt22Rod','24_Def', 'Hgt22Drone']]
    tree_actual_df_no_dead = tree_actual_df[tree_actual_df['24_Def'] != 'D']
    # min_height = tree_actual_df_no_dead['Hgt22Rod'].mean() - sigma * tree_actual_df_no_dead['Hgt22Rod'].std()
    min_height = tree_actual_df_no_dead['Hgt22Rod'].min()

    return tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height

def pred_heights(df, raster):

    # Sample the raster at every point location and store values in DataFrame
    pts = df[['X', 'Y', 'geometry']]
    pts.index = range(len(pts))
    coords = [(x,y) for x, y in zip(pts.X, pts.Y)]
    df['height_pred'] = [x[0] for x in raster.sample(coords)]

    return df

def local_maxima_func(chm_clipped_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df, window_size, min_height, shape_fig_name=None, save_shape_file=False, grid_search=False):
    # Load CHM for local maxima
    with rasterio.open(chm_clipped_path) as source:
        chm_img = source.read(1) # Read raster band 1 as a numpy array
        affine = source.transform

    # Load CHM for height sampling
    src = rasterio.open(chm_clipped_path)

    coordinates = peak_local_max(chm_img, min_distance=window_size, threshold_abs=min_height)
    X=coordinates[:, 1]
    y=coordinates[:, 0]
    xs, ys = affine * (X, y)
    df_global = pd.DataFrame({'X':xs, 'Y':ys})
    df_local = pd.DataFrame({'X':X, 'Y':y})
    df_global['geometry'] = df_global.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)
    df_global_gp = gpd.GeoDataFrame(df_global, geometry='geometry')

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

    for idx in range(len(df_global_gp)):

        current_tree = (df_global_gp.loc[idx, 'X'], df_global_gp.loc[idx, 'Y'])
        distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
        df_global_gp.loc[idx, 'tree_id_pred'] = tree_id
        df_global_gp.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
        df_global_gp.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

    ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(df_global_gp)

    # Merge with actual data to determine number of dead trees predicted
    tree_actual_df_no_dead = tree_actual_df[tree_actual_df['24_Def'] != 'D']
    tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
    # results_idx = 0
    # results_df = pd.DataFrame()
    # results_df.loc[results_idx, 'window_size'] = window_size
    # results_df.loc[results_idx, 'number_trees_pred'] = tree_locations_pred_df.shape[0]
    # results_df.loc[results_idx, 'number_unallocated'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]
    # results_df.loc[results_idx, 'number_dead_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
    # results_df.loc[results_idx, 'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
    # results_df.loc[results_idx, 'MAE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist'].mean()
    # results_df.loc[results_idx, 'MSE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist_squared'].mean()
    # results_df.loc[results_idx, 'max_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].max()
    # results_df.loc[results_idx, 'min_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].min()
    # results_df.loc[results_idx, 'MAE_height'] = np.abs(tree_locations_pred_df['Hgt22Rod'] - tree_locations_pred_df['height_pred']).mean()
    # results_df.loc[results_idx, 'MSE_height'] = np.square(np.abs(tree_locations_pred_df['Hgt22Rod'] - tree_locations_pred_df['height_pred'])).mean()
    # results_df.loc[results_idx, 'RMSE_height'] = np.sqrt(results_df.loc[results_idx, 'MSE_height'])
    # results_df.loc[results_idx, 'R2'] = r2_score(y_true, y_pred)
    # results_df.loc[results_idx, 'max_height_pred'] = tree_locations_pred_df['height_pred'].max()
    # results_df.loc[results_idx, 'min_height_pred'] = tree_locations_pred_df['height_pred'].min()

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

def deep_forest_pred(ortho_name, ortho_path, ortho_clipped_root, tree_point_calc_csv_path, tree_point_calc_shifted_csv_path, tree_actual_df, tree_actual_df_no_dead, patch_size, patch_overlap, thresh, iou_threshold, shape_fig_name=None, save_fig = False, save_shape = False):

    with rasterio.open(ortho_path) as source:
        img = source.read() # Read raster bands as a numpy array
        transform_crs = source.transform
        crs = source.crs

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

    results_df = pd.DataFrame()
    results_idx = 0
    # Create Predictions
    predictions_df = model.predict_tile(image=img_bgr, patch_size=patch_size, patch_overlap=patch_overlap, iou_threshold=iou_threshold)
    predictions_df = predictions_df[predictions_df['score'] > thresh]
    print(f"{predictions_df.shape[0]} predictions kept after applying threshold")

    predicted_raster_image = plot_predictions_from_df(predictions_df, img_bgr)

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

    if save_shape == True:
        # predictions_df_transform_for_shape = predictions_df_transform
        # predictions_df_transform_for_shape['geometry'] = predictions_df_transform_for_shape.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)
        if shape_fig_name == None:
            shape_file_name = 'deepforest_predictions/shapefiles/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + 'thresh' + str(thresh)  + 'iou' + str(iou_threshold) + '.shp'
        else:
            shape_file_name = 'deepforest_predictions/shapefiles/' + shape_fig_name + '.shp'
        predictions_df_transform['geometry'].to_file(shape_file_name, driver='ESRI Shapefile')

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

def get_average_box_size(predictions_df_transform):

    # Ensure ymax and xmax are greater than ymin and xmin
    if predictions_df_transform['xmax'].mean() < predictions_df_transform['xmin'].mean():
        predictions_df_transform = predictions_df_transform.rename(columns={'xmin': 'xmax', 'xmax': 'xmin'})
    if predictions_df_transform['ymax'].mean() < predictions_df_transform['ymin'].mean():
        predictions_df_transform = predictions_df_transform.rename(columns={'ymin': 'ymax', 'ymax': 'ymin'})

    # Calculate average box size
    predictions_df_transform['width'] = predictions_df_transform['xmax'] - predictions_df_transform['xmin']
    predictions_df_transform['height'] = predictions_df_transform['ymax'] - predictions_df_transform['ymin']
    average_width = predictions_df_transform['width'].mean()
    average_height =  predictions_df_transform['height'].mean()
    average_side = (average_width + average_height) / 2

    return average_width, average_height, average_side

def boxes_from_points(tree_positions_for_classification_all, expansion_size):

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

def dead_tree_classifier_dataset(predictions_df_transform, tree_actual_df, tree_positions_from_lm, chm_clipped_path, window_size=29, save_crops=False):

    # Filter DeepForest datatframe for tree points only
    tree_positions_from_df = predictions_df_transform[['X', 'Y']]

    # Isolate dead trees
    predictions_df_transform_incl_act = predictions_df_transform.merge(tree_actual_df, left_on='tree_id_pred', right_on='tree_id', how='left')
    predictions_df_transform_dead = predictions_df_transform_incl_act[predictions_df_transform_incl_act['24_Def'] == 'D'].reset_index(drop=True)
    tree_positions_from_df_dead_filtered = predictions_df_transform_dead[['X', 'Y']]
    # tree_positions_from_df_dead_filtered = tree_positions_from_df_dead_filtered.rename(columns={'X': 'X', 'Y': 'Y'})

    # Concatenate LocalMaxima and DeepForest outputs into one dataset
    tree_positions_all = pd.concat([tree_positions_from_lm, tree_positions_from_df]).reset_index(drop=True)

    # Randomly select trees from all trees (2 * number of dead trees)
    random_trees = np.random.randint(tree_positions_all.shape[0], size=tree_positions_from_df_dead_filtered.shape[0]*2)
    tree_positions_for_classification = tree_positions_all.loc[random_trees]

    # Create dataset
    tree_positions_for_classification_all = pd.concat([tree_positions_from_df_dead_filtered, tree_positions_for_classification]).reset_index(drop=True)

    # Expand points into boxes
    tree_positions_for_classification_all = boxes_from_points(tree_positions_for_classification_all, expansion_size)

    if save_crops == True:
        # Load ortho
        ortho_cropped_for_cropping = rxr.open_rasterio(ortho_clipped_path, masked=True).squeeze()

        crop_df = crop_array(ortho_cropped_for_cropping, tree_positions_for_classification_all, save_crops = save_crops)
        crop_df.to_csv('tree_classifier_crops/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") +'_tree_crops_rgb.csv')

        return tree_positions_for_classification_all, crop_df

    else: 
        return tree_positions_for_classification_all, None

def dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df, save_crops=False):

    # Filter DeepForest datatframe for tree points only
    tree_positions_from_df = tree_point_calc_shifted_df[['X', 'Y', 'tree_id']]

    # Isolate dead trees
    tree_positions_from_df = tree_positions_from_df.merge(tree_actual_df, on='tree_id', how='inner')

    tree_positions_from_df_not_dead = tree_positions_from_df[tree_positions_from_df['24_Def'] != 'D'].reset_index(drop=True)
    tree_positions_from_df_not_dead_filtered = tree_positions_from_df_not_dead[['X', 'Y']]
    tree_positions_from_df_not_dead_filtered['class'] = 1

    tree_positions_from_df_dead = tree_positions_from_df[tree_positions_from_df['24_Def'] == 'D'].reset_index(drop=True)
    tree_positions_from_df_dead_filtered = tree_positions_from_df_dead[['X', 'Y']]
    tree_positions_from_df_dead_filtered['class'] = 0

    if save_crops == True:
        # Load ortho
        ortho_cropped_for_cropping = rxr.open_rasterio(ortho_clipped_path, masked=True).squeeze()

        crop_df = crop_array(ortho_cropped_for_cropping, tree_positions_from_df_dead_filtered, save_crops = save_crops)
        crop_df.to_csv('tree_classifier_crops/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") +'_tree_crops_rgb.csv')

        return tree_positions_from_df_dead_filtered, tree_positions_from_df_not_dead_filtered, crop_df

    else: 
        return tree_positions_from_df_dead_filtered, tree_positions_from_df_not_dead_filtered

def crop_array(ortho_cropped_for_cropping, tree_positions_for_classification_all, save_crops = False):
    
    # Create crops
    crop_df = pd.DataFrame()
    idx_counter = 0 
    for idx in tree_positions_for_classification_all.index:

        geo = tree_positions_for_classification_all.loc[idx,'geometry']
        clip = ortho_cropped_for_cropping.rio.clip([geo])
        clip = np.array(clip)
        clip = clip.astype('uint8')
        clip_rgb = np.moveaxis(clip, 0, 2).copy()
        crop_df.loc[idx,'r_avg'] = clip_rgb[:,:,0].mean()
        crop_df.loc[idx,'g_avg'] = clip_rgb[:,:,1].mean()
        crop_df.loc[idx,'b_avg'] = clip_rgb[:,:,2].mean()
        crop_df.loc[idx,'r_sd'] = clip_rgb[:,:,0].std()
        crop_df.loc[idx,'g_sd'] = clip_rgb[:,:,1].std()
        crop_df.loc[idx,'b_sd'] = clip_rgb[:,:,2].std()

        if save_crops == True:
            tree_id_for_save = tree_positions_expanded_for_classification_all.loc[idx,'tree_id_pred']
            model_for_save = tree_positions_expanded_for_classification_all.loc[0,'model']
            tree_class_path = 'tree_classifier_crops/' + str(tree_id_for_save) + '_' + str(model_for_save) + '.png'
            plt.imsave(tree_class_path,arr=clip_rgb)

        if  (idx != 0) and (idx % 1000 == 0): 
            idx_counter += 1
            print(str(idx_counter*1000))

    return crop_df

def crop_pixels_to_df(ortho_cropped_for_cropping, classification_all, expansion_size = 36, save_crops = False, train=True):

    # images_df = pd.DataFrame()
    images_array = np.empty((classification_all.shape[0], expansion_size*expansion_size*3))
    # print(images_array.shape)
    # Create crops
    crop_df = pd.DataFrame()
    col_counter = 0 
    for idx in classification_all.index:

        geo = classification_all.loc[idx,'geometry']
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
        clip_rgb_flat = clip_rgb_cropped.flatten().T
        images_array[idx] = clip_rgb_flat

        if save_crops == True:
            tree_id_for_save = classification_all.loc[idx,'tree_id_pred']
            model_for_save = classification_all.loc[0,'model']
            tree_class_path = 'tree_classifier_crops/' + str(tree_id_for_save) + '_' + str(model_for_save) + '.png'
            plt.imsave(tree_class_path,arr=clip_rgb)

        # if  (idx != 0) and (idx % 1000 == 0): 
        #     print(str(idx))

    crop_df = pd.DataFrame(data=images_array)
    if train == True:
        crop_df['class'] = classification_all['class']

    return crop_df

def classification_scores(y_true, y_pred, y_pred_prob, model, features=None, feature_list_id=None):

    score_df = pd.DataFrame(columns=['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'features', 'feature_list_idx'])

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

def regression_scores(y_true, y_pred, model, features=None, feature_list_id=None):

    score_df = pd.DataFrame(columns=['model', 'mae', 'rmse', 'mape', 'r2', 'features', 'feature_list_idx'])

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

def tree_classifier(all_trees_pred, crop_df, model_path, lm_dec_func=None, df_dec_func=None):

    # Get feature names
    features_names = crop_df.columns
    # Scale features to match model input
    # scaler = MinMaxScaler()
    # scaler.fit(X)
    X_scaled_all_trees = crop_df / 255
    # X_scaled_all_trees[features_names] = scaler.fit_transform(crop_df[features_names])

    # Load tree classifier model
    tree_classifier = pickle.load(open(model_path,"rb"))

    # Generate predictions, probabilities and decision function values
    tree_classifications = tree_classifier.predict(X_scaled_all_trees)
    tree_classifications_prob_dead = tree_classifier.predict_proba(X_scaled_all_trees)
    tree_classifications_dec_func = tree_classifier.decision_function(X_scaled_all_trees)
    scaler = MinMaxScaler()
    tree_classifications_dec_func = scaler.fit_transform(tree_classifications_dec_func.reshape(-1, 1))


    all_trees_pred_incl_rgb = pd.concat([all_trees_pred,crop_df],axis=1)
    all_trees_pred_incl_rgb['class'] = tree_classifications
    all_trees_pred_incl_rgb['class_prob_dead'] = tree_classifications_prob_dead[:, 0]
    all_trees_pred_incl_rgb['dec_func'] = tree_classifications_dec_func

    # dec_func_range = all_trees_pred_incl_rgb['dec_func'].max() - all_trees_pred_incl_rgb['dec_func'].min()

    # all_trees_pred_after_removal_LM = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] > lm_dec_func) & (all_trees_pred_incl_rgb['model'] == 'LM')]
    # all_trees_pred_after_removal_DF = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] > df_dec_func) & (all_trees_pred_incl_rgb['model'] == 'DF')]
    # removed_LM = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] <= lm_dec_func) & (all_trees_pred_incl_rgb['model'] == 'LM')]
    # removed_DF = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] <= df_dec_func) & (all_trees_pred_incl_rgb['model'] == 'DF')]

    all_trees_pred_after_removal_LM = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['class'] != 0) & (all_trees_pred_incl_rgb['dec_func'] >= 0.35) & (all_trees_pred_incl_rgb['model'] == 'LM')]
    all_trees_pred_after_removal_DF = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['class'] != 0) & (all_trees_pred_incl_rgb['dec_func'] >= 0.35) & (all_trees_pred_incl_rgb['model'] == 'DF')]
    removed_LM = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['class'] == 0) & (all_trees_pred_incl_rgb['model'] == 'LM')]
    removed_DF = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['class'] == 0) & (all_trees_pred_incl_rgb['model'] == 'DF')]

    num_trees_removed_lm = df_global_gp_red.shape[0] - all_trees_pred_after_removal_LM.shape[0]
    num_trees_removed_df = predictions_df_transform_red.shape[0] - all_trees_pred_after_removal_DF.shape[0]
    # print("Number of trees removed for LM:     ", num_trees_removed_lm)
    # print("Percentage of trees removed for LM: ", np.round((num_trees_removed_lm / df_global_gp_red.shape[0])*100,2),'%')
    # print("Number of trees removed for DF:     ", num_trees_removed_df)
    # print("Percentage of trees removed for DF: ", np.round((num_trees_removed_df / predictions_df_transform_red.shape[0])*100,2),'%')

    return all_trees_pred_after_removal_LM, all_trees_pred_after_removal_DF, removed_LM, removed_DF

def tree_neighbour_stats(tree_pred_df):

    nn_list = []
    for idx in range(len(tree_pred_df)):

        trees = np.asarray(tree_pred_df[['X', 'Y']])
        current_tree = (trees[idx][0], trees[idx][1])
        current_tree = np.array(current_tree).reshape(-1, 1)
        distances = euclidean_distances(current_tree.T, trees)
        distances = np.delete(distances, np.where(distances == 0))
        nn_idx = distances.argmin()
        distance_to_nn = distances.T[nn_idx]
        nn_list.append(distance_to_nn)

        max_dist = max(nn_list)
        min_dist = min(nn_list)
        avg_dist = sum(nn_list) / len(nn_list)
        sd_dist = np.std(nn_list)

    return max_dist, min_dist, avg_dist, sd_dist

def tree_point_quality(tree_pred_df, max_dist, min_dist, dec_func, min_dist_coef = 1, max_dist_coef = 1):

    model = tree_pred_df.iloc[0]['model']
    num_trees_before = tree_pred_df.shape[0]
    for i, idx in enumerate(tree_pred_df.index):

        trees = np.asarray(tree_pred_df[['X', 'Y']])
        current_tree = (trees[i][0], trees[i][1])
        current_tree = np.array(current_tree).reshape(-1, 1)
        distances = euclidean_distances(current_tree.T, trees)
        distances = np.delete(distances, np.where(distances == 0))
        distances.sort()
        distances = distances[0:4]
        distances = distances[distances <= (max_dist*max_dist_coef)]
        tree_pred_df.loc[idx,'nn_min'] = min(distances)

    tree_pred_df = tree_pred_df[~((tree_pred_df['nn_min'] < min_dist*min_dist_coef) & (tree_pred_df['dec_func'] < dec_func))]


    num_trees_removed = num_trees_before - tree_pred_df.shape[0]
    perc_trees_removed = num_trees_removed / num_trees_before
    # print("Number of trees removed for ", model,":         ", num_trees_removed)
    # print("Percentage of trees removed for ", model,":     ", np.round(perc_trees_removed*100,2), '%')

    return tree_pred_df, num_trees_removed, perc_trees_removed

def train_test_plots(tree_actual_df, start_plot, num_rows):

    plot_list = []
    for i in range(12): 

        end_plot = start_plot + num_rows

        plot_list_int = list(range(start_plot,end_plot))
        for plot in plot_list_int:
            plot_list.append(plot)

        start_plot = start_plot + 69

    tree_actual_df_red = tree_actual_df[tree_actual_df['Plot'].isin(plot_list)]

    return tree_actual_df_red

def tree_selection(all_trees_pred_after_removal_LM_2, all_trees_pred_after_removal_DF_2, dec_func_diff_thresh):
    df_idx_to_keep = []
    lm_idx_to_remove = []

    for idx_df in all_trees_pred_after_removal_DF_2.index:
        box = all_trees_pred_after_removal_DF_2.loc[idx_df,'geometry']
        point_in_box_flag = False
        for idx_lm in all_trees_pred_after_removal_LM_2.index: 

            X_point = all_trees_pred_after_removal_LM_2.loc[idx_lm, 'X']
            Y_point = all_trees_pred_after_removal_LM_2.loc[idx_lm, 'Y']
            point = Point((float(X_point), float(Y_point)))

            if box.contains(point) == True:
                point_in_box_flag = True
                if np.abs(all_trees_pred_after_removal_DF_2.loc[idx_df, 'dec_func'] - all_trees_pred_after_removal_LM_2.loc[idx_lm, 'dec_func']) > dec_func_diff_thresh:

                    if all_trees_pred_after_removal_DF_2.loc[idx_df, 'dec_func'] > all_trees_pred_after_removal_LM_2.loc[idx_lm, 'dec_func']:
                        # print('here 2')
                        df_idx_to_keep.append(idx_df)
                        lm_idx_to_remove.append(idx_lm)
                
                    # else: 
                    #     # print('here 3')
                    #     df_idx_to_keep.append(idx_df)
                    #     lm_idx_to_remove.append(idx_lm)
                        # print('here 2')
                else: 
                    df_idx_to_keep.append(idx_df)
                    lm_idx_to_remove.append(idx_lm)
        
        if point_in_box_flag == False: 
            df_idx_to_keep.append(idx_df)
                    # print('here 3')
        # print('----------------')

    # Determine Local Maxima predictions to keep
    lm_idx_to_keep = [x for x in list(all_trees_pred_after_removal_LM_2.index) if x not in lm_idx_to_remove]

    # Remove duplicated boxes that contained more than one point
    df_idx_to_keep = list(set(df_idx_to_keep))

    # Remove unwanted points in DeepForest and Local Maxima predictions
    DF_predictions_final = all_trees_pred_after_removal_DF_2.loc[df_idx_to_keep]
    LM_predictions_final = all_trees_pred_after_removal_LM_2.loc[lm_idx_to_keep]
    total_trees = DF_predictions_final.shape[0] + LM_predictions_final.shape[0]

    return DF_predictions_final, LM_predictions_final, total_trees, df_idx_to_keep, lm_idx_to_keep

# %%
folder_path_train='df_crops_annotations/train/'
dataset_train='train'
annotations_df, annotations_csv_filename, annotations_csv_filepath = annotation_json_to_csv(folder_path_train, dataset_train)
folder_path_val='df_crops_annotations/val/'
dataset_val='val'
image_path = folder_path_val  + 'ortho_cropped/' + 'ortho_val_clip.tif'
annotations_df_val, annotations_csv_filename_val, annotations_csv_filepath_val = annotation_json_to_csv(folder_path_val, dataset_val, image_path='ortho_val_clip.tif')
root_dir_val_ortho = 'df_crops_annotations/val/ortho_cropped'
# %%
################################################
#             TRAIN DEEP FOREST                #
################################################
# annotations_cropped_csv_filepath_val = 'df_crops_annotations/val/ortho_cropped/split_annotations_val/ortho_val_clip.csv'
root_dir_val_crops = 'df_crops_annotations/val/ortho_cropped/split_annotations_val'
model = main.deepforest()
model.use_release()
# print("Current device is {}".format(model.device))
model.to("cuda")
# print("Current device is {}".format(model.device))
model.config["gpus"] = 1
# Set up model training
# root_dir = 
model.config["train"]["epochs"] = 5
model.config["workers"] = 0
# model.config["GPUS"] = 1
model.config["fast_dev_run"] = False
model.config["batch_size"]= 7
model.config["score_thresh"] = 0.5
model.config["train"]["optimiser"] = 'orig'
model.config["train"]["lr_schedule"] = 'orig'
model.config["train"]["lr"] = 0.005
model.config["train"]["patience"] = 5
model.config["train"]["factor"] = 0.05

model.config["train"]["csv_file"] = annotations_csv_filepath
model.config["train"]["root_dir"] = 'df_crops_annotations/train/crops'
# model.config["validation"]["csv_file"] = annotations_csv_filepath_val
# model.config["validation"]["root_dir"] = root_dir_val_ortho

# model.config["batch_size"]= 1
name = 'val_final'

logger = CSVLogger("logs", name=name)
model.create_trainer(logger=logger)
model.trainer.fit(model)
# %%

model_path = 'df_models/final_model.pt'
model.model.load_state_dict(torch.load(model_path))

folder_path_train='df_crops_annotations/train/'
dataset_train='train'
annotations_df, annotations_csv_filename, annotations_csv_filepath = annotation_json_to_csv(folder_path_train, dataset_train)

score_threshs = [0.3]
iou_thresholds = [0.4]
results_idx = 0
results_df = pd.DataFrame(columns=['dataset','trees_matched','box_precision', 'box_recall', 'box_f1', 'class_precision', 'class_recall', 'miou'])
for score_thresh in score_threshs:
    for iou_thresh in iou_thresholds:
        start =time.time()
        # Evaluate model on validation data
        
        model.config["score_thresh"] = score_thresh
        save_dir = 'results/df_validation_results'
        annotations_cropped_csv_filepath_val = 'df_crops_annotations/val/ortho_cropped/split_annotations_val/ortho_val_clip.csv'
        root_dir_val_crops = 'df_crops_annotations/val/ortho_cropped/split_annotations_val'
        results = model.evaluate(annotations_cropped_csv_filepath_val, root_dir=root_dir_val_crops, iou_threshold = iou_thresh, savedir = save_dir)

        results_df.loc[results_idx, 'box_precision'] = results['box_precision']
        results_df.loc[results_idx, 'box_recall'] = results['box_recall']
        results_df.loc[results_idx, 'class_recall'] = results['class_recall']['recall'][0]
        results_df.loc[results_idx, 'class_precision'] = results['class_recall']['precision'][0]
        results_df.loc[results_idx, 'miou'] = results['results']['IoU'].mean()
        results_df.loc[results_idx, 'dataset'] = 'validation'
        results_df.loc[results_idx, 'iou_thresh'] = iou_thresh
        results_df.loc[results_idx, 'score_thresh'] = score_thresh

        if results_df.loc[results_idx, 'box_recall'] + results_df.loc[results_idx, 'box_precision'] == 0: results_df.loc[results_idx, 'box_f1'] = 0
        else: results_df.loc[results_idx, 'box_f1'] = 2*((results_df.loc[results_idx, 'box_recall']*results_df.loc[results_idx, 'box_precision'])/(results_df.loc[results_idx, 'box_recall']+results_df.loc[results_idx, 'box_precision']))

        results_df.loc[results_idx, 'trees_matched'] = results['results']['match'].sum()
        # results_df.to_csv('df_models/df_val_eval_scores_v2.csv')
        results_idx += 1
        print('val done')

        # Evaluate model on train data
        # results_df = pd.DataFrame(columns=['trees_matched','box_precision', 'box_recall', 'box_f1', 'miou'])
        # model.config["score_thresh"] = 0.75
        save_dir = 'results/df_training_results'
        annotations_cropped_csv_filepath_val = 'df_crops_annotations/val/ortho_cropped/split_annotations_val/ortho_val_clip.csv'
        root_dir_val_crops = 'df_crops_annotations/val/ortho_cropped/split_annotations_val'
        results = model.evaluate(annotations_csv_filepath, root_dir='df_crops_annotations/train/crops', iou_threshold = iou_thresh, savedir = save_dir)

        results_df.loc[results_idx, 'box_precision'] = results['box_precision']
        results_df.loc[results_idx, 'box_recall'] = results['box_recall']
        results_df.loc[results_idx, 'class_recall'] = results['class_recall']['recall'][0]
        results_df.loc[results_idx, 'class_precision'] = results['class_recall']['precision'][0]
        results_df.loc[results_idx, 'miou'] = results['results']['IoU'].mean()
        results_df.loc[results_idx, 'dataset'] = 'train'
        results_df.loc[results_idx, 'iou_thresh'] = iou_thresh
        results_df.loc[results_idx, 'score_thresh'] = score_thresh

        if results_df.loc[results_idx, 'box_recall'] + results_df.loc[results_idx, 'box_precision'] == 0: results_df.loc[results_idx, 'box_f1'] = 0
        else: results_df.loc[results_idx, 'box_f1'] = 2*((results_df.loc[results_idx, 'box_recall']*results_df.loc[results_idx, 'box_precision'])/(results_df.loc[results_idx, 'box_recall']+results_df.loc[results_idx, 'box_precision']))

        results_df.loc[results_idx, 'trees_matched'] = results['results']['match'].sum()
        results_idx += 1

        end = time.time()       
        pipe_idx += 1
        print("test number ", results_idx/2, " of ", len(score_threshs)*len(iou_thresholds), " completed in ", round((end - start)/60, 2), "min")
        results_df.to_csv('df_models/df_train_eval_scores_final.csv')
# %%

results_df
# %%
# model_path = 'df_models/final_model.pt'
# torch.save(model.model.state_dict(),model_path)
# %%
tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
####### CHANGE FROM VAL TO TEST
# tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(tree_actual_df_path, 'TRAIN')
tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(tree_actual_df_path, 'VAL')
# def actual_tree_data(tree_actual_df_path, sheet_name):
    
# Clipped CHM path
# chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
# chm_train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
# chm_test_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
# chm_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_val_clip.tif")

# Clipped ortho path
# ortho_name = 'ortho_test_clip.tif'
ortho_name = 'ortho_val_clip.tif'
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
ortho_clipped_path = ortho_clipped_root + '/' + ortho_name

# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y
# %%
model = main.deepforest()
model.use_release()
print("Current device is {}".format(model.device))
model.to("cuda")
print("Current device is {}".format(model.device))
model.config["gpus"] = 1

model_path = 'df_models/final_model.pt'
model.model.load_state_dict(torch.load(model_path))

ortho_clipped_path_val = "ortho & pointcloud gen/outputs/GT/ortho_val_clip.tif"
with rasterio.open(ortho_clipped_path_val) as source:
    img_val = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img_val = img_val.astype(np.uint8)
img_val_rgb = np.moveaxis(img_val, 0, 2).copy()
img_val_bgr = img_val_rgb[...,::-1].copy()

patch_size = 850
patch_overlap = 0.4
thresh = 0.3
iou_threshold = 0.4
model.config["score_thresh"] = 0.3

# Create Predictions
predictions_df, predictions_df_transform, results_df_df, predicted_raster_image = deep_forest_pred(ortho_name, ortho_clipped_path, ortho_clipped_root, tree_point_calc_csv_path=tree_point_calc_csv_path, tree_point_calc_shifted_csv_path=tree_point_calc_shifted_csv_path, tree_actual_df=tree_actual_df, tree_actual_df_no_dead = tree_actual_df_no_dead, patch_size=patch_size, patch_overlap=patch_overlap, thresh=thresh, iou_threshold=iou_threshold, save_fig = False, save_shape = False)
# predictions_df = model.predict_tile(image=img_val_bgr, patch_size=patch_size, patch_overlap=patch_overlap, iou_threshold=iou_threshold)
# predictions_df = predictions_df[predictions_df['score'] > thresh]
print(f"{predictions_df.shape[0]} predictions kept after applying threshold")
df_image_save_path = 'deepforest_predictions/rasters_with_boxes/report_shit_3_train.png'
# df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + '_thresh-' + str(thresh)  + '_iou-' + str(iou_threshold) + 'TEST.png'
# predicted_raster_image = plot_predictions_from_df(predictions_df, img_val_bgr)
# plt.imsave(df_image_save_path,arr=predicted_raster_image)
# %%
results_df_df
# %%

# %%
tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(4)
tree_actual_df_train = train_test_plots(tree_actual_df_full, start_plot=37, num_rows=33)
tree_actual_df_val = train_test_plots(tree_actual_df_full, start_plot=1, num_rows=18)
tree_actual_df_test = train_test_plots(tree_actual_df_full, start_plot=19, num_rows=18)

with pd.ExcelWriter('data/EG0181T Riverdale A9b Train Test Validation.xlsx') as writer:  
    tree_actual_df_train.to_excel(writer, sheet_name='TRAIN')
    tree_actual_df_val.to_excel(writer, sheet_name='VAL')
    tree_actual_df_test.to_excel(writer, sheet_name='TEST')
# %%
# Read in dataframe of all trees
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

# Read in dataframe of all trees (shifted)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y
tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')
outerpoints_gp, outerpoints_df = outer_trees(tree_point_calc_shifted_df)
outer_trees_poly_df, outer_trees_buffered_poly_df = tree_buffer_area(outerpoints_gp, 1.5)
outer_trees_buffered_poly_df.to_file('ortho & pointcloud gen/outputs/GT/shape_files/boundary.shp', driver='ESRI Shapefile')

# Load, clip and save clipped ortho
ortho_raster_path = os.path.join("ortho & pointcloud gen","outputs","GT",
                            "ortho_corrected_no_compression.tif")
# ortho = rxr.open_rasterio(ortho_raster_path, masked=True).squeeze()
ortho = rxr.open_rasterio(ortho_raster_path, masked=True)
ortho_clipped = ortho.rio.clip(outer_trees_buffered_poly_df.geometry)
ortho_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT",
                            "ortho_corrected_no_compression_clipped.tif")
ortho_clipped.rio.to_raster(ortho_clipped_path)
# %%
with rasterio.open(ortho_clipped_path) as source:
    img = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img = img.astype(np.uint8)
img_rgb = np.moveaxis(img, 0, 2).copy()
img_bgr = img_rgb[...,::-1].copy()
plt.imshow(img_rgb)
# %%
print("The CRS for this data is:", ortho_clipped.rio.crs)
print("The spatial extent is:", ortho_clipped.rio.bounds())
print("The no data value is:", ortho_clipped.rio.nodata)
# %%

# %%
train_point_X_list = [802900.405, 802873.273, 802788.908, 802726.607, 802750.400]
train_point_Y_list = [6695320.204, 6695217.113, 6695239.194, 6695255.324, 6695359.291]

test_point_X_list = [802873.273, 802858.272, 802797.917, 802752.715, 802711.369, 802726.607, 802788.908]
test_point_Y_list = [6695217.113, 6695165.328, 6695181.150, 6695191.818, 6695203.817, 6695255.324, 6695239.194]

val_point_X_list = [802858.272, 802844.700, 802690.985, 802711.369, 802752.715, 802797.917]
val_point_Y_list = [6695165.328, 6695107.615, 6695143.427, 6695203.817, 6695191.818, 6695181.150]

polygon_train_geom = Polygon(zip(train_point_X_list, train_point_Y_list)).buffer(0.3) 
polygon_test_geom = Polygon(zip(test_point_X_list, test_point_Y_list)).buffer(0.3) 
polygon_val_geom = Polygon(zip(val_point_X_list, val_point_Y_list)).buffer(0.3) 
crs = {'init': 'epsg:32735'}
polygon_train = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_train_geom])  
polygon_test = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_test_geom])
polygon_val = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_val_geom]) 

polygon_train.to_file('data/QGIS and CC/train_poly.shp', driver='ESRI Shapefile')
polygon_test.to_file('data/QGIS and CC/test_poly.shp', driver='ESRI Shapefile')
polygon_val.to_file('data/QGIS and CC/val_poly.shp', driver='ESRI Shapefile')

ortho_train_clip = ortho_clipped.rio.clip(polygon_train.geometry)
ortho_test_clip = ortho_clipped.rio.clip(polygon_test.geometry)
ortho_val_clip = ortho_clipped.rio.clip(polygon_val.geometry)

train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "ortho_train_clip.tif")
test_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "ortho_test_clip.tif")
val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "ortho_val_clip.tif")

# ortho_train_clip.rio.to_raster(train_clip_path)
# ortho_test_clip.rio.to_raster(test_clip_path)
# ortho_val_clip.rio.to_raster(val_clip_path)
# %%
# Import and clip raster
chm_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_2_clipped.tif")
chm = rxr.open_rasterio(chm_path, masked=True).squeeze()

polygon_train_geom_chm = Polygon(zip(train_point_X_list, train_point_Y_list))
polygon_test_geom_chm = Polygon(zip(test_point_X_list, test_point_Y_list))
polygon_val_geom_chm = Polygon(zip(val_point_X_list, val_point_Y_list))
crs = {'init': 'epsg:32735'}
polygon_train_chm = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_train_geom_chm])  
polygon_test_chm = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_test_geom_chm])
polygon_val_chm = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_val_geom_chm]) 

chm_train_clip = chm.rio.clip(polygon_train_chm.geometry)
chm_test_clip = chm.rio.clip(polygon_test_chm.geometry)
chm_val_clip = chm.rio.clip(polygon_val_chm.geometry)

chm_train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
chm_test_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
chm_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_val_clip.tif")

# chm_train_clip.rio.to_raster(chm_train_clip_path)
# chm_test_clip.rio.to_raster(chm_test_clip_path)
# chm_val_clip.rio.to_raster(chm_val_clip_path)

# %%
chm_test_clip
# %%
# Create CHM of train and validation data
from rasterio.merge import merge
from rasterio.plot import show
chm_train_clip_for_merge = rasterio.open(chm_train_clip_path)
chm_val_clip_for_merge = rasterio.open(chm_val_clip_path)
chm_train_val_clip, out_trans  = merge([chm_train_clip_for_merge, chm_val_clip_for_merge])

chm_train_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_val_clip.tif")

out_meta = chm_train_clip_for_merge.meta.copy()
out_meta.update({"driver": "GTiff",
                  "height": chm_train_val_clip.shape[1],
                  "width": chm_train_val_clip.shape[2],
                  "transform": out_trans,
                  "crs": ortho_clipped.rio.crs
                  })

with rasterio.open(chm_train_val_clip_path, "w", **out_meta) as dest:
    dest.write(chm_train_val_clip)
# %%
chm_train_val_clip
# %%
ortho_clipped_path_train = "ortho & pointcloud gen/outputs/GT/ortho_train_clip.tif"
ortho_clipped_path_test = "ortho & pointcloud gen/outputs/GT/ortho_test_clip.tif"
ortho_clipped_path_val = "ortho & pointcloud gen/outputs/GT/ortho_val_clip.tif"
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"

with rasterio.open(ortho_clipped_path_train) as source:
    img_train = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img_train = img_train.astype(np.uint8)
img_train_rgb = np.moveaxis(img_train, 0, 2).copy()
img_train_bgr = img_train_rgb[...,::-1].copy()
# plt.imshow(img_train_rgb)

with rasterio.open(ortho_clipped_path_test) as source:
    img_test = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img_test = img_test.astype(np.uint8)
img_test_rgb = np.moveaxis(img_test, 0, 2).copy()
img_test_bgr = img_test_rgb[...,::-1].copy()
# plt.imshow(img_test_rgb)

with rasterio.open(ortho_clipped_path_val) as source:
    img_val = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img_val = img_val.astype(np.uint8)
img_val_rgb = np.moveaxis(img_val, 0, 2).copy()
img_val_bgr = img_val_rgb[...,::-1].copy()
# plt.imshow(img_val)
# %%
################################################
#       LM GRID SEARCH FOR WINDOW SIZE         #
################################################
# %%
# chm_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_val_clip.tif")
# chm_train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
# # chm_val = rxr.open_rasterio(chm_val_clip_path, masked=True)
# tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
# tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

# tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
# tree_actual_df_val, tree_actual_df_full_val, tree_actual_df_no_dead_val, min_height_val = actual_tree_data(tree_actual_df_path, 'VAL')
# tree_actual_df_train, tree_actual_df_full_train, tree_actual_df_no_dead_train, min_height_train = actual_tree_data(tree_actual_df_path, 'TRAIN')

# tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
# tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)

# datasets = ['val', 'train']
# window_sizes = range(10,51)
# for dataset in datasets:
#     results_grid_search = pd.DataFrame(columns=['window_size', 'number_trees_pred', 'number_unallocated','number_dead_pred', 'perc_trees_pred', 'MAE_position', 'MSE_position','max_dist', 'min_dist', 'MAE_height', 'MSE_height', 'RMSE_height','max_height_pred', 'min_height_pred'])
#     for i, window_size in enumerate(window_sizes):

#         if dataset == 'train':
#             df_global_gp_train, tree_positions_from_lm_train, results_df_lm_train = local_maxima_func(chm_train_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_train, window_size=window_size, min_height=0, save_shape_file=False)
#             results_grid_search = pd.concat([results_grid_search, results_df_lm_train])
#             if i % 10 == 0: print(i, "tests of ", len(window_sizes), " completed")

#         elif dataset == 'val':
#             df_global_gp_val, tree_positions_from_lm_val, results_df_lm_val = local_maxima_func(chm_val_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_val, window_size=window_size, min_height=0, save_shape_file=True)
#             results_grid_search = pd.concat([results_grid_search, results_df_lm_val])
#             if i % 10 == 0: print(i, "tests of ", len(window_sizes), " completed")
         
#     results_grid_search.to_csv('height_predictor/window_size_test_results_' + dataset + '.csv', index=False)
# %%
chm_train_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_val_clip.tif")
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df_val, tree_actual_df_full_val, tree_actual_df_no_dead_val, min_height_val = actual_tree_data(tree_actual_df_path, 'VAL')
tree_actual_df_train, tree_actual_df_full_train, tree_actual_df_no_dead_train, min_height_train = actual_tree_data(tree_actual_df_path, 'TRAIN')

tree_actual_df_train_val = pd.concat([tree_actual_df_train, tree_actual_df_val])

tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)

dataset = "train_val"
window_sizes = range(10,51)
results_grid_search = pd.DataFrame(columns=['window_size', 'number_trees_pred', 'number_unallocated','number_dead_pred', 'perc_trees_pred', 'MAE_position', 'MSE_position','max_dist', 'min_dist', 'MAE_height', 'MSE_height', 'RMSE_height','R2','max_height_pred', 'min_height_pred'])
for i, window_size in enumerate(window_sizes):

    df_global_gp_train_val, tree_positions_from_lm_train_val, results_df_lm_train_val,results_df_lm_train_val_2, tree_locations_pred_df = local_maxima_func(chm_train_val_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_train_val, window_size=window_size, min_height=0, save_shape_file=True)
    results_grid_search = pd.concat([results_grid_search, results_df_lm_train_val_2])
    if i % 10 == 0: print(i, "tests of ", len(window_sizes), " completed")
        
results_grid_search.to_csv('height_predictor/window_size_test_results_v4' + dataset + '.csv', index=False)
# %%
chm_test_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df_test, tree_actual_df_full_test, tree_actual_df_no_dead_test, min_height_test = actual_tree_data(tree_actual_df_path, 'TEST')

# tree_actual_df_train_val = pd.concat([tree_actual_df_train, tree_actual_df_val])

tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)

dataset = "test"

results_grid_search = pd.DataFrame(columns=['window_size', 'number_trees_pred', 'number_unallocated','number_dead_pred', 'perc_trees_pred', 'MAE_position', 'MSE_position','max_dist', 'min_dist', 'MAE_height', 'MSE_height', 'RMSE_height','R2','max_height_pred', 'min_height_pred'])

df_global_gp_test, tree_positions_from_lm_test, results_df_lm_test,results_df_lm_test_2, tree_locations_pred_df = local_maxima_func(chm_test_val_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_test, window_size=24, min_height=0, save_shape_file=True)
results_df_lm_test_2
# %%
tree_locations_pred_df.to_csv('../Main/images/working/lm_initial results.csv')
# %%
# results_df_lm_train_val_2
tree_locations_pred_df_no_nan= tree_locations_pred_df[tree_locations_pred_df['tree_id'].isna()==False]
# np.abs(tree_locations_pred_df['Hgt22Rod'] - tree_locations_pred_df['height_pred']).mean()
# tree_locations_pred_df['Hgt22Rod'] - tree_locations_pred_df['height_pred']
r2_score(tree_locations_pred_df_no_nan['Hgt22Rod'], tree_locations_pred_df_no_nan['height_pred'])
# %%
df_global_gp_train_val, tree_positions_from_lm_train_val, results_df_lm_train_val,results_df_lm_train_val_2, tree_locations_pred_df = local_maxima_func(chm_train_val_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_train_val, window_size=26, min_height=0, save_shape_file=True)
# %%
chm_train_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_val_clip.tif")
with rasterio.open(chm_train_val_clip_path) as source:
        chm_img = source.read(1) # Read raster band 1 as a numpy array
        affine = source.transform


size = 2 * 29 + 1
# footprint = np.ones((size, ) * chm_img.ndim, dtype=bool)
# mask = _get_peak_mask(chm_img, footprint, threshold)
image_max = ndi.maximum_filter(chm_img, size=20, mode='constant')
chm_img.shape
# %%
import matplotlib as mpl
test_img = chm_img[2000:2500,2000:2500]
# mpl.rcParams['font.family'] = 'serif'
plt.imshow(test_img, cmap=plt.cm.gray)
plt.savefig('../Main/images/crop_lit_review.png',bbox_inches = 'tight', dpi=300);
# %%
mpl.rcParams['font.serif'] = 'cmr10'
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
image_downscaled = downscale_local_mean(test_img, (9, 9))
plt.imshow(image_downscaled, cmap=plt.cm.gray)

plt.savefig('../Main/images/crop_lit_review_downscaled.png',bbox_inches = 'tight', dpi=300);
# %%
plt.savefig('../Main/images/crop_array_high_res.png',bbox_inches = 'tight', dpi=300);
# %%

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
np.savetxt("../Main/images/working/crop_array.csv", image_downscaled, delimiter=",")

# %%
################################################
#            CREATE TRAINING DATA              #
################################################
# %%
# Crop and save training, test and validation images
windows = preprocess.compute_windows(img_train_rgb, 850, 0.1)
for idx, window in enumerate(windows):

    crop = window.apply(img_train_rgb)
    idx_str = str(idx).zfill(2)
    deepforest.preprocess.save_crop('df_crops/train/crops', 'train', idx_str, crop)

windows = preprocess.compute_windows(img_test_rgb, 850, 0.1)
for idx, window in enumerate(windows):

    crop = window.apply(img_test_rgb)
    idx_str = str(idx).zfill(2)
    deepforest.preprocess.save_crop('df_crops/test/crops', 'test', idx_str, crop)

windows = preprocess.compute_windows(img_val_rgb, 850, 0.1)
for idx, window in enumerate(windows):

    crop = window.apply(img_val_rgb)
    idx_str = str(idx).zfill(2)
    deepforest.preprocess.save_crop('df_crops/val/crops', 'val', idx_str, crop)
# %%
################################################
#             TRAIN DEEP FOREST                #
################################################
# %%
folder_path_train='df_crops_annotations/train/'
dataset_train='train'
annotations_df, annotations_csv_filename, annotations_csv_filepath = annotation_json_to_csv(folder_path_train, dataset_train)
folder_path_val='df_crops_annotations/val/'
dataset_val='val'
image_path = folder_path_val  + 'ortho_cropped/' + 'ortho_val_clip.tif'
annotations_df_val, annotations_csv_filename_val, annotations_csv_filepath_val = annotation_json_to_csv(folder_path_val, dataset_val, image_path='ortho_val_clip.tif')
root_dir_val_ortho = 'df_crops_annotations/val/ortho_cropped'
# %%
annotations_df['label'].value_counts()
# %%
base_dir_split = 'df_crops_annotations/val/ortho_cropped/split_annotations_val'
annotations_files = preprocess.split_raster(annotations_file=annotations_csv_filepath_val, path_to_raster=image_path, base_dir=base_dir_split, patch_size=850, patch_overlap = 0.1, allow_empty=False, image_name='ortho_val_clip.tif')
# %%
model = main.deepforest()
model.use_release()
print("Current device is {}".format(model.device))
model.to("cuda")
print("Current device is {}".format(model.device))
model.config["gpus"] = 1
# %%
# Set up model training
# root_dir = 
model.config["train"]["epochs"] = 3
model.config["workers"] = 0
# model.config["GPUS"] = 1
model.config["save-snapshot"] = False
model.config["train"]["csv_file"] = annotations_csv_filepath
model.config["train"]["root_dir"] = 'df_crops_annotations/train/crops'
# model.config["validation"]["csv_file"] = 'df_crops_annotations/val/ortho_cropped/split_annotations_val/ortho_val_clip.csv'
# model.config["validation"]["root_dir"] = 'df_crops_annotations/val/ortho_cropped/split_annotations_val'
# model.config["train"]['fast_dev_run'] = False
# %%
model.config["validation"]["csv_file"] = None
model.config["validation"]["root_dir"] = None
# %%
# Train model 
from pytorch_lightning.loggers import CSVLogger
logger = CSVLogger("logs", name="my_exp_name")
model.create_trainer(logger=logger)
model.trainer.fit(model)
# %%
model.configure_optimizers
# %%
# Evaluate model on validation data
model.config["score_thresh"] = 0.5
save_dir = 'results/df_training_results'
annotations_cropped_csv_filepath_val = 'df_crops_annotations/val/ortho_cropped/split_annotations_val/ortho_val_clip.csv'
root_dir_val_crops = 'df_crops_annotations/val/ortho_cropped/split_annotations_val'
results = model.evaluate(annotations_cropped_csv_filepath_val, root_dir=root_dir_val_crops, iou_threshold = 0.4, savedir = save_dir)
# %%
results['results']
# %%
model_path = 'df_models/100_epoch_train' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.pt'
torch.save(model.model.state_dict(),model_path)
# %%
################################################
#              TEST DEEP FOREST                #
################################################
# %%
model = main.deepforest()
model.use_release()
print("Current device is {}".format(model.device))
model.to("cuda")
print("Current device is {}".format(model.device))
model.config["gpus"] = 1

model_path = 'df_models/' + '100_epoch_train20211001_0157.pt'
# model.model.load_state_dict(torch.load(model_path))
# %%
# ortho_clipped_path_train = "ortho & pointcloud gen/outputs/GT/ortho_train_clip.tif"
# with rasterio.open(ortho_clipped_path_train) as source:
#     img_train = source.read() # Read raster bands as a numpy array
#     transform_crs = source.transform
#     crs = source.crs

# img_train = img_train.astype(np.uint8)
# img_train_rgb = np.moveaxis(img_train, 0, 2).copy()
# img_train_bgr = img_train_rgb[...,::-1].copy()
# %%
patch_size = 950
patch_overlap = 0.5
thresh = 0.0
iou_threshold = 0.05
# Create Predictions
predictions_df = model.predict_tile(image=img_train_bgr, patch_size=patch_size, patch_overlap=patch_overlap, iou_threshold=iou_threshold)
predictions_df = predictions_df[predictions_df['score'] > thresh]
print(f"{predictions_df.shape[0]} predictions kept after applying threshold")
df_image_save_path = 'deepforest_predictions/rasters_with_boxes/report_shit.png'
# df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + '_thresh-' + str(thresh)  + '_iou-' + str(iou_threshold) + 'TEST.png'
predicted_raster_image = plot_predictions_from_df(predictions_df, img_train_bgr)
plt.imsave(df_image_save_path,arr=predicted_raster_image)
# %%
# # Read in dataframe of all trees
# tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
# tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

# # Import actual tree data from Sappi
# tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(4)

#  # Read in dataframe of all trees (shifted)
# tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
# tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
# tree_point_calc_shifted_df['tree_easting'] = tree_point_calc_shifted_df['geometry'].x
# tree_point_calc_shifted_df['tree_northing'] = tree_point_calc_shifted_df['geometry'].y
# tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')

# # patch_sizes = [700, 750, 800, 850, 900, 950]
# # patch_overlaps = [0.2, 0.3, 0.4, 0.5, 0.6]
# # thresholds = [0.2, 0.3, 0.4, 0.5]
# # iou_thresholds = [0.05, 0.1, 0.15, 0.2]

# patch_sizes = [950]
# patch_overlaps = [0.5]
# thresholds = [0.9]
# iou_thresholds = [0.05]

# results_idx = 0
# results_df = pd.DataFrame()
# for patch_size in patch_sizes:
#     for patch_overlap in patch_overlaps:
#         for thresh in thresholds:
#             for iou_threshold in iou_thresholds:

#                 try:
#                     # Create Predictions
#                     predictions_df = model.predict_tile(image=img_bgr, patch_size=patch_size, patch_overlap=patch_overlap, iou_threshold=iou_threshold)
#                     predictions_df = predictions_df[predictions_df['score'] > thresh]
#                     print(f"{predictions_df.shape[0]} predictions kept after applying threshold")
#                     df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + '_thresh-' + str(thresh)  + '_iou-' + str(iou_threshold) + '.png'
#                     predicted_raster_image = plot_predictions_from_df(predictions_df, img_bgr)
#                     plt.imsave(df_image_save_path,arr=predicted_raster_image)

#                     # Transform predictions to original CRS
#                     predictions_df_transform = predictions_df.copy()
#                     predictions_df_transform['image_path'] = "ortho_corrected_no_compression_clipped.tif"
#                     predictions_df_transform = predictions_df_transform[['xmin', 'ymin', 'xmax', 'ymax','image_path']]
#                     predictions_df_transform = utilities.project_boxes(predictions_df_transform, root_dir=ortho_clipped_root, transform=True)

#                     predictions_df_transform['X'] = predictions_df_transform['xmin'] + (predictions_df_transform['xmax'] - predictions_df_transform['xmin'])/2
#                     predictions_df_transform['Y'] = predictions_df_transform['ymin'] + (predictions_df_transform['ymax'] - predictions_df_transform['ymin'])/2
#                     predictions_df_transform['geometry'] = predictions_df_transform.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)

#                     shape_file_name = 'deepforest_predictions/shapefiles/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + 'thresh' + str(thresh)  + 'iou' + str(iou_threshold) + '.shp'
#                     predictions_df_transform.to_file(shape_file_name, driver='ESRI Shapefile')
                    
#                     clear_flag = 0
#                     for idx in range(len(predictions_df_transform)):

#                         if idx == range(len(predictions_df_transform))[-1]: clear_flag = 1

#                         current_tree = (predictions_df_transform.loc[idx, 'X'], predictions_df_transform.loc[idx, 'Y'])
#                         distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df, clear_flag)
#                         predictions_df_transform.loc[idx, 'tree_id_pred'] = tree_id
#                         predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
#                         predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

#                         del distance_to_nn
#                         del distance_to_nn_squared
#                         del tree_id

#                     # Allocate predictions to actual trees
#                     ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_df_transform)

#                     # Merge with actual data to determine number of dead trees predicted
#                     tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')

#                     # Save results
#                     results_df.loc[results_idx, 'patch_size'] = patch_size
#                     results_df.loc[results_idx, 'patch_overlap'] = patch_overlap
#                     results_df.loc[results_idx, 'thresh'] = thresh
#                     results_df.loc[results_idx, 'iou_threshold'] = iou_threshold
#                     results_df.loc[results_idx, 'number_trees_pred'] = tree_locations_pred_df.shape[0]
#                     results_df.loc[results_idx, 'number_unallocated'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]
#                     results_df.loc[results_idx, 'number_dead_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
#                     results_df.loc[results_idx, 'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
#                     results_df.loc[results_idx, 'MAE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist'].mean()
#                     results_df.loc[results_idx, 'MSE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist_squared'].mean()
#                     results_df.loc[results_idx, 'max_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].max()
#                     results_df.loc[results_idx, 'min_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].min()

#                     results_idx += 1
#                     print('test ' + str(results_idx) + ' done')

#                 except: 
#                     # Save results
#                     results_df.loc[results_idx, 'patch_size'] = patch_size
#                     results_df.loc[results_idx, 'patch_overlap'] = patch_overlap
#                     results_df.loc[results_idx, 'thresh'] = thresh
#                     results_df.loc[results_idx, 'iou_threshold'] = iou_threshold


#                     gc.collect()

#                     results_idx += 1

# results_file_path = 'deepforest_predictions/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '_predict_grid_search_results.csv'
# results_df.to_csv(results_file_path)

# %%
# Import actual tree data from Sappi
tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(4)
# %%
################################################
#         BUILD DEAD TREE CLASSIFIER           #
################################################
# %%
# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'

# Clipped CHM path
# chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
chm_train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
chm_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_val_clip.tif")

# Clipped ortho path
ortho_name = 'ortho_train_clip.tif'
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
ortho_clipped_path = ortho_clipped_root + '/' + ortho_name
# %%
# Import actual tree data from Sappi
tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df_train, tree_actual_df_full_train, tree_actual_df_no_dead_train, min_height_train = actual_tree_data(tree_actual_df_path, 'TRAIN')
# %%
# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y

patch_size = 850
patch_overlap = 0.5
thresh = 0.4
iou_threshold = 0.05

# Load model
model = main.deepforest()
model.use_release()
print("Current device is {}".format(model.device))
model.to("cuda")
print("Current device is {}".format(model.device))
model.config["gpus"] = 1

model_path = 'df_models/' + '100_epoch_train20211001_0157.pt'
model.model.load_state_dict(torch.load(model_path))

# Get tree positions from DeepForest
predictions_df, predictions_df_transform, results_df_df, predicted_raster_image = deep_forest_pred(ortho_name, ortho_clipped_path, ortho_clipped_root, tree_point_calc_csv_path=tree_point_calc_csv_path, tree_point_calc_shifted_csv_path=tree_point_calc_shifted_csv_path, tree_actual_df=tree_actual_df_train, tree_actual_df_no_dead = tree_actual_df_no_dead_train, patch_size=patch_size, patch_overlap=patch_overlap, thresh=thresh, iou_threshold=iou_threshold, save_fig = True, save_shape = True)
# %%
# Get tree positions from LocalMaxima
# Clipped CHM path
# chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_2_clipped.tif")
df_global_gp, tree_positions_from_lm = local_maxima_func(chm_clipped_path, tree_actual_df, window_size=29, min_height=min_height, save_shape_file=True)
# %%
# Get average width, height and side of bounding box
average_width, average_height, average_side = get_average_box_size(predictions_df_transform)

expansion_factor = 0.5
expansion_size = average_side * expansion_factor

# %%


# dead_tree_for_classification, crop_df = dead_tree_classifier_dataset(predictions_df_transform, tree_actual_df, tree_positions_from_lm, chm_clipped_path, window_size=29, save_crops=False)
dead_trees_classified, non_dead_tree_classified  = dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df_train, save_crops=False)
classification_all = pd.concat([dead_trees_classified[['X', 'Y', 'class']], non_dead_tree_classified[['X', 'Y', 'class']]], ignore_index=True)
classification_all = boxes_from_points(classification_all, expansion_size)
# dead_tree_classification_df = boxes_from_points(dead_tree_classification_df, expansion_size)
# %%
# classification_df_1 = pd.read_csv('tree_classifier_crops/tree_crops_rgb_classified.csv')
classification_df_2 = pd.read_csv('tree_classifier_crops/crops_file_from_all_trees/20210926_2312_tree_crops_avg_rgb_classified.csv')
classification_df_2 = classification_df_2[classification_df_2['class'].isna()==False]
classification_df_2['class'] = classification_df_2['class'].astype(int)
# %%
# classification_all = pd.concat([dead_tree_classification_df[['X', 'Y', 'class']], classification_df_2[['X', 'Y', 'class']]], ignore_index=True)
# classification_all = boxes_from_points(classification_all, expansion_size)
# %%
classification_all
# %%
ortho_cropped_for_cropping = rxr.open_rasterio(ortho_clipped_path, masked=True).squeeze()
# Crop all trees and obtain mean and SD R, G and B values for each crop
crop_df = crop_array(ortho_cropped_for_cropping, classification_all, save_crops = False)
# %%
crop_df['class'] = classification_all['class']
# %%
feature_lists_df = pd.DataFrame(columns=['features'])
features_names = ['r_avg', 'g_avg', 'b_avg', 'r_sd', 'g_sd', 'b_sd']
from itertools import combinations
df_iter = 0
for r in range(1,len(features_names)-1):

    feature_comb = list(combinations(features_names, r))
    for i in range(len(feature_comb)):
        feature_lists_df.loc[df_iter,'features'] =  list(feature_comb[i])

        df_iter +=1

feature_lists_df.loc[feature_lists_df.shape[0],'features'] = features_names
 # %%
X = crop_df[features_names]
Y = crop_df['class']
# %%
# Minmax scaler
scaler = MinMaxScaler()
# scaler.fit(X)
X_scaled = pd.DataFrame()
X_scaled[features_names] = scaler.fit_transform(X[features_names])
# %%
scores_df_all_tests = pd.DataFrame(columns=['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'model'])
test_counter = 0
# for idx in range(len(feature_lists_df)):
#     features_test = feature_lists_df.loc[idx,'features']
#     X_features = X_scaled[features_test]
#     print(features_test)

for i in range(20):
    # Split dataset into test and train
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.25, stratify=Y)
    # Instantiate model and fit
    

    rf = RandomForestClassifier(n_estimators = 100, max_depth=None).fit(x_train, np.ravel(y_train))
    ab = AdaBoostClassifier(n_estimators = 50, learning_rate = 1).fit(x_train, np.ravel(y_train))
    svm_model = svm.SVC(probability=True).fit(x_train, np.ravel(y_train))
    xgb_model = xgb.XGBClassifier(n_estimators = 100, objective="reg:squarederror", random_state=42, use_label_encoder=False).fit(x_train, np.ravel(y_train))

    # Make predictions
    predictions_rf = rf.predict(x_test)
    predictions_ab = ab.predict(x_test)
    predictions_xgb = xgb_model.predict(x_test)
    predictions_svm = svm_model.predict(x_test)
    predictions_rf_prob = rf.predict_proba(x_test)
    predictions_ab_prob = ab.predict_proba(x_test)
    predictions_svm_prob = svm_model.predict_proba(x_test)

    score_df_rf = classification_scores(y_true=y_test, y_pred=predictions_rf, y_pred_prob=predictions_rf_prob[:, 1], model='random forest')
    score_df_ab = classification_scores(y_true=y_test, y_pred=predictions_ab, y_pred_prob=predictions_ab_prob[:, 1], model='adaboost')
    score_df_xgb = classification_scores(y_true=y_test, y_pred=np.round(predictions_xgb), y_pred_prob=predictions_xgb, model='xgboost')
    score_df_svm = classification_scores(y_true=y_test, y_pred=predictions_svm, y_pred_prob=predictions_svm_prob[:, 1], model='svm')

    scores_df_all_models = pd.concat([score_df_rf,score_df_ab,score_df_xgb,score_df_svm]).reset_index(drop=True)
    # scores_df_all_models.loc[df_idx,'feature_list'] = features_test

    scores_df_all_tests = pd.concat([scores_df_all_tests, scores_df_all_models]).reset_index(drop=True)

    test_counter += 1

    if test_counter % 200 == 0:

        print(test_counter, ' tests of ', feature_lists_df.shape[0]*10*4, 'completed') 

scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc']] = scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc']].astype(float)
scores_df_all_tests_red = scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc', 'model']]
scores_df_all_tests_red = scores_df_all_tests_red.reset_index(drop=True)
scores_df_all_tests_avg = scores_df_all_tests_red.groupby(['model'], as_index=False).mean()
# %%
predictions_rf_prob

# %%
# %%
# scores_df_all_tests.to_csv('tree_classifier_crops/classifier_scores/model_tests_all_scaled_v3_with_feature_combinations.csv')
scores_df_all_tests.to_csv('tree_classifier_crops/classifier_scores/model_tests_all_scaled_v4.csv')
scores_df_all_tests_avg.to_csv('tree_classifier_crops/classifier_scores/model_tests_average_scores_scaled_v4.csv')
# %%
feature_lists_df.loc[56,'features']
# %%
feature_lists_df.loc[42,'features']
# %%
# svm grid search
feature_list_idxs = [42, 44, 45, 56]
cs = [0.1, 0.5, 0.8, 1, 1.5]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
degrees = [1,3,5]

grid_search_idx = 0
svm_parameter_search = pd.DataFrame(columns=['feature_idx','C', 'kernel', 'degree', 'accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'model'])
for feature_list_idx in feature_list_idxs:

    feature_list_for_test = feature_lists_df.loc[feature_list_idx,'features']
    x_train, x_test, y_train, y_test = train_test_split(X_scaled[feature_list_for_test], Y, test_size = 0.25, stratify=Y, random_state=42)
    
    for c in cs:
        for kernel in kernels:
            for degree in degrees:

                svm_model = svm.SVC(C=c, kernel=kernel, degree=degree, probability=True).fit(x_train, np.ravel(y_train))
                predictions_svm = svm_model.predict(x_test)
                predictions_svm_prob = svm_model.predict_proba(x_test)
                score_df_svm = classification_scores(y_true=y_test, y_pred=predictions_svm, y_pred_prob=predictions_svm_prob[:, 1], model='svm',features=None, feature_list_id=feature_list_idx)
                svm_parameter_search.loc[grid_search_idx,'C'] = c
                svm_parameter_search.loc[grid_search_idx,'kernel'] = kernel
                svm_parameter_search.loc[grid_search_idx,'degree'] = degree
                svm_parameter_search.loc[grid_search_idx,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_svm.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]

                grid_search_idx += 1 
    print('feature list ', feature_list_idx, ' done')

svm_parameter_search.to_csv('tree_classifier_crops/classifier_scores/svm_scores_v5.csv')
# %%
# Random Forest grid search
feature_list_idxs = [42, 41, 43, 56]
n_estimators = [10, 50, 100, 150, 200, 250, 300, 350]
criterions = ['gini', 'entropy']
max_depths = [None, 10, 20, 50, 100, 150, 200, 250]

grid_search_idx = 0
rf_parameter_search = pd.DataFrame(columns=['n_estimators', 'criteria', 'max_depth', 'accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'model'])
for feature_list_idx in feature_list_idxs:

    feature_list_for_test = feature_lists_df.loc[feature_list_idx,'features']
    x_train, x_test, y_train, y_test = train_test_split(X_scaled[feature_list_for_test], Y, test_size = 0.25, stratify=Y, random_state=21)

    for n_estimator in n_estimators:
        for criterion in criterions:
            for max_depth in max_depths:

                rf = RandomForestClassifier(n_estimators = n_estimator, max_depth=max_depth, criterion = criterion, random_state=42).fit(x_train, np.ravel(y_train))
                predictions_rf = rf.predict(x_test)
                predictions_rf_prob = rf.predict_proba(x_test)
                score_df_rf = classification_scores(y_true=y_test, y_pred=predictions_rf, y_pred_prob=predictions_rf_prob[:, 1], model='random forest',features=None, feature_list_id=feature_list_idx)

                rf_parameter_search.loc[grid_search_idx,'n_estimators'] = n_estimator
                rf_parameter_search.loc[grid_search_idx,'criteria'] = criterion
                rf_parameter_search.loc[grid_search_idx,'max_depth'] = max_depth
                rf_parameter_search.loc[grid_search_idx,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_rf.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]

                grid_search_idx += 1

    print('feature list ', feature_list_idx, ' done') 
rf_parameter_search.to_csv('tree_classifier_crops/classifier_scores/random_forest_scores_v5.csv')
# %%
# Build and save model (n_estimators = 50, criterion = 'gini, max_depth=None)
rf = RandomForestClassifier(n_estimators = 150, max_depth=10, criterion = 'entropy', random_state=42).fit(x_train, np.ravel(y_train))
# rf_model_filename = 'tree_classifier_crops/saved_models/random_forest_v2.sav'
# pickle.dump(rf, open(rf_model_filename, 'wb'))
features_names = X_scaled.columns

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
forest_importances = pd.Series(importances, index=features_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
# %%
# %%
for feature_list_idx in feature_list_idxs:

    feature_list_for_test = feature_lists_df.loc[56,'features']
    x_train, x_test, y_train, y_test = train_test_split(X_scaled[feature_list_for_test], Y, test_size = 0.25, stratify=Y, random_state=42)
svm_model = svm.SVC(C=0.5, kernel='poly', degree=5, probability=True).fit(x_train, np.ravel(y_train))
svm_model_filename = 'tree_classifier_crops/saved_models/svm.sav'
pickle.dump(svm_model, open(svm_model_filename, 'wb'))
# %%
################################################
#          TEST DEAD TREE CLASSIFIER           #
################################################
# Import actual tree data from Sappi
tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(4)
# Clipped CHM path
chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_2_clipped.tif")

# Clipped ortho path
ortho_clipped_path = "ortho & pointcloud gen/outputs/GT/ortho_corrected_no_compression_clipped.tif"
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
# %%
# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'

# Load DF model
model = main.deepforest()
model.use_release()
print("Current device is {}".format(model.device))
model.to("cuda")
print("Current device is {}".format(model.device))
model.config["gpus"] = 1

model_path = 'df_models/final_model.pt'
model.model.load_state_dict(torch.load(model_path))

patch_size = 950
patch_overlap = 0.5
thresh = 0.9
iou_threshold = 0.05

# Get tree positions from DeepForest
predictions_df, predictions_df_transform, results_df_df, predicted_raster_image = deep_forest_pred(ortho_clipped_path, ortho_clipped_root, tree_point_calc_csv_path, tree_point_calc_shifted_csv_path, tree_actual_df, tree_actual_df_no_dead = tree_actual_df_no_dead, patch_size, patch_overlap, thresh, iou_threshold, save_fig = False, save_shape = False)

# Get average width, height and side of bounding box
average_width, average_height, average_side = get_average_box_size(predictions_df_transform)

expansion_factor = 0.5
expansion_size = average_side * expansion_factor
# %%
# Get tree positions from LocalMaxima
df_global_gp, tree_positions_from_lm, results_df_lm = local_maxima_func(chm_clipped_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df,  window_size=29, min_height=min_height, save_shape_file=True)
# %%
total_trees = results_df_df.iloc[0]['number_trees_pred'] + results_df_lm.iloc[0]['number_trees_pred']
total_dead_trees = results_df_df.iloc[0]['number_dead_pred'] + results_df_lm.iloc[0]['number_dead_pred']
total_unallocated = results_df_df.iloc[0]['number_unallocated'] + results_df_lm.iloc[0]['number_unallocated']
print('total trees: ',total_trees, '\ntotal dead trees: ', total_dead_trees, '\ntotal_unallocated: ', total_unallocated)
# %%
df_global_gp_red = df_global_gp[['X', 'Y', 'tree_id_pred']]
df_global_gp_red['model'] = 'LM'
predictions_df_transform_red = predictions_df_transform[['X', 'Y', 'tree_id_pred']]
predictions_df_transform_red.loc[:,'model'] = 'DF'
all_trees_pred = pd.concat([df_global_gp_red, predictions_df_transform_red]).reset_index(drop=True)
# %%
tree_positions_expanded_for_classification_all = boxes_from_points(all_trees_pred, expansion_size)
# %%
# Load ortho
ortho_cropped_for_cropping = rxr.open_rasterio(ortho_clipped_path, masked=True).squeeze()
# Crop all trees and obtain average R, G and B values for each crop
crop_df = crop_array(ortho_cropped_for_cropping, tree_positions_expanded_for_classification_all, save_crops = False)
crop_df.to_csv('tree_classifier_crops/crops_file_from_all_trees/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") +'_tree_crops_avg_rgb.csv',index=False)
# %%
crop_df = pd.read_csv('tree_classifier_crops/crops_file_from_all_trees/20210927_2243_tree_crops_avg_rgb.csv')
crop_df = crop_df.drop(columns=['Unnamed: 0'],axis=1)
# %%
# Scale features to match model input
scaler = MinMaxScaler()
# scaler.fit(X)
X_scaled_all_trees = pd.DataFrame()
X_scaled_all_trees[features_names] = scaler.fit_transform(crop_df[features_names])
X_scaled_all_trees.shape
# %%
# Load tree classifier model
svm_model_filename = 'tree_classifier_crops/saved_models/svm.sav'
tree_classifier = pickle.load(open(svm_model_filename,"rb"))
# %%
# Make tree classification predictions
tree_classifications = tree_classifier.predict(X_scaled_all_trees)
tree_classifications_prob_dead = tree_classifier.predict_proba(X_scaled_all_trees)
tree_classifications_dec_func = tree_classifier.decision_function(X_scaled_all_trees)

all_trees_pred_incl_rgb = pd.concat([all_trees_pred,crop_df],axis=1)
all_trees_pred_incl_rgb['class'] = tree_classifications
all_trees_pred_incl_rgb['class_prob_dead'] = tree_classifications_prob_dead[:, 0]
all_trees_pred_incl_rgb['dec_func'] = tree_classifications_dec_func
# all_trees_pred_incl_rgb.to_csv('tree_classifier_crops/crops_file_from_all_trees/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") +'_tree_crops_avg_rgb.csv')
# %%
# Read this: https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
# %%
# Remove all trees with a decision function output of less than -6.5
all_trees_pred_after_removal_LM = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] > -6) & (all_trees_pred_incl_rgb['model'] == 'LM')]
all_trees_pred_after_removal_DF = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] > -6) & (all_trees_pred_incl_rgb['model'] == 'DF')]
removed_LM = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] <= -6) & (all_trees_pred_incl_rgb['model'] == 'LM')]
removed_DF = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] <= -6) & (all_trees_pred_incl_rgb['model'] == 'DF')]
# %%
lm_nn_list = []
for idx in all_trees_pred_after_removal_LM.index:
    current_tree = (all_trees_pred_after_removal_LM.loc[idx, 'X'], all_trees_pred_after_removal_LM.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    lm_nn_list.append(distance_to_nn)

df_nn_list = []
for idx in all_trees_pred_after_removal_DF.index:
    current_tree = (all_trees_pred_after_removal_DF.loc[idx, 'X'], all_trees_pred_after_removal_DF.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    df_nn_list.append(distance_to_nn)

df_removed_nn_list = []
for idx in removed_DF.index:
    current_tree = (removed_DF.loc[idx, 'X'], removed_DF.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    df_removed_nn_list.append(distance_to_nn)

lm_removed_nn_list = []
for idx in removed_LM.index:
    current_tree = (removed_LM.loc[idx, 'X'], removed_LM.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    lm_removed_nn_list.append(distance_to_nn)
# %%
print('DF MAE in position before classifier: ', np.round(results_df_df.loc[0,'MAE_position'],4))
print('LM MAE in position before classifier: ', np.round(results_df_lm.loc[0,'MAE_position'],4))
print('DF MAE in position after classifier: ', np.round(sum(df_nn_list)/len(df_nn_list),4))
print('LM MAE in position after classifier: ', np.round(sum(lm_nn_list)/len(lm_nn_list),4))
print('DF MAE of removed trees: ', np.round(sum(df_removed_nn_list)/len(df_removed_nn_list),4))
print('LM MAE of removed trees: ', np.round(sum(lm_removed_nn_list)/len(lm_removed_nn_list),4))
print('Predictions lost from DF: ', (predictions_df_transform_red.shape[0] - all_trees_pred_after_removal_DF.shape[0]))
print('Predictions lost from LM: ', (df_global_gp_red.shape[0] - all_trees_pred_after_removal_LM.shape[0]))
# %%
all_trees_pred_after_removal_LM_incl_act = all_trees_pred_after_removal_LM.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
all_trees_pred_after_removal_DF_incl_act = all_trees_pred_after_removal_DF.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
all_trees_after_removal = pd.concat([all_trees_pred_after_removal_DF_incl_act[['tree_id_pred','model', '24_Def']], all_trees_pred_after_removal_LM_incl_act[['tree_id_pred','model', '24_Def']]])
# %%
total_trees = results_df_df.iloc[0]['number_trees_pred'] + results_df_lm.iloc[0]['number_trees_pred']
total_dead_trees = results_df_df.iloc[0]['number_dead_pred'] + results_df_lm.iloc[0]['number_dead_pred']
total_unallocated = results_df_df.iloc[0]['number_unallocated'] + results_df_lm.iloc[0]['number_unallocated']

total_trees_after_removal = all_trees_after_removal.shape[0]
total_dead_trees_after_removal = all_trees_after_removal[all_trees_after_removal['24_Def'].isin(['D', 'DT'])].shape[0]
total_unallocated_after_removal = all_trees_after_removal[all_trees_after_removal['tree_id_pred'].isna()].shape[0]

print('Before removal:\ntotal trees: ',total_trees, '\ntotal dead trees: ', total_dead_trees, '\ntotal_unallocated: ', total_unallocated)
print('\nAfter removal:\ntotal trees: ',total_trees_after_removal, '\ntotal dead trees: ', total_dead_trees_after_removal, '\ntotal_unallocated: ', total_unallocated_after_removal)
# %%
################################################
#            TREE POSITION QUALITY             #
################################################
# %%
# current_tree = (df_global_gp.loc[idx, 'X'], df_global_gp.loc[idx, 'Y'])
# distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
# df_global_gp.loc[idx, 'tree_id_pred'] = tree_id
# df_global_gp.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
# df_global_gp.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared
# #%% 
# def nearest_neighbor(tree, tree_locations_df):

#     trees = np.asarray(tree_locations_df[['X', 'Y']])
#     # distances = cdist([current_tree], trees, 'euclidean').T
#     tree = np.array(tree).reshape(-1, 1)
#     distances = euclidean_distances(tree.T, trees)
#     nn_idx = distances.argmin()
#     distance_to_nn = distances.T[nn_idx][0]
#     distance_to_nn_squared = (distances.T[nn_idx][0])**2
#     tree_id = tree_locations_df.loc[nn_idx, 'tree_id']

#     return distance_to_nn, distance_to_nn_squared, tree_id
# # predictions_df_transform
# # %%
# all_trees_pred_after_removal_LM
# all_trees_pred_after_removal_DF
# %%
################# LM TEST ##################
all_trees_pred_after_removal_LM = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] > -6) & (all_trees_pred_incl_rgb['model'] == 'LM')] = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] > -6) & (all_trees_pred_incl_rgb['model'] == 'LM')]
nn_list = []
for idx in range(len(all_trees_pred_after_removal_LM)):

    trees = np.asarray(all_trees_pred_after_removal_LM[['X', 'Y']])
    current_tree = (trees[idx][0], trees[idx][1])
    current_tree = np.array(current_tree).reshape(-1, 1)
    distances = euclidean_distances(current_tree.T, trees)
    distances = np.delete(distances, np.where(distances == 0))
    nn_idx = distances.argmin()
    distance_to_nn = distances.T[nn_idx]
    nn_list.append(distance_to_nn)

average_dist = sum(nn_list) / len(nn_list)
max_dist = max(nn_list)
sd_dist = np.std(nn_list)

for i, idx in enumerate(all_trees_pred_after_removal_LM.index):

    trees = np.asarray(all_trees_pred_after_removal_LM[['X', 'Y']])
    current_tree = (trees[i][0], trees[i][1])
    current_tree = np.array(current_tree).reshape(-1, 1)
    distances = euclidean_distances(current_tree.T, trees)
    distances = np.delete(distances, np.where(distances == 0))
    distances.sort()
    distances = distances[0:4]
    distances = distances[distances <= max_dist]
    all_trees_pred_after_removal_LM.loc[idx,'nn_mean'] = distances.mean()
    all_trees_pred_after_removal_LM.loc[idx,'nn_sd'] = np.std(distances)
    all_trees_pred_after_removal_LM.loc[idx,'nn_min'] = min(distances)

# %%
print(all_trees_pred_after_removal_LM[(all_trees_pred_after_removal_LM['nn_min']<1.4) & (all_trees_pred_after_removal_LM['dec_func']<0.5)].isna().sum()['tree_id_pred'])
print(all_trees_pred_after_removal_LM[(all_trees_pred_after_removal_LM['nn_min']<1.4) & (all_trees_pred_after_removal_LM['dec_func']<0.5)].shape[0])
# %%
################# DF TEST ##################
# %%
all_trees_pred_after_removal_DF = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] > -6) & (all_trees_pred_incl_rgb['model'] == 'DF')]
nn_list = []
for idx in range(len(all_trees_pred_after_removal_DF)):

    trees = np.asarray(all_trees_pred_after_removal_DF[['X', 'Y']])
    current_tree = (trees[idx][0], trees[idx][1])
    current_tree = np.array(current_tree).reshape(-1, 1)
    distances = euclidean_distances(current_tree.T, trees)
    distances = np.delete(distances, np.where(distances == 0))
    nn_idx = distances.argmin()
    distance_to_nn = distances.T[nn_idx]
    nn_list.append(distance_to_nn)

average_dist = sum(nn_list) / len(nn_list)
max_dist = max(nn_list)
sd_dist = np.std(nn_list)

for i, idx in enumerate(all_trees_pred_after_removal_DF.index):

    trees = np.asarray(all_trees_pred_after_removal_DF[['X', 'Y']])
    current_tree = (trees[i][0], trees[i][1])
    current_tree = np.array(current_tree).reshape(-1, 1)
    distances = euclidean_distances(current_tree.T, trees)
    distances = np.delete(distances, np.where(distances == 0))
    distances.sort()
    distances = distances[0:4]
    distances = distances[distances <= max_dist]
    all_trees_pred_after_removal_DF.loc[idx,'nn_mean'] = distances.mean()
    all_trees_pred_after_removal_DF.loc[idx,'nn_sd'] = np.std(distances)
    all_trees_pred_after_removal_DF.loc[idx,'nn_min'] = min(distances)

# %%
# %%
all_trees_pred_after_removal_DF[(all_trees_pred_after_removal_DF['nn_min']<1.9) & (all_trees_pred_after_removal_DF['dec_func']<0)]
# %%
print(all_trees_pred_after_removal_DF[(all_trees_pred_after_removal_DF['nn_min']<1.9) & (all_trees_pred_after_removal_DF['dec_func']<0.4)].shape[0])
print(all_trees_pred_after_removal_DF[(all_trees_pred_after_removal_DF['nn_min']<1.9) & (all_trees_pred_after_removal_DF['dec_func']<0.4)].isna().sum()['tree_id_pred'])
print(all_trees_pred_after_removal_DF[(all_trees_pred_after_removal_DF['nn_min']<1.9) & (all_trees_pred_after_removal_DF['dec_func']<0.4)].shape[0])
# %%
# %%
all_trees_pred_after_removal_LM_2 = all_trees_pred_after_removal_LM[~((all_trees_pred_after_removal_LM['nn_min']<1.5) & (all_trees_pred_after_removal_LM['dec_func']<0.5))]
all_trees_pred_after_removal_DF_2 = all_trees_pred_after_removal_DF[~((all_trees_pred_after_removal_DF['nn_min']<1.9) & (all_trees_pred_after_removal_DF['dec_func']<-0.4))]

all_trees_pred_after_removal_LM_incl_act_2 = all_trees_pred_after_removal_LM_2.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
all_trees_pred_after_removal_DF_incl_act_2 = all_trees_pred_after_removal_DF_2.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
all_trees_after_removal_2 = pd.concat([all_trees_pred_after_removal_DF_incl_act_2[['tree_id_pred','model', '24_Def']], all_trees_pred_after_removal_LM_incl_act_2[['tree_id_pred','model', '24_Def']]])

total_trees = results_df_df.iloc[0]['number_trees_pred'] + results_df_lm.iloc[0]['number_trees_pred']
total_dead_trees = results_df_df.iloc[0]['number_dead_pred'] + results_df_lm.iloc[0]['number_dead_pred']
total_unallocated = results_df_df.iloc[0]['number_unallocated'] + results_df_lm.iloc[0]['number_unallocated']

total_trees_after_removal = all_trees_after_removal.shape[0]
total_dead_trees_after_removal = all_trees_after_removal[all_trees_after_removal['24_Def'].isin(['D', 'DT'])].shape[0]
total_unallocated_after_removal = all_trees_after_removal[all_trees_after_removal['tree_id_pred'].isna()].shape[0]

total_trees_after_removal_2 = all_trees_after_removal_2.shape[0]
total_dead_trees_after_removal_2 = all_trees_after_removal_2[all_trees_after_removal_2['24_Def'].isin(['D', 'DT'])].shape[0]
total_unallocated_after_removal_2 = all_trees_after_removal_2[all_trees_after_removal_2['tree_id_pred'].isna()].shape[0]

print('Before removal:\ntotal trees: ',total_trees, '\ntotal dead trees: ', total_dead_trees, '\ntotal_unallocated: ', total_unallocated)
print('\nAfter removal:\ntotal trees: ',total_trees_after_removal, '\ntotal dead trees: ', total_dead_trees_after_removal, '\ntotal_unallocated: ', total_unallocated_after_removal)
print('\nAfter removal 2:\ntotal trees: ',total_trees_after_removal_2, '\ntotal dead trees: ', total_dead_trees_after_removal_2, '\ntotal_unallocated: ', total_unallocated_after_removal_2)
# %%
removed_LM_2 = all_trees_pred_after_removal_LM[((all_trees_pred_after_removal_LM['nn_min']<1.5) & (all_trees_pred_after_removal_LM['dec_func']<0.5))]
removed_DF_2 = all_trees_pred_after_removal_DF[((all_trees_pred_after_removal_DF['nn_min']<1.9) & (all_trees_pred_after_removal_DF['dec_func']<-0.4))]
# %%
lm_nn_list_2 = []
for idx in all_trees_pred_after_removal_LM_2.index:
    current_tree = (all_trees_pred_after_removal_LM_2.loc[idx, 'X'], all_trees_pred_after_removal_LM_2.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    lm_nn_list_2.append(distance_to_nn)

df_nn_list_2 = []
for idx in all_trees_pred_after_removal_DF_2.index:
    current_tree = (all_trees_pred_after_removal_DF_2.loc[idx, 'X'], all_trees_pred_after_removal_DF_2.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    df_nn_list_2.append(distance_to_nn)

df_removed_nn_list_2 = []
for idx in removed_DF_2.index:
    current_tree = (removed_DF_2.loc[idx, 'X'], removed_DF_2.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    df_removed_nn_list_2.append(distance_to_nn)

lm_removed_nn_list_2 = []
for idx in removed_LM_2.index:
    current_tree = (removed_LM_2.loc[idx, 'X'], removed_LM_2.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    lm_removed_nn_list_2.append(distance_to_nn)
# %%
print('DF MAE in position before classifier: ', np.round(results_df_df.loc[0,'MAE_position'],4))
print('LM MAE in position before classifier: ', np.round(results_df_lm.loc[0,'MAE_position'],4),'\n')
print('DF MAE in position after classifier: ', np.round(sum(df_nn_list)/len(df_nn_list),6))
print('LM MAE in position after classifier: ', np.round(sum(lm_nn_list)/len(lm_nn_list),6))
print('DF MAE of removed trees: ', np.round(sum(df_removed_nn_list)/len(df_removed_nn_list),6))
print('LM MAE of removed trees: ', np.round(sum(lm_removed_nn_list)/len(lm_removed_nn_list),6))
print('Predictions lost from DF: ', (predictions_df_transform_red.shape[0] - all_trees_pred_after_removal_DF.shape[0]))
print('Predictions lost from LM: ', (df_global_gp_red.shape[0] - all_trees_pred_after_removal_LM.shape[0]),'\n')
print('DF MAE in position after position quality: ', np.round(sum(df_nn_list_2)/len(df_nn_list_2),6))
print('LM MAE in position after position quality: ', np.round(sum(lm_nn_list_2)/len(lm_nn_list_2),6))
print('DF MAE of removed trees: ', np.round(sum(df_removed_nn_list_2)/len(df_removed_nn_list_2),6))
print('LM MAE of removed trees: ', np.round(sum(lm_removed_nn_list_2)/len(lm_removed_nn_list_2),6))
print('Predictions lost from DF: ', (all_trees_pred_after_removal_DF.shape[0] - all_trees_pred_after_removal_DF_2.shape[0]))
print('Predictions lost from LM: ', (all_trees_pred_after_removal_LM.shape[0] - all_trees_pred_after_removal_LM_2.shape[0]))

# %%
################################################
#         DF HEIGHT MODEL TRAINING             #
################################################z
# %%
folder_path='df_crops_annotations/train/'
dataset='train'
annotations_df, annotations_csv_filename, annotations_csv_filepath = annotation_json_to_csv(folder_path, dataset)
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
# %%
with rasterio.open(ortho_clipped_path_train) as source:
    img_train = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img_train = img_train.astype(np.uint8)
img_train_rgb = np.moveaxis(img_train, 0, 2).copy()
img_train_bgr = img_train_rgb[...,::-1].copy()

with rasterio.open(ortho_clipped_path_val) as source:
    img_val = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img_val = img_val.astype(np.uint8)
img_val_rgb = np.moveaxis(img_val, 0, 2).copy()
img_val_bgr = img_val_rgb[...,::-1].copy()

with rasterio.open(ortho_clipped_path_test) as source:
    img_test = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img_test = img_test.astype(np.uint8)
img_test_rgb = np.moveaxis(img_test, 0, 2).copy()
img_test_bgr = img_test_rgb[...,::-1].copy()
# %%
windows = preprocess.compute_windows(img_train_rgb, 850, 0.1)
crop_details_df = pd.DataFrame(columns=['image_name', 'image_path', 'xmin', 'ymin', 'xmax', 'ymax'])
for idx, window in enumerate(windows):
    rect = window.getRect()
    idx_str = str(idx).zfill(2)
    crop_details_df.loc[idx,'image_name'] = 'train_' + idx_str + '.png'
    crop_details_df.loc[idx,'image_path'] = ortho_name_train
    crop_details_df.loc[idx,'xmin'] = rect[0]
    crop_details_df.loc[idx,'ymin'] = rect[1]
    crop_details_df.loc[idx,'xmax'] = rect[0] + rect[3]
    crop_details_df.loc[idx,'ymax'] = rect[1] + rect[3]
    
crop_details_df_transform = utilities.project_boxes(crop_details_df, root_dir=ortho_clipped_root)
# %%
ortho_for_transform = rxr.open_rasterio(ortho_clipped_path, masked=True)
crs = ortho_for_transform.rio.crs
for idx in crop_details_df_transform.index:

    geom = crop_details_df_transform.loc[idx,'geometry']
    geom_df = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[geom])  
    ortho_for_transform_crop = ortho_for_transform.rio.clip(geom_df.geometry)
    image_name = crop_details_df_transform.loc[idx,'image_name'].replace('.png','')
    ortho_clipped_transform_path = 'df_crops_annotations/train/cropped_ortho/' + image_name + '.tif'
    ortho_for_transform_crop.rio.to_raster(ortho_clipped_transform_path)
# %%
annotations_df['ortho_path'] = annotations_df['image_path']
# annotations_df['ortho_path'] = annotations_df['ortho_path'].astype('str')
annotations_df['image_path'] = annotations_df['image_path'].replace('.png', '.tif', regex=True)
annotations_df['image_path'] = annotations_df['image_path'].replace('df_crops_annotations/train/crops/', '', regex=True)
# %%
ortho_clipped_tranform_root = 'df_crops_annotations/train/cropped_ortho'
annotations_df_transform = pd.DataFrame(columns = annotations_df.columns)
for image_path in annotations_df['image_path'].unique():

    working_df = annotations_df[annotations_df['image_path'] == image_path]
    working_df_transform = utilities.project_boxes(working_df, root_dir=ortho_clipped_tranform_root)

    annotations_df_transform = pd.concat([annotations_df_transform, working_df_transform])

annotations_df_transform_gp = gpd.GeoDataFrame(annotations_df_transform, geometry='geometry').reset_index(drop=True)
shape_file_path= 'df_crops_annotations/train/annotation shape files/'  + datetime.datetime.now().strftime("%Y%m%d_%H%M") + 'training_annotations.shp'
annotations_df_transform_gp.to_file(shape_file_path, driver='ESRI Shapefile')

# %%
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

annotations_df_transform['X'] = annotations_df_transform['xmin'] + (annotations_df_transform['xmax'] - annotations_df_transform['xmin'])/2
annotations_df_transform['Y'] = annotations_df_transform['ymin'] + (annotations_df_transform['ymax'] - annotations_df_transform['ymin'])/2
# annotations_df_transform['geometry'] = annotations_df_transform.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)
# %%
folder_path_val='df_crops_annotations/val/'
dataset_val='val'
annotations_df_val, annotations_csv_filename_val, annotations_csv_filepath_val = annotation_json_to_csv(folder_path_val, dataset_val, image_path = 'ortho_cropped/ortho_val_clip.tif')
annotations_df_val['image_path'] = 'ortho_val_clip.tif'
root_dir_val = folder_path_val + 'ortho_cropped'
annotations_df_val_transform = utilities.project_boxes(annotations_df_val, root_dir=root_dir_val)
# %%
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

annotations_df_val_transform['X'] = annotations_df_val_transform['xmin'] + (annotations_df_val_transform['xmax'] - annotations_df_val_transform['xmin'])/2
annotations_df_val_transform['Y'] = annotations_df_val_transform['ymin'] + (annotations_df_val_transform['ymax'] - annotations_df_val_transform['ymin'])/2
# %%
annotations_df_val_transform_gp = gpd.GeoDataFrame(annotations_df_val_transform, geometry='geometry').reset_index(drop=True)
shape_file_path= 'df_crops_annotations/val/annotation shape files/'  + datetime.datetime.now().strftime("%Y%m%d_%H%M") + 'validation_annotations.shp'
annotations_df_val_transform_gp.to_file(shape_file_path, driver='ESRI Shapefile')
# %%
annotations_df_val_transform_gp = gpd.GeoDataFrame(annotations_df_val_transform, geometry='geometry').reset_index(drop=True)
annotations_df_transform_gp = gpd.GeoDataFrame(annotations_df_transform, geometry='geometry').reset_index(drop=True)
# %%
annotations_df_transform['centroid'] = annotations_df_transform_gp['geometry'].centroid
annotations_df_val_transform['centroid'] = annotations_df_val_transform_gp['geometry'].centroid
annotations_df_val_transform['X'] = gpd.GeoSeries(annotations_df_val_transform['centroid']).x
annotations_df_val_transform['Y'] = gpd.GeoSeries(annotations_df_val_transform['centroid']).y
annotations_df_transform['X'] = gpd.GeoSeries(annotations_df_transform['centroid']).x
annotations_df_transform['Y'] = gpd.GeoSeries(annotations_df_transform['centroid']).y
# %%
# Allocate trees to boxes
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y
tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')

for idx in range(len(annotations_df_transform)):

    current_tree = (annotations_df_transform.loc[idx, 'X'], annotations_df_transform.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    annotations_df_transform.loc[idx, 'tree_id_pred'] = tree_id
    annotations_df_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
    annotations_df_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

for idx in range(len(annotations_df_val_transform)):

    current_tree = (annotations_df_val_transform.loc[idx, 'X'], annotations_df_val_transform.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    annotations_df_val_transform.loc[idx, 'tree_id_pred'] = tree_id
    annotations_df_val_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
    annotations_df_val_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

# ids_to_remove_pred_id, annotations_df_transform = find_duplicates(annotations_df_transform)
# %%
remove_list_train = []
for tree_id in annotations_df_transform['tree_id_pred'].unique():

    working_df = annotations_df_transform[annotations_df_transform['tree_id_pred'] == tree_id].sort_values(by='tree_id_pred_nn_dist', ascending=True)

    if working_df.shape[0] == 1: continue

    else:
        remove_idxs = list(working_df.iloc[1::].index)
        for idx in remove_idxs:
            remove_list_train.append(idx)
    
annotations_df_transform_cleaned_train = annotations_df_transform[~annotations_df_transform.index.isin(remove_list_train)]

remove_list_val = []
for tree_id in annotations_df_val_transform['tree_id_pred'].unique():

    working_df = annotations_df_val_transform[annotations_df_val_transform['tree_id_pred'] == tree_id].sort_values(by='tree_id_pred_nn_dist', ascending=True)

    if working_df.shape[0] == 1: continue

    else:
        remove_idxs = list(working_df.iloc[1::].index)
        for idx in remove_idxs:
            remove_list_val.append(idx)
    
annotations_df_val_transform_cleaned_val = annotations_df_val_transform[~annotations_df_val_transform.index.isin(remove_list_val)]

# %%
# Import actual tree data from Sappi - TRAIN
tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df_train, tree_actual_df_full_train, tree_actual_df_no_dead_train, min_height_train = actual_tree_data(tree_actual_df_path, 'TRAIN')

tree_actual_df_ground_train = tree_actual_df_train.copy()
tree_actual_df_ground_train['Hgt22Rod'] = pd.to_numeric(tree_actual_df_ground_train['Hgt22Rod'], errors='coerce')
tree_actual_df_ground_train['Hgt22Drone'] = pd.to_numeric(tree_actual_df_ground_train['Hgt22Drone'], errors='coerce')
tree_actual_df_ground_train = tree_actual_df_ground_train[(tree_actual_df_ground_train['Hgt22Drone'].isna() == False) & (tree_actual_df_ground_train['Hgt22Rod'].isna() == False)]

annotations_df_transform_cleaned_actual_train = annotations_df_transform_cleaned_train.merge(tree_actual_df_ground_train[['tree_id', 'Hgt22Rod']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
annotations_df_transform_cleaned_actual_train = annotations_df_transform_cleaned_actual_train[annotations_df_transform_cleaned_actual_train['tree_id'].isna() == False].reset_index(drop=True)
# Import actual tree data from Sappi - VAL
tree_actual_df_val, tree_actual_df_full_val, tree_actual_df_no_dead_val, min_height_val = actual_tree_data(tree_actual_df_path, 'VAL')

tree_actual_df_ground_val = tree_actual_df_val.copy()
tree_actual_df_ground_val['Hgt22Rod'] = pd.to_numeric(tree_actual_df_ground_val['Hgt22Rod'], errors='coerce')
tree_actual_df_ground_val['Hgt22Drone'] = pd.to_numeric(tree_actual_df_ground_val['Hgt22Drone'], errors='coerce')
tree_actual_df_ground_val = tree_actual_df_ground_val[(tree_actual_df_ground_val['Hgt22Drone'].isna() == False) & (tree_actual_df_ground_val['Hgt22Rod'].isna() == False)]

annotations_df_transform_cleaned_actual_val = annotations_df_val_transform_cleaned_val.merge(tree_actual_df_ground_val[['tree_id', 'Hgt22Rod']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
annotations_df_transform_cleaned_actual_val = annotations_df_transform_cleaned_actual_val[annotations_df_transform_cleaned_actual_val['tree_id'].isna() == False].reset_index(drop=True)
# %%
# Calculate correlations
annotations_df_transform_cleaned_actual_train_cor= pd.DataFrame()
annotations_df_transform_cleaned_actual_train_cor[['Perimeter', 'Area', 'Avg height', 'Max height', 'Min height','SD height', 'Actual Height']] = annotations_df_transform_cleaned_actual_train[['perimeter', 'area', 'avg_chm', 'max_chm', 'min_chm','std_chm', 'Hgt22Rod']]
# calculate the correlation matrix
corr = annotations_df_transform_cleaned_actual_train_cor.corr()
img = plt.figure(figsize = (6.5,6.5))
# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,cmap="YlGnBu")
plt.yticks(rotation=0) 

# plt.figure(figsize=(10, 7))
# sns.set_theme(style="whitegrid", font='cmr10',font_scale=1.2)
# ax = sns.histplot(data=dec_func_proba_df, x="value", hue="Method",stat='density', bins=20,element="step")
# ax.set_xlabel('Probability / scaled decision function', labelpad =5, fontsize = 17, **{'fontfamily':'serif'})
# ax.set_ylabel('Density', labelpad =5, fontsize = 19)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
plt.savefig('../Main/images/corr_mat.png',bbox_inches = 'tight', dpi=300)
# plt.show()


# %%
annotations_df_transform_cleaned_actual_train
 # %%
features_names = ['area', 'avg_chm', 'min_chm','max_chm', 'std_chm']
X_train = annotations_df_transform_cleaned_actual_train[features_names]
X_train['set'] = 'train'
y_train = annotations_df_transform_cleaned_actual_train['Hgt22Rod']

X_val = annotations_df_transform_cleaned_actual_val[features_names]
X_val['set'] = 'val'
y_val = annotations_df_transform_cleaned_actual_val['Hgt22Rod']

X = pd.concat([X_train, X_val]).reset_index(drop=True)
# %%
# Minmax scaler
scaler = MinMaxScaler()
X_scaled = pd.DataFrame()
X_scaled[features_names] = scaler.fit_transform(X[features_names])

# save the scaler
pickle.dump(scaler, open('height_predictor/saved_models/scaler.pkl', 'wb'))
# %%
X_scaled['set'] = X['set']

x_train = X_scaled[X_scaled['set'] == 'train']
x_train = x_train.drop('set', axis=1).reset_index(drop=True)

x_val = X_scaled[X_scaled['set'] == 'val']
x_val = x_val.drop('set', axis=1).reset_index(drop=True)
# %%
print('Shape of X train = ', x_train.shape)
print('Shape of X val = ', x_val.shape)
print('Shape of Y train = ', y_train.shape)
print('Shape of Y val = ', y_val.shape)
# %%
scores_df_all_tests = pd.DataFrame(columns=['mae', 'rmse','mape', 'r2', 'model'])
test_counter = 0

for i in range(20):
    # Split dataset into test and train
    # x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.25)
    # Instantiate model and fit

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

# %%
scores_df_all_tests_avg 
# %%
from sklearn.inspection import permutation_importance
r = permutation_importance(nn_reg, x_val, y_val,n_repeats=30, random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{features_names[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")
# %%
scores_df_all_tests.to_csv('height_predictor/scores/height_model_tests_v3.csv')
scores_df_all_tests_avg.to_csv('height_predictor/scores/height_model_tests_avg_v3.csv')
# %%
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
# %%
nn_model_search = pd.DataFrame(columns=['model_no', 'mae', 'rmse', 'mape', 'r2'])
for i in range(50):

    nn_reg = MLPRegressor(hidden_layer_sizes=(20, 133, 205), solver='adam', activation='relu',learning_rate='invscaling', max_iter=3000, early_stopping=True).fit(x_train, np.ravel(y_train))
    predictions_nn_reg = nn_reg.predict(x_val)
    score_df_nn = regression_scores(y_true=y_val, y_pred=predictions_nn_reg, model='NN')

    nn_model_search.loc[i,'model_no'] = i
    nn_model_search.loc[i,['mae', 'rmse', 'mape', 'r2']] = score_df_nn.loc[0,['mae', 'rmse', 'mape', 'r2']]

    nn_model_filename = 'height_predictor/saved_models/nn_' + str(i) + '.sav'
    pickle.dump(nn_reg, open(nn_model_filename, 'wb'))

nn_model_search.to_csv('height_predictor/scores/nn_height_model_search.csv')
# %%
nn_model_filename = 'height_predictor/saved_models/nn_7.sav'
height_predictor = pickle.load(open(nn_model_filename,"rb"))
# %%
train_pred = height_predictor.predict(x_train)
score_df_nn = regression_scores(y_true=y_train, y_pred=train_pred, model='NN')
score_df_nn
# %%
x = list(range(len(height_predictor.loss_curve_)))
y = height_predictor.loss_curve_
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid", font='cmr10',font_scale=1.2)
ax = sns.lineplot(x=x[2::], y=y[2::])
ax.set_xlabel('Iteration', labelpad =5, fontsize = 17, **{'fontfamily':'serif'})
ax.set_ylabel('Loss', labelpad =5, fontsize = 19)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
plt.savefig('../Main/images/mlp_loss.png',bbox_inches = 'tight', dpi=300)
# %%
annotations_df_transform_cleaned_actual_train.columns
# %%
df_global_gp, tree_positions_from_lm, results_df_lm, results_df_2_lm, tree_locations_pred_df_lm_train = local_maxima_func(chm_train_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_train, window_size=24, min_height=0.4, save_shape_file=False)
df_global_gp, tree_positions_from_lm, results_df_lm, results_df_2_lm, tree_locations_pred_df_lm_val = local_maxima_func(chm_val_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_val, window_size=24, min_height=0.4, save_shape_file=False)
df_global_gp, tree_positions_from_lm, results_df_lm, results_df_2_lm_test, tree_locations_pred_df_lm_test = local_maxima_func(chm_test_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df_test, window_size=24, min_height=0.4, save_shape_file=False)
# %%
train_df = tree_locations_pred_df_lm_train[tree_locations_pred_df_lm_train['Hgt22Rod'].isna()!=True][['height_pred', 'Hgt22Rod']]
test_df = tree_locations_pred_df_lm_test[tree_locations_pred_df_lm_test['Hgt22Rod'].isna()!=True][['height_pred', 'Hgt22Rod']]
y_lm = np.array(train_df['Hgt22Rod'])
x_lm = np.array(train_df['height_pred']).reshape(-1, 1)
# %%
scaler = MinMaxScaler()
x_lm_scaled = scaler.fit_transform(x_lm)
x_test_scaled = scaler.transform(np.array(test_df['height_pred']).reshape(-1, 1))
nn_reg = MLPRegressor(hidden_layer_sizes=(20, 133, 205), solver='adam', activation='relu',learning_rate='invscaling', max_iter=3000, early_stopping=True).fit(x_lm_scaled, np.ravel(y_lm))
nn_predictions = nn_reg.predict(x_test_scaled)

nn_result = regression_scores(np.array(test_df['Hgt22Rod']), nn_predictions,'nn')
nn_result
# mlp_lm_model_filename = 'height_predictor/saved_models/nn_lm.sav'
# pickle.dump(nn_reg, open('height_predictor/saved_models/nn_lm.sav', 'wb'))
# # save the scaler
# pickle.dump(scaler, open('height_predictor/saved_models/scaler_lm.pkl', 'wb'))
# %%
################################################
#               FINAL PIPELINE                 #
################################################
# %%

# Import actual tree data from Sappi
tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
####### CHANGE FROM VAL TO TEST
tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(tree_actual_df_path, 'TEST')
# tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(tree_actual_df_path, 'VAL')
# def actual_tree_data(tree_actual_df_path, sheet_name):
    
# Clipped CHM path
# chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
# chm_train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
chm_test_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
# chm_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_val_clip.tif")

# Clipped ortho path
# ortho_name = 'ortho_test_clip.tif'
ortho_name = 'ortho_test_clip.tif'
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
ortho_clipped_path = ortho_clipped_root + '/' + ortho_name

# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y

run_number = 'TEST_2'
# %%
final_results_df.loc[pipe_idx, 'test_name'] 
# %%
import warnings
warnings.filterwarnings("ignore")
import time
pipe_idx = 0
final_results_df = pd.DataFrame(columns=['test_name','patch_size', 'patch_overlap', 'iou_threshold', 'thresh', 'min_distance', 'dec_func',\
                                         'min_dist_coef', 'dec_func_diff_thresh','total_trees_lm','total_trees_df', 'unalloc_lm', 'unalloc_df', 'dead_lm', \
                                         'dead_df','max_dist_coef', 'rem_class_lm', 'rem_class_df','rem_qual_lm', 'rem_qual_df',\
                                         'mae_pos_lm', 'mae_pos_df','mae_pos_final', 'mae_pos_perc_orig',\
                                         'mae_height_lm', 'mae_height_df','mae_height_final', 'mae_height_perc_orig',\
                                         'rmse_height_lm', 'rmse_height_df','rmse_height_final', 'rmse_height_perc_orig',\
                                         'r2_height_lm', 'r2_height_df','r2_height_final', 'r2_height_perc_orig', 'final_trees_pred', 'perc_trees_pred',\
                                         'final_dead', 'final_unalloc','duration'])

# patch_sizes = [850, 950]
patch_sizes = [850]
patch_size=850
# patch_overlaps = [0.3, 0.5]
window_sizes = [24]
patch_overlaps = [0.4]
iou_thresholds = [0.3]
threshs = [0.5]
thresh = 0.5
# threshs = [0.4]
dec_funcs = [0.6]
dec_func_diff_threshs = [0.3]
min_dist_coefs = [1.8]
max_dist_coefs = [1]
patch_overlap = 0.4

num_of_tests = len(min_dist_coefs)*len(max_dist_coefs)*len(patch_sizes)*len(window_sizes)*len(patch_overlaps)*len(threshs)*len(dec_funcs)*len(dec_func_diff_threshs)

for iou_threshold in iou_thresholds:
    for min_dist_coef in min_dist_coefs:
        for max_dist_coef in max_dist_coefs: 
            for dec_func in dec_funcs:
                for window_size in window_sizes:
                    for dec_func_diff_thresh in dec_func_diff_threshs:

                        # Start Timer
                        start = time.time()
                        print("test number ", pipe_idx+1,  " started.")

                        patch_size = patch_size
                        patch_overlap = patch_overlap
                        dec_func_diff_thresh = dec_func_diff_thresh
                        thresh = thresh
                        iou_threshold = iou_threshold

                        window_size = 24
                        dec_func = dec_func


                        test_name = run_number + '-' + str(pipe_idx)

                        final_results_df.loc[pipe_idx, 'test_name'] = test_name
                        final_results_df.loc[pipe_idx, 'patch_size'] = patch_size
                        final_results_df.loc[pipe_idx, 'patch_overlap'] = patch_overlap
                        final_results_df.loc[pipe_idx, 'iou_threshold'] = iou_threshold
                        final_results_df.loc[pipe_idx, 'thresh'] = thresh
                        final_results_df.loc[pipe_idx, 'min_distance'] = window_size
                        final_results_df.loc[pipe_idx, 'dec_func'] = dec_func
                        final_results_df.loc[pipe_idx, 'min_dist_coef'] = min_dist_coef
                        final_results_df.loc[pipe_idx, 'max_dist_coef'] = max_dist_coef
                        final_results_df.loc[pipe_idx, 'dec_func_diff_thresh'] = dec_func_diff_thresh

                        # Load model
                        model = main.deepforest()
                        # model.use_release()
                        print("Current device is {}".format(model.device))
                        model.to("cuda")
                        print("Current device is {}".format(model.device))
                        model.config["gpus"] = 1
                        model.config["nms_thresh"] = iou_threshold

                        model_path = 'df_models/final_model.pt'
                        model.model.load_state_dict(torch.load(model_path))

                        # Get tree positions from DeepForest
                        predictions_df, predictions_df_transform, results_df_df, predicted_raster_image = deep_forest_pred(ortho_name, ortho_clipped_path, ortho_clipped_root, tree_point_calc_csv_path=tree_point_calc_csv_path, tree_point_calc_shifted_csv_path=tree_point_calc_shifted_csv_path, tree_actual_df=tree_actual_df, tree_actual_df_no_dead = tree_actual_df_no_dead, patch_size=patch_size, patch_overlap=patch_overlap, thresh=thresh, iou_threshold=iou_threshold,shape_fig_name=test_name, save_fig = True, save_shape = True)

                        # Get tree positions from LocalMaxima
                        df_global_gp, tree_positions_from_lm, results_df_2_lm, tree_locations_pred_df_lm = local_maxima_func(chm_test_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df, window_size=window_size, min_height=min_height, shape_fig_name=test_name, save_shape_file=True)

                        total_trees_LM = results_df_2_lm.loc[0,'number_trees_pred']
                        total_trees_DF = results_df_df.loc[0,'number_trees_pred']
                        total_dead_trees_LM = results_df_2_lm.loc[0,'number_dead_pred']
                        total_dead_trees_DF = results_df_df.loc[0,'number_dead_pred']
                        total_unallocated_LM = results_df_2_lm.loc[0,'number_unallocated']
                        total_unallocated_DF = results_df_df.loc[0,'number_unallocated']
                        # print('total trees DF: ',total_trees_DF, '\ntotal dead trees DF: ', total_dead_trees_DF, '\ntotal_unallocated DF: ', total_unallocated_DF,'\n')
                        # print('total trees LM: ',total_trees_LM, '\ntotal dead trees LM: ', total_dead_trees_LM, '\ntotal_unallocated LM: ', total_unallocated_LM)

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

                        expansion_size = 25
                        expansion_size_metres = expansion_size*0.0317*1.1

                        all_trees_pred_boxes = boxes_from_points(all_trees_pred, expansion_size_metres)

                        ortho_cropped_for_cropping = rxr.open_rasterio(ortho_clipped_path, masked=True).squeeze()
                        # Crop all tree positions to create dataset on which classifications can be carried out on
                        crop_df = crop_pixels_to_df(ortho_cropped_for_cropping, all_trees_pred_boxes, expansion_size = expansion_size, save_crops = False, train=False)

                        # Run classifier and filter trees
                        model_path = 'tree_classifier_crops/saved_models/svm_v2.sav'
                        all_trees_pred_after_removal_LM, all_trees_pred_after_removal_DF, removed_LM, removed_DF = tree_classifier(all_trees_pred, crop_df, model_path)

                        final_results_df.loc[pipe_idx,'rem_class_lm'] = removed_LM.shape[0]
                        final_results_df.loc[pipe_idx,'rem_class_df'] = removed_DF.shape[0]

                        # Obtain tree neighbour stats
                        max_dist_lm, min_dist_lm, avg_dist_lm, sd_dist_lm = tree_neighbour_stats(all_trees_pred_after_removal_LM)
                        max_dist_df, min_dist_df, avg_dist_df, sd_dist_df = tree_neighbour_stats(all_trees_pred_after_removal_DF)

                        # Obtain tree point quality and filter
                        all_trees_pred_after_removal_LM_2, num_trees_removed_LM, perc_trees_removed_LM = tree_point_quality(all_trees_pred_after_removal_LM, max_dist_lm, min_dist_lm, dec_func=dec_func, min_dist_coef=min_dist_coef, max_dist_coef=max_dist_coef)
                        all_trees_pred_after_removal_DF_2, num_trees_removed_DF, perc_trees_removed_DF = tree_point_quality(all_trees_pred_after_removal_DF, max_dist_df, min_dist_df, dec_func=dec_func, min_dist_coef=min_dist_coef, max_dist_coef=max_dist_coef)

                        final_results_df.loc[pipe_idx,'rem_qual_lm'] = num_trees_removed_LM
                        final_results_df.loc[pipe_idx,'rem_qual_df'] = num_trees_removed_LM

                        geoms = [] 
                        for idx in all_trees_pred_after_removal_DF_2.index:

                            box_list = list(all_trees_pred_after_removal_DF_2.loc[idx,['xmin_box', 'ymin_box', 'xmax_box','ymax_box']])
                            geom = Polygon([[box_list[0], box_list[1]], [box_list[0],box_list[3]], [box_list[2],box_list[3]], [box_list[2],box_list[1]]])
                            geoms.append(geom)

                        all_trees_pred_after_removal_DF_2['geometry'] = geoms

                        DF_predictions_final, LM_predictions_final, total_trees, df_idx_to_keep, lm_idx_to_keep = tree_selection(all_trees_pred_after_removal_LM_2, all_trees_pred_after_removal_DF_2, dec_func_diff_thresh = dec_func_diff_thresh)

                        all_predictions_final = pd.concat([DF_predictions_final[['X', 'Y', 'tree_id_pred']], LM_predictions_final[['X', 'Y', 'tree_id_pred']]])
                        all_predictions_final_gp = all_predictions_final.copy()
                        all_predictions_final_gp['geometry'] = all_predictions_final_gp.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)
                        all_predictions_final_gp = gpd.GeoDataFrame(all_predictions_final_gp, geometry='geometry').reset_index(drop=True)
                        shape_file_path= 'results/shape_files/' + test_name + 'final_tree_positions.shp'
                        all_predictions_final_gp.to_file(shape_file_path, driver='ESRI Shapefile')

                        nn_list_final = []
                        for idx in all_predictions_final_gp.index:
                            current_tree = (all_predictions_final_gp.loc[idx, 'X'], all_predictions_final_gp.loc[idx, 'Y'])
                            distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
                            nn_list_final.append(distance_to_nn)

                        chm_test = rxr.open_rasterio(chm_test_clip_path, masked=True).squeeze()

                        # Calculate the perimeter and area of bounding boxes
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

                        ################# UNCOMMENT WHEN RUNNING FULL PIPELINE ###############################
                        DF_predictions_final['area'] = (np.sqrt(np.array(DF_predictions_final['area']))/0.0317)**2

                        # features_names = ['area', 'avg_chm', 'max_chm', 'std_chm']
                        features_names = ['area', 'avg_chm', 'min_chm','max_chm', 'std_chm']
                        # Load DF scaler
                        scaler = pickle.load(open('height_predictor/saved_models/scaler.pkl', 'rb'))
                        X_scaled = pd.DataFrame()
                        X_scaled[features_names] = scaler.transform(DF_predictions_final[features_names])

                        # Load LM scaler
                        scaler_lm = pickle.load(open('height_predictor/saved_models/scaler_lm.pkl', 'rb'))
                        X_scaled_lm = scaler_lm.transform(np.array(LM_predictions_final['height_pred']).reshape(-1, 1))

                        # Load models
                        height_predictor = pickle.load(open('height_predictor/saved_models/nn_7.sav',"rb"))
                        height_predictor_lm = pickle.load(open('height_predictor/saved_models/nn_lm.sav',"rb"))

                        # Make height predictions
                        predictions_df = height_predictor.predict(X_scaled)
                        predictions_lm = height_predictor_lm.predict(X_scaled_lm)

                        DF_predictions_final_incl_actual = DF_predictions_final.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
                        DF_predictions_final_incl_actual['height_pred_model'] = predictions_df

                        LM_predictions_final_incl_actual = LM_predictions_final.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
                        LM_predictions_final_incl_actual['height_pred_model'] = predictions_lm

                        columns_to_keep = ['X', 'Y', 'tree_id_pred', 'height_pred', 'height_pred_model','Hgt22Rod']
                        predictions_final = pd.concat([DF_predictions_final_incl_actual[columns_to_keep], LM_predictions_final_incl_actual[columns_to_keep]]).reset_index(drop=True)

                        # score_df_LM = regression_scores(LM_predictions_final_incl_actual['Hgt22Rod'], LM_predictions_final_incl_actual['height_pred'], model='LM')
                        DF_score_df = DF_predictions_final_incl_actual[(DF_predictions_final_incl_actual['Hgt22Rod'].isna()!=True) & (DF_predictions_final_incl_actual['height_pred_model'].isna()!=True)]
                        LM_score_df = LM_predictions_final_incl_actual[(LM_predictions_final_incl_actual['Hgt22Rod'].isna() != True) & (LM_predictions_final_incl_actual['height_pred_model'].isna() != True)]
                        final_score_df = pd.concat([DF_score_df[columns_to_keep], LM_score_df[columns_to_keep]]).reset_index(drop=True)
                        score_df_DF = regression_scores(DF_score_df['Hgt22Rod'], DF_score_df['height_pred_model'], model='height_predictor')
                        score_df_final = regression_scores(final_score_df['Hgt22Rod'], final_score_df['height_pred_model'], model='height_predictor_lm')

                        # lm_mae_pos = np.round(results_df_lm.loc[0,'MAE_position'],4)
                        lm_mae_pos = 0.3515
                        df_mae_pos = np.round(results_df_df.loc[0,'MAE_position'],4)
                        final_mae_pos = np.round(sum(nn_list_final)/len(nn_list_final),4)
                        final_mae_pos_perc = np.round(((lm_mae_pos-final_mae_pos)/lm_mae_pos)*100,2)

                        # lm_mae_height = np.round(score_df_LM.loc[0,'mae'],4)
                        lm_mae_height = 0.4949
                        df_mae_height = np.round(score_df_DF.loc[0,'mae'],4)
                        final_mae_height = np.round(score_df_final.loc[0,'mae'],4)
                        final_mae_height_perc = np.round(((lm_mae_height-final_mae_height)/lm_mae_height)*100,2)

                        # lm_rmse_height = np.round(score_df_LM.loc[0,'rmse'],4)
                        lm_rmse_height = 0.6435
                        df_rmse_height = np.round(score_df_DF.loc[0,'rmse'],4)
                        final_rmse_height = np.round(score_df_final.loc[0,'rmse'],4)
                        final_rmse_height_perc = np.round(((lm_rmse_height-final_rmse_height)/lm_rmse_height)*100,2)

                        # lm_r2_height = np.round(score_df_LM.loc[0,'r2'],4)
                        lm_r2_height = 0.6662
                        df_r2_height = np.round(score_df_DF.loc[0,'r2'],4)
                        final_r2_height = np.round(score_df_final.loc[0,'r2'],4)
                        final_r2_height_perc = np.round(((lm_r2_height-final_r2_height)/lm_r2_height)*100,2)

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


                        for idx in range(len(predictions_final)):

                            current_tree = (predictions_final.loc[idx, 'X'], predictions_final.loc[idx, 'Y'])
                            distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
                            predictions_final.loc[idx, 'tree_id_pred'] = tree_id
                            predictions_final.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
                            predictions_final.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

                        ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_final)

                        # Merge with actual data to determine number of dead trees predicted
                        tree_actual_df_no_dead = tree_actual_df[tree_actual_df['24_Def'] != 'D']
                        tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
                        results_idx = 0
                        results_df = pd.DataFrame()

                        # print('number_trees_pred:  ', tree_locations_pred_df.shape[0])
                        # print('number_unallocated: ', tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0])
                        # print('number_dead_pred:   ', tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0])
                        # print('perc_trees_pred:    ', tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0])

                        final_results_df.loc[pipe_idx,'final_trees_pred'] = tree_locations_pred_df.shape[0]
                        final_results_df.loc[pipe_idx,'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
                        final_results_df.loc[pipe_idx,'final_dead'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
                        final_results_df.loc[pipe_idx,'final_unalloc'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]

                        end = time.time()
                        final_results_df.loc[pipe_idx,'duration'] = round((end - start)/60, 2)

                        csv_file_path = 'final_results/' + test_name + '.csv'
                        final_results_df.to_csv(csv_file_path)
                        
                        pipe_idx += 1
                        print("test number ", pipe_idx, " of ", num_of_tests, " completed in ", round((end - start)/60, 2), "min")
# %%
# df_global_gp, tree_positions_from_lm, results_df_2_lm, tree_locations_pred_df_lm = local_maxima_func(chm_test_clip_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df, window_size=window_size, min_height=min_height, shape_fig_name=test_name, save_shape_file=False)
plot_df = predictions_final[predictions_final['Hgt22Rod'].isna()!=True]
# %%
plot_df = tree_locations_pred_df_lm[tree_locations_pred_df_lm['Hgt22Rod'].isna()==False]
# %%
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 7))

# sns.set_style("whitegrid")
# sns.set_theme(style='whitegrid', font={'font.family':'serif', 'font.serif':'Computer Modern Roman'})
sns.set_theme(style="whitegrid", font='cmr10',font_scale=1.2)
# sns.set_style("ticks")
# sns.set_theme(style="white")

ax = sns.regplot(data=plot_df, x='Hgt22Rod', y='height_pred',fit_reg=True)
ax.set_xlabel('Measured height (m)', labelpad =5, fontsize = 15, **{'fontfamily':'serif'})
ax.set_ylabel('Estimated height (m)', labelpad =5, fontsize = 17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# ax.grid(axis='x')
# ax.grid(axis='y')
ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
# plt.ylim([1138, 1154])
# plt.xlim([0, 18])
plt.savefig('../Main/images/plot_lm_res.png',bbox_inches = 'tight', dpi=300)
plt.show()
# %%
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(np.array(plot_df['height_pred']).reshape(-1, 1), plot_df['Hgt22Rod'])
# reg.score(X, y)

print(reg.intercept_)
print(reg.coef_)
# %%
all_trees_pred_after_removal_DF[all_trees_pred_after_removal_DF['dec_func']<0.35].shape[0]
# %%
test = tree_locations_pred_df_lm[tree_locations_pred_df_lm['Hgt22Rod'].isna()!=True]
actual = test['Hgt22Rod']
pred = test['height_pred']
np.corrcoef(actual,pred)
# %%
tree_locations_pred_df_lm.head()
# %%
total_trees_LM = results_df_lm.iloc[0]['number_trees_pred']
total_trees_DF = results_df_df.iloc[0]['number_trees_pred']
total_dead_trees_LM = results_df_lm.iloc[0]['number_dead_pred']
total_dead_trees_DF = results_df_df.iloc[0]['number_dead_pred']
total_unallocated_LM = results_df_lm.iloc[0]['number_unallocated']
total_unallocated_DF = results_df_df.iloc[0]['number_unallocated']
# %%

# %%

print('LM MAE in position      : ', lm_mae_pos, 'm')
print('DF MAE in position      : ', df_mae_pos, 'm')
print('Final MAE in position   : ', final_mae_pos, 'm')
print('Final MAE in position % : ', final_mae_pos_perc, '%\n')

print('LM MAE in height        : ', lm_mae_height, 'm')
print('DF MAE in height        : ', df_mae_height, 'm')
print('Final MAE in height     : ', final_mae_height, 'm')
print('Final MAE in height %   : ', final_mae_height_perc, '%\n')

print('LM RMSE in height        : ', lm_rmse_height, 'm')
print('DF RMSE in height        : ', df_rmse_height, 'm')
print('Final RMSE in height     : ', final_rmse_height, 'm')
print('Final RMSE in height %   : ', final_rmse_height_perc, '%\n')

print('LM R^2                   : ', lm_r2_height)
print('DF R^2                   : ', df_r2_height)
print('Final R^2                : ', final_r2_height)
print('Final R^2 %              : ', final_r2_height_perc, '%')
# %%

# %%
end = time.time()
int_time = round((time.time() - start)/60, 2)
print("Total time: ", round((time.time() - start)/60 - int_time, 2), "min")
# %%
# Get crops of all bounding boxes
crop_df_bounding_boxes = crop_array(ortho_cropped_for_cropping, DF_predictions_final, save_crops = False)
# %%

# %%
print('Min: ', clip.std().values)
# %%
print('Mean:', sjer_chm_data.mean().values)
print('Max:', sjer_chm_data.max().values)
print('Min:', sjer_chm_data.min().values)
# %%

# %%

# %%
# %%
# import math
# for i in predictions_final.index:
#     tree_pred = predictions_final.loc[i, 'tree_id_pred']
#     tree_pred_pos = np.asarray(predictions_final.loc[i, ['X', 'Y']])
#     tree_true_pos = np.asarray(tree_point_calc_shifted_df[tree_point_calc_shifted_df['tree_id'] == tree_pred][['X', 'Y']])[0]
#     # tree_1 = np.asarray(tree_locations_df[['X', 'Y']])
#     # distances = cdist([current_tree], trees, 'euclidean').T
#     # tree = np.array(tree).reshape(-1, 1)
#     predictions_final.loc[i,'distance_nn'] = math.dist(tree_pred_pos, tree_true_pos)
#     # predictions_final.loc[i,'distance_nn'] = euclidean_distances(tree_pred_pos, tree_true_pos)
#     # nn_idx = distances.argmin()
#     # distance_to_nn = distances.T[nn_idx][0]
#     predictions_final.loc[i,'distance_nn_squared'] = (predictions_final.loc[i,'distance_nn'])**2
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
all_trees_pred_after_removal_LM_2.columns
# %%
all_trees_pred_after_removal_DF_2.columns
# %%
 # Read in dataframe of all trees (shifted)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['tree_easting'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['tree_northing'] = tree_point_calc_shifted_df['geometry'].y
tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')

patch_size = 950
patch_overlap = 0.5
thresh = 0.5
iou_threshold = 0.05

results_df = pd.DataFrame()
results_idx = 0
# Create Predictions
predictions_df = model.predict_tile(image=img_bgr, patch_size=patch_size, patch_overlap=patch_overlap, iou_threshold=iou_threshold)
predictions_df = predictions_df[predictions_df['score'] > thresh]
print(f"{predictions_df.shape[0]} predictions kept after applying threshold")
df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + '_thresh-' + str(thresh)  + '_iou-' + str(iou_threshold) + '.png'
predicted_raster_image = plot_predictions_from_df(predictions_df, img_bgr)
plt.imsave(df_image_save_path,arr=predicted_raster_image)

# Transform predictions to original CRS
predictions_df_transform = predictions_df.copy()
predictions_df_transform['image_path'] = "ortho_corrected_no_compression_clipped.tif"
predictions_df_transform = predictions_df_transform[['xmin', 'ymin', 'xmax', 'ymax','image_path']]
predictions_df_transform = utilities.project_boxes(predictions_df_transform, root_dir=ortho_clipped_root, transform=True)

predictions_df_transform['X'] = predictions_df_transform['xmin'] + (predictions_df_transform['xmax'] - predictions_df_transform['xmin'])/2
predictions_df_transform['Y'] = predictions_df_transform['ymin'] + (predictions_df_transform['ymax'] - predictions_df_transform['ymin'])/2
predictions_df_transform['geometry'] = predictions_df_transform.apply(lambda x: Point((float(x.X), float(x.Y))), axis=1)

# shape_file_name = 'deepforest_predictions/shapefiles/' + str(results_idx) + '_patch_size-' + str(patch_size) + '_patch_overlap-' + str(patch_overlap) + 'thresh' + str(thresh)  + 'iou' + str(iou_threshold) + '.shp'
# predictions_df_transform.to_file(shape_file_name, driver='ESRI Shapefile')

clear_flag = 0
for idx in range(len(predictions_df_transform)):

    if idx == range(len(predictions_df_transform))[-1]: clear_flag = 1

    current_tree = (predictions_df_transform.loc[idx, 'X'], predictions_df_transform.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df, clear_flag)
    predictions_df_transform.loc[idx, 'tree_id_pred'] = tree_id
    predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
    predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

    del distance_to_nn
    del distance_to_nn_squared
    del tree_id

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
print('test ' + str(results_idx) + ' done')
# %%

# %%

# %%
test_clip = np.array(test_clip)
test_clip = test_clip.astype('uint8')
test_clip_rgb = np.moveaxis(test_clip, 0, 2).copy()
plt.imshow(test_clip_rgb)
plt.imsave('test.png',arr=test_clip_rgb)
# %%
test_clip_rgb
# %%
img_rgb
# %%
# 
# %%
################################################
#         BUILD DEAD TREE CLASSIFIER           #
################################################
# %%
# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'

# Clipped CHM path
# chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
chm_train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
chm_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_val_clip.tif")

# Clipped ortho path
ortho_name_train = 'ortho_train_clip.tif'
ortho_name_val = 'ortho_val_clip.tif'
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
ortho_clipped_path_train = ortho_clipped_root + '/' + ortho_name_train
ortho_clipped_path_val = ortho_clipped_root + '/' + ortho_name_val
# %%
# Import actual tree data from Sappi
tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df_train, tree_actual_df_full_train, tree_actual_df_no_dead_train, min_height_train = actual_tree_data(tree_actual_df_path, 'TRAIN')
tree_actual_df_val, tree_actual_df_full_val, tree_actual_df_no_dead_val, min_height_val = actual_tree_data(tree_actual_df_path, 'VAL')
# %%
# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y

#%%

patch_size = 850
patch_overlap = 0.5
thresh = 0.4
iou_threshold = 0.05

# Load model
model = main.deepforest()
model.use_release()
print("Current device is {}".format(model.device))
model.to("cuda")
print("Current device is {}".format(model.device))
model.config["gpus"] = 1

model_path = 'df_models/final_model.pt'
model.model.load_state_dict(torch.load(model_path))

# Get tree positions from DeepForest
predictions_df, predictions_df_transform, results_df_df, predicted_raster_image = deep_forest_pred(ortho_name, ortho_clipped_path, ortho_clipped_root, tree_point_calc_csv_path=tree_point_calc_csv_path, tree_point_calc_shifted_csv_path=tree_point_calc_shifted_csv_path, tree_actual_df=tree_actual_df_train, tree_actual_df_no_dead = tree_actual_df_no_dead_train, patch_size=patch_size, patch_overlap=patch_overlap, thresh=thresh, iou_threshold=iou_threshold, save_fig = True, save_shape = True)
# %%
predictions_df
# %%
# Get tree positions from LocalMaxima
# Clipped CHM path
# chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_2_clipped.tif")
# df_global_gp, tree_positions_from_lm = local_maxima_func(chm_clipped_path, tree_actual_df, window_size=29, min_height=min_height, save_shape_file=True)
# %%
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y
# %%
# Get average width, height and side of bounding box
# average_width, average_height, average_side = get_average_box_size(predictions_df_transform)

# expansion_factor = 0.5
# expansion_size = average_side * expansion_factor
expansion_size = 36
expansion_size_metres = expansion_size*0.0317*1.1
# expansion_size_metres

dead_trees_classified_train, non_dead_tree_classified_train  = dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df_train, save_crops=False)
classification_all_train = pd.concat([dead_trees_classified_train[['X', 'Y', 'class']], non_dead_tree_classified_train[['X', 'Y', 'class']]], ignore_index=True)
classification_all_train = boxes_from_points(classification_all_train, expansion_size_metres)

dead_trees_classified_val, non_dead_tree_classified_val  = dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df_val, save_crops=False)
classification_all_val = pd.concat([dead_trees_classified_val[['X', 'Y', 'class']], non_dead_tree_classified_val[['X', 'Y', 'class']]], ignore_index=True)
classification_all_val = boxes_from_points(classification_all_val, expansion_size_metres)

tree_actual_df_test, tree_actual_df_full_test, tree_actual_df_no_dead_test, min_height_test = actual_tree_data(tree_actual_df_path, 'TEST')
dead_trees_classified_test, non_dead_tree_classified_test  = dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df_test, save_crops=False)
classification_all_test = pd.concat([dead_trees_classified_test[['X', 'Y', 'class']], non_dead_tree_classified_test[['X', 'Y', 'class']]], ignore_index=True)
classification_all_test = boxes_from_points(classification_all_test, expansion_size_metres)

ortho_clipped_path_test = "ortho & pointcloud gen/outputs/GT/ortho_test_clip.tif"

ortho_cropped_for_cropping_train = rxr.open_rasterio(ortho_clipped_path_train, masked=True).squeeze()
ortho_cropped_for_cropping_val = rxr.open_rasterio(ortho_clipped_path_val, masked=True).squeeze()
ortho_cropped_for_cropping_test = rxr.open_rasterio(ortho_clipped_path_test, masked=True).squeeze()

# crop_df_train_36 = crop_pixels_to_df(ortho_cropped_for_cropping_train, classification_all_train, expansion_size = expansion_size, save_crops = False)
crop_df_val_36 = crop_pixels_to_df(ortho_cropped_for_cropping_val, classification_all_val, expansion_size = expansion_size, save_crops = False)
crop_df_test_36 = crop_pixels_to_df(ortho_cropped_for_cropping_test, classification_all_test, expansion_size = expansion_size, save_crops = False)
# %%
# # %%
# ortho_cropped_for_cropping_train = rxr.open_rasterio(ortho_clipped_path_train, masked=True).squeeze()
# ortho_cropped_for_cropping_val = rxr.open_rasterio(ortho_clipped_path_val, masked=True).squeeze()
# # Crop all trees and dataframe of flattened pixels
# # crop_df_train = crop_pixels_to_df(ortho_cropped_for_cropping_train, classification_all_train, save_crops = False)
# # crop_df_val = crop_pixels_to_df(ortho_cropped_for_cropping_val, classification_all_val, save_crops = False)
# crop_df_train_25 = crop_pixels_to_df(ortho_cropped_for_cropping_train, classification_all_train, expansion_size = expansion_size, save_crops = False)
# crop_df_val_25 = crop_pixels_to_df(ortho_cropped_for_cropping_val, classification_all_val, expansion_size = expansion_size, save_crops = False)
# %%
# %%
r_cols = list(range(0,1296))
g_cols = list(range(1296,1297+1296-1))
b_cols = list(range(1297+1296-1,3888))

train_dead_r = crop_df_train_36[crop_df_train_36['class']==1][r_cols].melt().reset_index(drop=True)
train_dead_g = crop_df_train_36[crop_df_train_36['class']==1][g_cols].melt().reset_index(drop=True)
train_dead_b = crop_df_train_36[crop_df_train_36['class']==1][b_cols].melt().reset_index(drop=True)
train_no_dead_r = crop_df_train_36[crop_df_train_36['class']==0][r_cols].melt().reset_index(drop=True)
train_no_dead_g = crop_df_train_36[crop_df_train_36['class']==0][g_cols].melt().reset_index(drop=True)
train_no_dead_b = crop_df_train_36[crop_df_train_36['class']==0][b_cols].melt().reset_index(drop=True)

dead_hist_df = pd.DataFrame()
dead_hist_df['Red'] = train_dead_r['value']
dead_hist_df['Green'] = train_dead_g['value']
dead_hist_df['Blue'] = train_dead_b['value']
dead_hist_df = dead_hist_df.melt()
dead_hist_df = dead_hist_df.rename(columns={'variable':'Channel'})

no_dead_hist_df = pd.DataFrame()
no_dead_hist_df['Red'] = train_no_dead_r['value']
no_dead_hist_df['Green'] = train_no_dead_g['value']
no_dead_hist_df['Blue'] = train_no_dead_b['value']
no_dead_hist_df = no_dead_hist_df.melt()
no_dead_hist_df = no_dead_hist_df.rename(columns={'variable':'Channel'})

# %%
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid", font='cmr10',font_scale=1.2)
ax = sns.histplot(data=dead_hist_df, x="value", hue="Channel",stat='density', bins=30,element="poly",hue_order=['Red','Green','Blue'],palette=['tomato', 'lightgreen', 'skyblue'])
ax.set_xlabel('Pixel intensity', labelpad =5, fontsize = 17, **{'fontfamily':'serif'})
ax.set_ylabel('Density', labelpad =5, fontsize = 19)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('../Main/images/dead_rgb_hist.png',bbox_inches = 'tight', dpi=300)
plt.show()
# %%

# %%
# %%
# crop_df_train_16['avg_r'] = crop_df_train_16[r_cols].mean(axis=1)
# crop_df_train_16['avg_g'] = crop_df_train_16[g_cols].mean(axis=1)
# crop_df_train_16['avg_b'] = crop_df_train_16[b_cols].mean(axis=1)
# crop_df_train_16['sd_r'] = crop_df_train_16[r_cols].std(axis=1)
# crop_df_train_16['sd_g'] = crop_df_train_16[g_cols].std(axis=1)
# crop_df_train_16['sd_b'] = crop_df_train_16[b_cols].std(axis=1)

# crop_df_val_16['avg_r'] = crop_df_val_16[r_cols].mean(axis=1)
# crop_df_val_16['avg_g'] = crop_df_val_16[g_cols].mean(axis=1)
# crop_df_val_16['avg_b'] = crop_df_val_16[b_cols].mean(axis=1)
# crop_df_val_16['sd_r'] = crop_df_val_16[r_cols].std(axis=1)
# crop_df_val_16['sd_g'] = crop_df_val_16[g_cols].std(axis=1)
# crop_df_val_16['sd_b'] = crop_df_val_16[b_cols].std(axis=1)

# %%
x_train = crop_df_train_25.drop(columns=['class'],axis=1)
x_test = crop_df_val_25.drop(columns=['class'],axis=1)
y_train = crop_df_train_25['class']
y_test = crop_df_val_25['class']
all_x = pd.concat([x_test,x_train])
# %%
y_test.shape[0] - y_test.sum()
# %%
print('Shape of train living class', y_train.sum())
print('Shape of train dead class', y_train.shape[0]-y_train.sum())
print('Shape of test living class', y_test.sum())
print('Shape of test dead class', y_test.shape[0]-y_test.sum())
# %%
# %%
x_train = x_train/255
x_test = x_train/255
# %%
scores_df_all_tests = pd.DataFrame(columns=['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'model'])
test_counter = 0
num_of_tests_for_avg = 10
# expansion_sizes = [10, 15, 20, 25, 30]

# for expansion_size in expansion_sizes:

#     expansion_size_metres = expansion_size*0.0317*1.1
#     # expansion_size_metres

#     dead_trees_classified_train, non_dead_tree_classified_train  = dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df_train, save_crops=False)
#     classification_all_train = pd.concat([dead_trees_classified_train[['X', 'Y', 'class']], non_dead_tree_classified_train[['X', 'Y', 'class']]], ignore_index=True)
#     classification_all_train = boxes_from_points(classification_all_train, expansion_size_metres)

#     dead_trees_classified_val, non_dead_tree_classified_val  = dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df_val, save_crops=False)
#     classification_all_val = pd.concat([dead_trees_classified_val[['X', 'Y', 'class']], non_dead_tree_classified_val[['X', 'Y', 'class']]], ignore_index=True)
#     classification_all_val = boxes_from_points(classification_all_val, expansion_size_metres)

#     ortho_cropped_for_cropping_train = rxr.open_rasterio(ortho_clipped_path_train, masked=True).squeeze()
#     ortho_cropped_for_cropping_val = rxr.open_rasterio(ortho_clipped_path_val, masked=True).squeeze()

#     crop_df_train = crop_pixels_to_df(ortho_cropped_for_cropping_train, classification_all_train, expansion_size = expansion_size, save_crops = False)
#     crop_df_val = crop_pixels_to_df(ortho_cropped_for_cropping_val, classification_all_val, expansion_size = expansion_size, save_crops = False)

#     x_train = crop_df_train.drop(columns='class')
#     x_test = crop_df_val.drop(columns='class')
#     y_train = crop_df_train['class']
#     y_test = crop_df_val['class']

# Scaler
# x_train = x_train / 255
# x_test = x_test / 255

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
    # scores_df_all_models.loc[df_idx,'feature_list'] = features_test

    scores_df_all_tests = pd.concat([scores_df_all_tests, scores_df_all_models]).reset_index(drop=True)

    test_counter += 1

    if test_counter % 5 == 0:

        print(test_counter, ' tests of ', num_of_tests_for_avg*2, 'completed') 

scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc']] = scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc']].astype(float)
scores_df_all_tests_red = scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc', 'model']]
scores_df_all_tests_red = scores_df_all_tests_red.reset_index(drop=True)
scores_df_all_tests_avg = scores_df_all_tests_red.groupby(['model'], as_index=False).mean()

scores_df_all_tests.to_csv('tree_classifier_crops/classifier_scores/model_tests_avg_std_v2.csv')
scores_df_all_tests_avg.to_csv('tree_classifier_crops/classifier_scores/model_tests_average_score_avg_std_v2.csv')
# %%
# scores_df_all_tests.to_csv('tree_classifier_crops/classifier_scores/model_tests_all_scaled_v3_with_feature_combinations.csv')
scores_df_all_tests.to_csv('tree_classifier_crops/classifier_scores/model_tests_all_pixels_16.csv')
scores_df_all_tests_avg.to_csv('tree_classifier_crops/classifier_scores/model_tests_average_score_all_pixels_16.csv')
# %%
tree_point_calc_shifted_df
# %%
cs = [0.5, 0.7, 1, 1.5]
# kernels = ['rbf', 'linear','poly', 'sigmoid']
kernels = ['rbf']
degrees = [1]
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
            

svm_parameter_search.to_csv('tree_classifier_crops/classifier_scores/svm_scores_VAL_v4.csv')
# %%
# %%
x_train = crop_df_train_36.drop(columns=['class'],axis=1)
# x_test = crop_df_val_36.drop(columns=['class'],axis=1)
# x_test_test = crop_df_test_36.drop(columns=['class'],axis=1)
# all_x = pd.concat([x_test,x_train])

y_train = crop_df_train_36['class']
# y_test = crop_df_val_36['class']
# y_test_test = crop_df_test_36['class']

print('Shape of train living class', y_train.sum())
print('Shape of train dead class', y_train.shape[0]-y_train.sum())
# print('Shape of val living class', y_test.sum())
# print('Shape of val dead class', y_test.shape[0]-y_test.sum())
# print('Shape of test living class', y_test_test.sum())
# print('Shape of test dead class', y_test_test.shape[0]-y_test_test.sum())
# %%
# %%
x_train = crop_df_train_36.drop(columns=['class'],axis=1)
x_test = crop_df_val_36.drop(columns=['class'],axis=1)
x_test_test = crop_df_test_36.drop(columns=['class'],axis=1)
y_train = crop_df_train_36['class']
y_test = crop_df_val_36['class']
y_test_test = crop_df_test_36['class']
all_x = pd.concat([x_test,x_train])
x_train = x_train / 255
x_test = x_test / 255
x_test_test = x_test_test / 255

c = 0.5
# kernels = ['rbf', 'linear','poly', 'sigmoid']
kernel = 'rbf'
gamma = 'scale'

svm_parameter_search = pd.DataFrame(columns=['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc'])

svm_model = svm.SVC(C=c, kernel=kernel, gamma=gamma, probability=True).fit(x_train, np.ravel(y_train))

predictions_svm_val = svm_model.predict(x_test)
predictions_svm_val_prob = svm_model.predict_proba(x_test)

predictions_svm_train = svm_model.predict(x_train)
predictions_svm_train_prob = svm_model.predict_proba(x_train)

predictions_svm_test_test = svm_model.predict(x_test_test)
predictions_svm_test_test_prob = svm_model.predict_proba(x_test_test)

score_df_svm_val = classification_scores(y_true=y_test, y_pred=predictions_svm_val, y_pred_prob=predictions_svm_val_prob[:, 1], model='svm')
score_df_svm_train = classification_scores(y_true=y_train, y_pred=predictions_svm_train, y_pred_prob=predictions_svm_train_prob[:, 1], model='svm')
score_df_svm_test_test= classification_scores(y_true=y_test_test, y_pred=predictions_svm_test_test, y_pred_prob=predictions_svm_test_test_prob[:, 1], model='svm')
svm_parameter_search.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_svm_train.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]
svm_parameter_search.loc[1,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_svm_val.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]
svm_parameter_search.loc[2,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_svm_test_test.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]
# svm_parameter_search.to_csv('tree_classifier_crops/classifier_scores/svm_scores_final.csv')
# %%
svm_model_filename = 'tree_classifier_crops/saved_models/svm_v2.sav'
pickle.dump(svm_model, open(svm_model_filename, 'wb'))
# %%
svm_model.predict_proba.probA_(x_train)
# %%
test_prob = predictions_svm_test_test_prob[:, 1]
# %%
# Minmax scaler
dec_func = svm_model.decision_function(x_test_test)
scaler = MinMaxScaler()
dec_func_scaled = scaler.fit_transform(dec_func.reshape(-1, 1))
dec_func_scaled
# %%
dec_func_proba_df = pd.DataFrame(columns=['Probability', 'Scaled decision function'])
dec_func_proba_df['Probability'] = test_prob
dec_func_proba_df['Scaled decision function'] = dec_func_scaled
dec_func_proba_df = dec_func_proba_df.melt()
dec_func_proba_df = dec_func_proba_df.rename(columns={'variable':'Method'})
# %%
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid", font='cmr10',font_scale=1.2)
ax = sns.histplot(data=dec_func_proba_df, x="value", hue="Method",stat='density', bins=20,element="step", legend=False)
ax.set_xlabel('Probability / scaled decision function', labelpad =5, fontsize = 17, **{'fontfamily':'serif'})
ax.set_ylabel('Density', labelpad =5, fontsize = 19)
ax.legend(labels=['Probability', 'Scaled decision function'], loc='upper center')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.ylim([0,11])
plt.savefig('../Main/images/proba_dec_func_hist.png',bbox_inches = 'tight', dpi=300)
plt.show()
# %%
# ############### NN Grid Search ###############
# solvers = ['adam', 'lbfgs', 'sgd']
# learning_rates = ['constant', 'invscaling', 'adaptive']
# # solver = 'adam'
# # learning_rate = 'invscaling'
# hl1s = [20, 50, 100, 200]
# hl2s = [50, 100, 200, 300]
# hl3s = [50, 100, 200, 300]

# # max_iters = [100, 200, 300, 500, 800, 1000, 2000]
# # alphas = [0.0001, 0.005, 0.001, 0.05, 0.01]

# grid_search_idx = 0
# nn_parameter_search = pd.DataFrame(columns=['arch', 'solver', 'learning_rate', 'accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc'])
# nn_parameter_search['arch'] = nn_parameter_search['arch'].astype(object)

# for hl1 in hl1s:
#     for hl2 in hl2s:
#         for hl3 in hl3s:
#             for solver in solvers:
#                 for learning_rate in learning_rates:
#                     hidden_layer_sizes = (hl1, hl2, hl3)
#                     nn_reg = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver=solver, learning_rate=learning_rate, max_iter=1500, random_state = 42).fit(x_train, np.ravel(y_train))
#                     predictions_nn = nn_reg.predict(x_test)
#                     predictions_nn_prob = svm_model.predict_proba(x_test)
#                     score_df_nn = classification_scores(y_true=y_test, y_pred=predictions_nn, y_pred_prob=predictions_nn_prob[:, 1], model='nn')

#                     nn_parameter_search.loc[grid_search_idx,'arch'] = hidden_layer_sizes
#                     nn_parameter_search.loc[grid_search_idx,'solver'] = solver
#                     nn_parameter_search.loc[grid_search_idx,'learning_rate'] = learning_rate
#                     # nn_parameter_search.loc[grid_search_idx,'alpha'] = alpha
#                     # nn_parameter_search.loc[grid_search_idx,'activation'] = activation
#                     # nn_parameter_search.loc[grid_search_idx,'early_stopping'] = early_stopping
#                     # nn_parameter_search.loc[grid_search_idx,'max_iter'] = max_iter
#                     nn_parameter_search.loc[grid_search_idx,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc']] = score_df_nn.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc']]

#                     grid_search_idx += 1

#                     if grid_search_idx % 100 == 0: 
#                         print(grid_search_idx, " of ", len(hl1s)*len(hl2s)*len(hl3s), "tests completed")

# nn_parameter_search.to_csv('tree_classifier_crops/classifier_scores/nn_scores_VAL_v2.csv')
# # %%
# # Random Forest grid search
# feature_list_idxs = [42, 41, 43, 56]
# n_estimators = [10, 50, 100, 150, 200, 250, 300, 350]
# criterions = ['gini', 'entropy']
# max_depths = [None, 10, 20, 50, 100, 150, 200, 250]

# grid_search_idx = 0
# rf_parameter_search = pd.DataFrame(columns=['n_estimators', 'criteria', 'max_depth', 'accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'model'])
# for feature_list_idx in feature_list_idxs:

#     feature_list_for_test = feature_lists_df.loc[feature_list_idx,'features']
#     x_train, x_test, y_train, y_test = train_test_split(X_scaled[feature_list_for_test], Y, test_size = 0.25, stratify=Y, random_state=21)

#     for n_estimator in n_estimators:
#         for criterion in criterions:
#             for max_depth in max_depths:

#                 rf = RandomForestClassifier(n_estimators = n_estimator, max_depth=max_depth, criterion = criterion, random_state=42).fit(x_train, np.ravel(y_train))
#                 predictions_rf = rf.predict(x_test)
#                 predictions_rf_prob = rf.predict_proba(x_test)
#                 score_df_rf = classification_scores(y_true=y_test, y_pred=predictions_rf, y_pred_prob=predictions_rf_prob[:, 1], model='random forest',features=None, feature_list_id=feature_list_idx)

#                 rf_parameter_search.loc[grid_search_idx,'n_estimators'] = n_estimator
#                 rf_parameter_search.loc[grid_search_idx,'criteria'] = criterion
#                 rf_parameter_search.loc[grid_search_idx,'max_depth'] = max_depth
#                 rf_parameter_search.loc[grid_search_idx,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_rf.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]

#                 grid_search_idx += 1

#     print('feature list ', feature_list_idx, ' done') 
# rf_parameter_search.to_csv('tree_classifier_crops/classifier_scores/random_forest_scores_VAL.csv')
# # %%
# # Build and save model (n_estimators = 50, criterion = 'gini, max_depth=None)
# rf = RandomForestClassifier(n_estimators = 150, max_depth=10, criterion = 'entropy', random_state=42).fit(x_train, np.ravel(y_train))
# # rf_model_filename = 'tree_classifier_crops/saved_models/random_forest_v2.sav'
# # pickle.dump(rf, open(rf_model_filename, 'wb'))
# features_names = X_scaled.columns

# importances = rf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
# forest_importances = pd.Series(importances, index=features_names)

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# # %%
# # %%
# for feature_list_idx in feature_list_idxs:

#     feature_list_for_test = feature_lists_df.loc[56,'features']
#     x_train, x_test, y_train, y_test = train_test_split(X_scaled[feature_list_for_test], Y, test_size = 0.25, stratify=Y, random_state=42)
# svm_model = svm.SVC(C=0.5, kernel='poly', degree=5, probability=True).fit(x_train, np.ravel(y_train))
# svm_model_filename = 'tree_classifier_crops/saved_models/svm.sav'
# pickle.dump(svm_model, open(svm_model_filename, 'wb'))
# # %%
################################################
#          TEST DEAD TREE CLASSIFIER           #
################################################
# Import actual tree data from Sappi
tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(4)
# Clipped CHM path
chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_2_clipped.tif")

# Clipped ortho path
ortho_clipped_path = "ortho & pointcloud gen/outputs/GT/ortho_corrected_no_compression_clipped.tif"
ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
# %%
# Get tree positions from DeepForest
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'

# Load DF model
model = main.deepforest()
model.use_release()
print("Current device is {}".format(model.device))
model.to("cuda")
print("Current device is {}".format(model.device))
model.config["gpus"] = 1

model_path = 'df_models/final_model.pt'
model.model.load_state_dict(torch.load(model_path))

patch_size = 950
patch_overlap = 0.5
thresh = 0.9
iou_threshold = 0.05

# Get tree positions from DeepForest
predictions_df, predictions_df_transform, results_df_df, predicted_raster_image = deep_forest_pred(ortho_clipped_path, ortho_clipped_root, tree_point_calc_csv_path, tree_point_calc_shifted_csv_path, tree_actual_df, tree_actual_df_no_dead = tree_actual_df_no_dead, patch_size, patch_overlap, thresh, iou_threshold, save_fig = False, save_shape = False)

# Get average width, height and side of bounding box
average_width, average_height, average_side = get_average_box_size(predictions_df_transform)

expansion_factor = 0.5
expansion_size = average_side * expansion_factor
# %%
# Get tree positions from LocalMaxima
df_global_gp, tree_positions_from_lm, results_df_lm = local_maxima_func(chm_clipped_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df,  window_size=29, min_height=min_height, save_shape_file=True)
# %%
total_trees = results_df_df.iloc[0]['number_trees_pred'] + results_df_lm.iloc[0]['number_trees_pred']
total_dead_trees = results_df_df.iloc[0]['number_dead_pred'] + results_df_lm.iloc[0]['number_dead_pred']
total_unallocated = results_df_df.iloc[0]['number_unallocated'] + results_df_lm.iloc[0]['number_unallocated']
print('total trees: ',total_trees, '\ntotal dead trees: ', total_dead_trees, '\ntotal_unallocated: ', total_unallocated)
# %%
df_global_gp_red = df_global_gp[['X', 'Y', 'tree_id_pred']]
df_global_gp_red['model'] = 'LM'
predictions_df_transform_red = predictions_df_transform[['X', 'Y', 'tree_id_pred']]
predictions_df_transform_red.loc[:,'model'] = 'DF'
all_trees_pred = pd.concat([df_global_gp_red, predictions_df_transform_red]).reset_index(drop=True)
# %%
tree_positions_expanded_for_classification_all = boxes_from_points(all_trees_pred, expansion_size)
# %%
# Load ortho
ortho_cropped_for_cropping = rxr.open_rasterio(ortho_clipped_path, masked=True).squeeze()
# Crop all trees and obtain average R, G and B values for each crop
crop_df = crop_array(ortho_cropped_for_cropping, tree_positions_expanded_for_classification_all, save_crops = False)
crop_df.to_csv('tree_classifier_crops/crops_file_from_all_trees/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") +'_tree_crops_avg_rgb.csv',index=False)
# %%
crop_df = pd.read_csv('tree_classifier_crops/crops_file_from_all_trees/20210927_2243_tree_crops_avg_rgb.csv')
crop_df = crop_df.drop(columns=['Unnamed: 0'],axis=1)
# %%
# Scale features to match model input
scaler = MinMaxScaler()
# scaler.fit(X)
X_scaled_all_trees = pd.DataFrame()
X_scaled_all_trees[features_names] = scaler.fit_transform(crop_df[features_names])
X_scaled_all_trees.shape
# %%
# Load tree classifier model
svm_model_filename = 'tree_classifier_crops/saved_models/svm_v2.sav'
tree_classifier = pickle.load(open(svm_model_filename,"rb"))
# %%
# Make tree classification predictions
tree_classifications = tree_classifier.predict(X_scaled_all_trees)
tree_classifications_prob_dead = tree_classifier.predict_proba(X_scaled_all_trees)
tree_classifications_dec_func = tree_classifier.decision_function(X_scaled_all_trees)

all_trees_pred_incl_rgb = pd.concat([all_trees_pred,crop_df],axis=1)
all_trees_pred_incl_rgb['class'] = tree_classifications
all_trees_pred_incl_rgb['class_prob_dead'] = tree_classifications_prob_dead[:, 0]
all_trees_pred_incl_rgb['dec_func'] = tree_classifications_dec_func
# all_trees_pred_incl_rgb.to_csv('tree_classifier_crops/crops_file_from_all_trees/' + datetime.datetime.now().strftime("%Y%m%d_%H%M") +'_tree_crops_avg_rgb.csv')
# %%
# Read this: https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
# %%
# Remove all trees with a decision function output of less than -6.5
all_trees_pred_after_removal_LM = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] > -6) & (all_trees_pred_incl_rgb['model'] == 'LM')]
all_trees_pred_after_removal_DF = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] > -6) & (all_trees_pred_incl_rgb['model'] == 'DF')]
removed_LM = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] <= -6) & (all_trees_pred_incl_rgb['model'] == 'LM')]
removed_DF = all_trees_pred_incl_rgb[(all_trees_pred_incl_rgb['dec_func'] <= -6) & (all_trees_pred_incl_rgb['model'] == 'DF')]
# %%
lm_nn_list = []
for idx in all_trees_pred_after_removal_LM.index:
    current_tree = (all_trees_pred_after_removal_LM.loc[idx, 'X'], all_trees_pred_after_removal_LM.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    lm_nn_list.append(distance_to_nn)

df_nn_list = []
for idx in all_trees_pred_after_removal_DF.index:
    current_tree = (all_trees_pred_after_removal_DF.loc[idx, 'X'], all_trees_pred_after_removal_DF.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    df_nn_list.append(distance_to_nn)

df_removed_nn_list = []
for idx in removed_DF.index:
    current_tree = (removed_DF.loc[idx, 'X'], removed_DF.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    df_removed_nn_list.append(distance_to_nn)

lm_removed_nn_list = []
for idx in removed_LM.index:
    current_tree = (removed_LM.loc[idx, 'X'], removed_LM.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    lm_removed_nn_list.append(distance_to_nn)
# %%
print('DF MAE in position before classifier: ', np.round(results_df_df.loc[0,'MAE_position'],4))
print('LM MAE in position before classifier: ', np.round(results_df_lm.loc[0,'MAE_position'],4))
print('DF MAE in position after classifier: ', np.round(sum(df_nn_list)/len(df_nn_list),4))
print('LM MAE in position after classifier: ', np.round(sum(lm_nn_list)/len(lm_nn_list),4))
print('DF MAE of removed trees: ', np.round(sum(df_removed_nn_list)/len(df_removed_nn_list),4))
print('LM MAE of removed trees: ', np.round(sum(lm_removed_nn_list)/len(lm_removed_nn_list),4))
print('Predictions lost from DF: ', (predictions_df_transform_red.shape[0] - all_trees_pred_after_removal_DF.shape[0]))
print('Predictions lost from LM: ', (df_global_gp_red.shape[0] - all_trees_pred_after_removal_LM.shape[0]))
# %%
all_trees_pred_after_removal_LM_incl_act = all_trees_pred_after_removal_LM.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
all_trees_pred_after_removal_DF_incl_act = all_trees_pred_after_removal_DF.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')
all_trees_after_removal = pd.concat([all_trees_pred_after_removal_DF_incl_act[['tree_id_pred','model', '24_Def']], all_trees_pred_after_removal_LM_incl_act[['tree_id_pred','model', '24_Def']]])
# %%
total_trees = results_df_df.iloc[0]['number_trees_pred'] + results_df_lm.iloc[0]['number_trees_pred']
total_dead_trees = results_df_df.iloc[0]['number_dead_pred'] + results_df_lm.iloc[0]['number_dead_pred']
total_unallocated = results_df_df.iloc[0]['number_unallocated'] + results_df_lm.iloc[0]['number_unallocated']

total_trees_after_removal = all_trees_after_removal.shape[0]
total_dead_trees_after_removal = all_trees_after_removal[all_trees_after_removal['24_Def'].isin(['D', 'DT'])].shape[0]
total_unallocated_after_removal = all_trees_after_removal[all_trees_after_removal['tree_id_pred'].isna()].shape[0]

print('Before removal:\ntotal trees: ',total_trees, '\ntotal dead trees: ', total_dead_trees, '\ntotal_unallocated: ', total_unallocated)
print('\nAfter removal:\ntotal trees: ',total_trees_after_removal, '\ntotal dead trees: ', total_dead_trees_after_removal, '\ntotal_unallocated: ', total_unallocated_after_removal)
# %%
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# Read in dataframe of all trees
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

# Read in dataframe of all trees (shifted)
tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
tree_point_calc_shifted_df['tree_easting'] = tree_point_calc_shifted_df['geometry'].x
tree_point_calc_shifted_df['tree_northing'] = tree_point_calc_shifted_df['geometry'].y
tree_point_calc_shifted_df = tree_point_calc_shifted_df.merge(tree_pos_calc_df[['tree_id', 'Row', 'Plot', 'Tree no']], on='tree_id', how='left')
# outerpoints_gp, outerpoints_df = outer_trees(tree_point_calc_shifted_df)
# outer_trees_poly_df, outer_trees_buffered_poly_df = tree_buffer_area(outerpoints_gp, 0.8)
# outer_trees_buffered_poly_df.to_file('ortho & pointcloud gen/outputs/GT/shape_files/boundary.shp', driver='ESRI Shapefile')
# %%
# Import actual tree data from Sappi
# tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(4)
tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
tree_actual_df_val, tree_actual_df_full_val, tree_actual_df_no_dead_val, min_height_val = actual_tree_data(tree_actual_df_path, 'TRAIN')
# %%
results_df = pd.DataFrame()
results_idx = 0
for idx in range(len(predictions_df_transform)):

    current_tree = (predictions_df_transform.loc[idx, 'X'], predictions_df_transform.loc[idx, 'Y'])
    distance_to_nn, distance_to_nn_squared, tree_id = nearest_neighbor(current_tree, tree_point_calc_shifted_df)
    predictions_df_transform.loc[idx, 'tree_id_pred'] = tree_id
    predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist'] = distance_to_nn
    predictions_df_transform.loc[idx, 'tree_id_pred_nn_dist_squared'] = distance_to_nn_squared

# Allocate predictions to actual trees
ids_to_remove_pred_id, tree_locations_pred_df = find_duplicates(predictions_df_transform)

# Merge with actual data to determine number of dead trees predicted
tree_locations_pred_df = tree_locations_pred_df.merge(tree_actual_df[['tree_id', 'Hgt22Rod', '24_Def']], left_on='tree_id_pred', right_on='tree_id', how = 'left')

# tree_locations_pred_df_cleaned = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()==False]
# results_df.loc[results_idx, 'window_size'] = size
results_df.loc[results_idx, 'number_trees_pred'] = tree_locations_pred_df.shape[0]
results_df.loc[results_idx, 'number_unallocated'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna()].shape[0]
results_df.loc[results_idx, 'number_dead_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() == False) & (tree_locations_pred_df['24_Def'] == 'D')].shape[0]
results_df.loc[results_idx, 'perc_trees_pred'] = tree_locations_pred_df[(tree_locations_pred_df['tree_id_pred'].isna() != True) & (tree_locations_pred_df['24_Def'] != 'D')].shape[0] / tree_actual_df_no_dead.shape[0]
results_df.loc[results_idx, 'MAE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist'].mean()
results_df.loc[results_idx, 'MSE_position'] = tree_locations_pred_df['tree_id_pred_nn_dist_squared'].mean()
results_df.loc[results_idx, 'max_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].max()
results_df.loc[results_idx, 'min_dist'] = tree_locations_pred_df[tree_locations_pred_df['tree_id_pred'].isna() != True]['tree_id_pred_nn_dist'].min()
# %%
results_df
# %%

# %%

# %%

# %%
predictions_df_transform
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
for patch_size in [850,900,950,1000]:

    predicted_raster = model.predict_tile(image=img_bgr, return_plot = True, patch_size=patch_size,patch_overlap=0.75)
    df_image_save_path = 'deepforest_predictions/rasters_with_boxes/' + str(patch_size) + '_from_bgr_75_overlap.png'
    plt.imsave(df_image_save_path,arr=predicted_raster)
    print(patch_size)
# %%

# %%
img = plt.figure(figsize = (20,20))
# plt.imshow(predicted_raster[:,:,::-1])
plt.imshow(img_bgr)
# plt.imshow(boxes[:,:,::-1])
# plt.show()
# %%
# Split raster into crops for training

# %%


# %%
model.config
# %%
os.path.dirname(annotations_csv_filepath)
# %%
_ROOT
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
plt.imsave("predicted_raster_overlap_0.1.png",arr=predicted_raster[:,:,::-1])
# predicted_raster[0][0]
# %%
annotations_file = get_data("annotations.csv")
# %%
model.config["epochs"] = 1
model.config["save-snapshot"] = False
model.config["train"]["csv_file"] = annotations_file
model.config["train"]["root_dir"] = os.path.dirname(annotations_file)
# model.config["train"]["root_dir"] = ""

model.create_trainer()

# %%
model.config["train"]["fast_dev_run"] = True
# %%
model.trainer.fit(model)
# %%
print(annotations_file)
# %%
# %%
################################################
# #         BUILD DEAD TREE CLASSIFIER           #
# ################################################
# # %%
# # Get tree positions from DeepForest
# tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/tputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
# shape_files/tree_points_calculated.csv'
# tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/ou
# # Clipped CHM path
# # chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
# chm_train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
# chm_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_val_clip.tif")

# # Clipped ortho path
# ortho_name = 'ortho_train_clip.tif'
# ortho_clipped_root = "ortho & pointcloud gen/outputs/GT"
# ortho_clipped_path = ortho_clipped_root + '/' + ortho_name
# # %%
# # Import actual tree data from Sappi
# tree_actual_df_path = 'data/EG0181T Riverdale A9b Train Test Validation.xlsx'
# tree_actual_df_train, tree_actual_df_full_train, tree_actual_df_no_dead_train, min_height_train = actual_tree_data(tree_actual_df_path, 'TRAIN')
# # %%
# # Get tree positions from DeepForest
# tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
# tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)
# tree_point_calc_shifted_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_manual_shift_v2.shp'
# tree_point_calc_shifted_df = gpd.read_file(tree_point_calc_shifted_csv_path)
# tree_point_calc_shifted_df['X'] = tree_point_calc_shifted_df['geometry'].x
# tree_point_calc_shifted_df['Y'] = tree_point_calc_shifted_df['geometry'].y

# patch_size = 850
# patch_overlap = 0.5
# thresh = 0.4
# iou_threshold = 0.05

# # Load model
# model = main.deepforest()
# model.use_release()
# print("Current device is {}".format(model.device))
# model.to("cuda")
# print("Current device is {}".format(model.device))
# model.config["gpus"] = 1

# model_path = 'df_models/final_model.pt'
# model.model.load_state_dict(torch.load(model_path))

# # Get tree positions from DeepForest
# predictions_df, predictions_df_transform, results_df_df, predicted_raster_image = deep_forest_pred(ortho_name, ortho_clipped_path, ortho_clipped_root, tree_point_calc_csv_path=tree_point_calc_csv_path, tree_point_calc_shifted_csv_path=tree_point_calc_shifted_csv_path, tree_actual_df=tree_actual_df_train, tree_actual_df_no_dead = tree_actual_df_no_dead_train, patch_size=patch_size, patch_overlap=patch_overlap, thresh=thresh, iou_threshold=iou_threshold, save_fig = True, save_shape = True)
# # %%
# # Get tree positions from LocalMaxima
# # Clipped CHM path
# chm_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_2_clipped.tif")
# df_global_gp, tree_positions_from_lm, results_df, results_df_2, tree_locations_pred_df = local_maxima_func(chm_clipped_path, tree_point_calc_shifted_csv_path = tree_point_calc_csv_path, tree_actual_df = tree_actual_df_train, tree_pos_calc_df = tree_pos_calc_df, window_size=26, min_height=0.4, save_shape_file=True, grid_search=False)
# # chm_clipped_path, tree_point_calc_shifted_csv_path, tree_pos_calc_df, tree_actual_df, window_size, min_height, save_shape_file=False, grid_search=False)
# # %%
# # Get average width, height and side of bounding box
# average_width, average_height, average_side = get_average_box_size(predictions_df_transform)

# expansion_factor = 0.5
# expansion_size = average_side * expansion_factor
# # %%


# # dead_tree_for_classification, crop_df = dead_tree_classifier_dataset(predictions_df_transform, tree_actual_df, tree_positions_from_lm, chm_clipped_path, window_size=29, save_crops=False)
# dead_trees_classified, non_dead_tree_classified  = dead_tree_dataset(tree_point_calc_shifted_df, tree_actual_df_train, save_crops=False)
# classification_all = pd.concat([dead_trees_classified[['X', 'Y', 'class']], non_dead_tree_classified[['X', 'Y', 'class']]], ignore_index=True)
# classification_all = boxes_from_points(classification_all, expansion_size)
# # dead_tree_classification_df = boxes_from_points(dead_tree_classification_df, expansion_size)
# # %%
# # classification_df_1 = pd.read_csv('tree_classifier_crops/tree_crops_rgb_classified.csv')
# classification_df_2 = pd.read_csv('tree_classifier_crops/crops_file_from_all_trees/20210926_2312_tree_crops_avg_rgb_classified.csv')
# classification_df_2 = classification_df_2[classification_df_2['class'].isna()==False]
# classification_df_2['class'] = classification_df_2['class'].astype(int)
# # %%
# # classification_all = pd.concat([dead_tree_classification_df[['X', 'Y', 'class']], classification_df_2[['X', 'Y', 'class']]], ignore_index=True)
# # classification_all = boxes_from_points(classification_all, expansion_size)
# # %%
# classification_all
# # %%
# ortho_cropped_for_cropping = rxr.open_rasterio(ortho_clipped_path, masked=True).squeeze()
# # Crop all trees and obtain mean and SD R, G and B values for each crop
# crop_df = crop_array(ortho_cropped_for_cropping, classification_all, save_crops = False)
# # %%
# crop_df['class'] = classification_all['class']
# # %%
# feature_lists_df = pd.DataFrame(columns=['features'])
# features_names = ['r_avg', 'g_avg', 'b_avg', 'r_sd', 'g_sd', 'b_sd']
# from itertools import combinations
# df_iter = 0
# for r in range(1,len(features_names)-1):

#     feature_comb = list(combinations(features_names, r))
#     for i in range(len(feature_comb)):
#         feature_lists_df.loc[df_iter,'features'] =  list(feature_comb[i])

#         df_iter +=1

# feature_lists_df.loc[feature_lists_df.shape[0],'features'] = features_names
#  # %%
# X = crop_df[features_names]
# Y = crop_df['class']
# # %%
# # Minmax scaler
# scaler = MinMaxScaler()
# # scaler.fit(X)
# X_scaled = pd.DataFrame()
# X_scaled[features_names] = scaler.fit_transform(X[features_names])
# # %%
# scores_df_all_tests = pd.DataFrame(columns=['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'model'])
# test_counter = 0
# # for idx in range(len(feature_lists_df)):
# #     features_test = feature_lists_df.loc[idx,'features']
# #     X_features = X_scaled[features_test]
# #     print(features_test)

# for i in range(20):
#     # Split dataset into test and train
#     x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.25, stratify=Y)
#     # Instantiate model and fit
    

#     rf = RandomForestClassifier(n_estimators = 100, max_depth=None).fit(x_train, np.ravel(y_train))
#     ab = AdaBoostClassifier(n_estimators = 50, learning_rate = 1).fit(x_train, np.ravel(y_train))
#     svm_model = svm.SVC(probability=True).fit(x_train, np.ravel(y_train))
#     xgb_model = xgb.XGBClassifier(n_estimators = 100, objective="reg:squarederror", random_state=42, use_label_encoder=False).fit(x_train, np.ravel(y_train))

#     # Make predictions
#     predictions_rf = rf.predict(x_test)
#     predictions_ab = ab.predict(x_test)
#     predictions_xgb = xgb_model.predict(x_test)
#     predictions_svm = svm_model.predict(x_test)
#     predictions_rf_prob = rf.predict_proba(x_test)
#     predictions_ab_prob = ab.predict_proba(x_test)
#     predictions_svm_prob = svm_model.predict_proba(x_test)

#     score_df_rf = classification_scores(y_true=y_test, y_pred=predictions_rf, y_pred_prob=predictions_rf_prob[:, 1], model='random forest')
#     score_df_ab = classification_scores(y_true=y_test, y_pred=predictions_ab, y_pred_prob=predictions_ab_prob[:, 1], model='adaboost')
#     score_df_xgb = classification_scores(y_true=y_test, y_pred=np.round(predictions_xgb), y_pred_prob=predictions_xgb, model='xgboost')
#     score_df_svm = classification_scores(y_true=y_test, y_pred=predictions_svm, y_pred_prob=predictions_svm_prob[:, 1], model='svm')

#     scores_df_all_models = pd.concat([score_df_rf,score_df_ab,score_df_xgb,score_df_svm]).reset_index(drop=True)
#     # scores_df_all_models.loc[df_idx,'feature_list'] = features_test

#     scores_df_all_tests = pd.concat([scores_df_all_tests, scores_df_all_models]).reset_index(drop=True)

#     test_counter += 1

#     if test_counter % 200 == 0:

#         print(test_counter, ' tests of ', feature_lists_df.shape[0]*10*4, 'completed') 

# scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc']] = scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc']].astype(float)
# scores_df_all_tests_red = scores_df_all_tests[['accuracy', 'f1', 'precision', 'recall', 'auc', 'model']]
# scores_df_all_tests_red = scores_df_all_tests_red.reset_index(drop=True)
# scores_df_all_tests_avg = scores_df_all_tests_red.groupby(['model'], as_index=False).mean()
# # %%
# predictions_rf_prob

# # %%
# # %%
# # scores_df_all_tests.to_csv('tree_classifier_crops/classifier_scores/model_tests_all_scaled_v3_with_feature_combinations.csv')
# scores_df_all_tests.to_csv('tree_classifier_crops/classifier_scores/model_tests_all_scaled_v4.csv')
# scores_df_all_tests_avg.to_csv('tree_classifier_crops/classifier_scores/model_tests_average_scores_scaled_v4.csv')
# # %%
# feature_lists_df.loc[56,'features']
# # %%
# feature_lists_df.loc[42,'features']
# # %%
# # svm grid search
# feature_list_idxs = [42, 44, 45, 56]
# cs = [0.1, 0.5, 0.8, 1, 1.5]
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# degrees = [1,3,5]

# grid_search_idx = 0
# svm_parameter_search = pd.DataFrame(columns=['feature_idx','C', 'kernel', 'degree', 'accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'model'])
# for feature_list_idx in feature_list_idxs:

#     feature_list_for_test = feature_lists_df.loc[feature_list_idx,'features']
#     x_train, x_test, y_train, y_test = train_test_split(X_scaled[feature_list_for_test], Y, test_size = 0.25, stratify=Y, random_state=42)
    
#     for c in cs:
#         for kernel in kernels:
#             for degree in degrees:

#                 svm_model = svm.SVC(C=c, kernel=kernel, degree=degree, probability=True).fit(x_train, np.ravel(y_train))
#                 predictions_svm = svm_model.predict(x_test)
#                 predictions_svm_prob = svm_model.predict_proba(x_test)
#                 score_df_svm = classification_scores(y_true=y_test, y_pred=predictions_svm, y_pred_prob=predictions_svm_prob[:, 1], model='svm',features=None, feature_list_id=feature_list_idx)
#                 svm_parameter_search.loc[grid_search_idx,'C'] = c
#                 svm_parameter_search.loc[grid_search_idx,'kernel'] = kernel
#                 svm_parameter_search.loc[grid_search_idx,'degree'] = degree
#                 svm_parameter_search.loc[grid_search_idx,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_svm.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]

#                 grid_search_idx += 1 
#     print('feature list ', feature_list_idx, ' done')

# svm_parameter_search.to_csv('tree_classifier_crops/classifier_scores/svm_scores_v5.csv')
# # %%
# # Random Forest grid search
# feature_list_idxs = [42, 41, 43, 56]
# n_estimators = [10, 50, 100, 150, 200, 250, 300, 350]
# criterions = ['gini', 'entropy']
# max_depths = [None, 10, 20, 50, 100, 150, 200, 250]

# grid_search_idx = 0
# rf_parameter_search = pd.DataFrame(columns=['n_estimators', 'criteria', 'max_depth', 'accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'model'])
# for feature_list_idx in feature_list_idxs:

#     feature_list_for_test = feature_lists_df.loc[feature_list_idx,'features']
#     x_train, x_test, y_train, y_test = train_test_split(X_scaled[feature_list_for_test], Y, test_size = 0.25, stratify=Y, random_state=21)

#     for n_estimator in n_estimators:
#         for criterion in criterions:
#             for max_depth in max_depths:

#                 rf = RandomForestClassifier(n_estimators = n_estimator, max_depth=max_depth, criterion = criterion, random_state=42).fit(x_train, np.ravel(y_train))
#                 predictions_rf = rf.predict(x_test)
#                 predictions_rf_prob = rf.predict_proba(x_test)
#                 score_df_rf = classification_scores(y_true=y_test, y_pred=predictions_rf, y_pred_prob=predictions_rf_prob[:, 1], model='random forest',features=None, feature_list_id=feature_list_idx)

#                 rf_parameter_search.loc[grid_search_idx,'n_estimators'] = n_estimator
#                 rf_parameter_search.loc[grid_search_idx,'criteria'] = criterion
#                 rf_parameter_search.loc[grid_search_idx,'max_depth'] = max_depth
#                 rf_parameter_search.loc[grid_search_idx,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']] = score_df_rf.loc[0,['accuracy', 'f1', 'precision', 'recall', 'confusion', 'auc', 'feature_list_idx']]

#                 grid_search_idx += 1

#     print('feature list ', feature_list_idx, ' done') 
# rf_parameter_search.to_csv('tree_classifier_crops/classifier_scores/random_forest_scores_v5.csv')
# # %%
# # Build and save model (n_estimators = 50, criterion = 'gini, max_depth=None)
# rf = RandomForestClassifier(n_estimators = 150, max_depth=10, criterion = 'entropy', random_state=42).fit(x_train, np.ravel(y_train))
# # rf_model_filename = 'tree_classifier_crops/saved_models/random_forest_v2.sav'
# # pickle.dump(rf, open(rf_model_filename, 'wb'))
# features_names = X_scaled.columns

# importances = rf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
# forest_importances = pd.Series(importances, index=features_names)

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# # %%
# # %%
# for feature_list_idx in feature_list_idxs:

#     feature_list_for_test = feature_lists_df.loc[56,'features']
#     x_train, x_test, y_train, y_test = train_test_split(X_scaled[feature_list_for_test], Y, test_size = 0.25, stratify=Y, random_state=42)
# svm_model = svm.SVC(C=0.5, kernel='poly', degree=5, probability=True).fit(x_train, np.ravel(y_train))
# svm_model_filename = 'tree_classifier_crops/saved_models/svm.sav'
# pickle.dump(svm_model, open(svm_model_filename, 'wb'))