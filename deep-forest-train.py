import os
from re import I
import pandas as pd
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
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

################################################
#      CREATE IMAGE CROPS FOR ANNOATIONS       #
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


################################################
#             TRAIN DEEP FOREST                #
################################################

# Convert CVAT annotations to dataframe as per DF requirements (train)
folder_path_train='df_crops_annotations/train/'
dataset_train='train'
annotations_df, annotations_csv_filename, annotations_csv_filepath = annotation_json_to_csv(folder_path_train, dataset_train)

# Convert CVAT annotations to dataframe as per DF requirements (val)
folder_path_val='df_crops_annotations/val/'
dataset_val='val'
image_path = folder_path_val  + 'ortho_cropped/' + 'ortho_val_clip.tif'
annotations_df_val, annotations_csv_filename_val, annotations_csv_filepath_val = annotation_json_to_csv(folder_path_val, dataset_val, image_path='ortho_val_clip.tif')
root_dir_val_ortho = 'df_crops_annotations/val/ortho_cropped'

# Set csv path and root dir to validation orthomosaic (required for config file)
annotations_cropped_csv_filepath_val = 'df_crops_annotations/val/ortho_cropped/split_annotations_val/ortho_val_clip.csv'
root_dir_val_crops = 'df_crops_annotations/val/ortho_cropped/split_annotations_val'

# Instantiate model
model = main.deepforest()
# Use latest release
model.use_release()

# Use GPU rather than CPU
model.to("cuda")
model.config["gpus"] = 1

# Set up model training
model.config["train"]["epochs"] = 5                 # Number of epochs
model.config["workers"] = 0                         # Number of CPU workers (0 because of GPU) 
model.config["batch_size"]= 7                       # Set batch size
model.config["score_thresh"] = 0.5                  # Set score threshold
model.config["train"]["optimiser"] = 'orig'         # Set optimiser (custom option)
model.config["train"]["lr_schedule"] = 'orig'       # Set learning rate schedule (custom option)
model.config["train"]["lr"] = 0.005                 # Set initial learning rate
model.config["train"]["patience"] = 5               # Set patience (custom option)
model.config["train"]["factor"] = 0.05              # Set factor (custom options)
        
# Set train and validation root directory (annotation csv) and fiilepath to annotation images
model.config["train"]["csv_file"] = annotations_csv_filepath
model.config["train"]["root_dir"] = 'df_crops_annotations/train/crops'
model.config["validation"]["csv_file"] = annotations_csv_filepath_val
model.config["validation"]["root_dir"] = root_dir_val_ortho

# Instantiate logger to track losses
name = 'val_final'
logger = CSVLogger("logs", name=name)

# Create trainer and train model
model.create_trainer(logger=logger)
model.trainer.fit(model)

# Save model
model_path = 'df_models/final_model.pt'
torch.save(model.model.state_dict(),model_path)

model_path = 'df_models/final_model.pt'
model.model.load_state_dict(torch.load(model_path))

########################################################
# DEEP FOREST PART GRID SEARCH (FOR INFORMATION ONLY!) #
########################################################

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
        results_df.to_csv('df_models/df_val_eval_scores.csv')
        results_idx += 1
        print('val done')

        # Evaluate model on train data
        results_df = pd.DataFrame(columns=['trees_matched','box_precision', 'box_recall', 'box_f1', 'miou'])
        model.config["score_thresh"] = 0.75
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
