import os
from re import I
import pandas as pd
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import mapping
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from os import listdir
from os.path import isfile, join
import sys

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

def train_test_plots(tree_actual_df, start_plot, num_rows):
    """
	Function to split ground truth data according to given plot and number of rows
	
	Parameters:
	tree_actual_df (dataframe): Dataframe containing all ground truth data (tree heights, row #, plot #, etc.)
	start_plot (int): Plot number at which to start split
    num_rows (int): Number of rows included in subset
	
	Returns:
	tree_actual_df_red (dataframe): Reduced dataframe consisting only of ground truth data according to start_plot and num_rows
	"""

    plot_list = []
    for i in range(12): 

        end_plot = start_plot + num_rows

        plot_list_int = list(range(start_plot,end_plot))
        for plot in plot_list_int:
            plot_list.append(plot)

        start_plot = start_plot + 69

    tree_actual_df_red = tree_actual_df[tree_actual_df['Plot'].isin(plot_list)]

    return tree_actual_df_red

# Create train, test, and validation sets
tree_actual_df, tree_actual_df_full, tree_actual_df_no_dead, min_height = actual_tree_data(4)
tree_actual_df_train = train_test_plots(tree_actual_df_full, start_plot=37, num_rows=33)
tree_actual_df_val = train_test_plots(tree_actual_df_full, start_plot=1, num_rows=18)
tree_actual_df_test = train_test_plots(tree_actual_df_full, start_plot=19, num_rows=18)

with pd.ExcelWriter('data/EG0181T Riverdale A9b Train Test Validation.xlsx') as writer:  
    tree_actual_df_train.to_excel(writer, sheet_name='TRAIN')
    tree_actual_df_val.to_excel(writer, sheet_name='VAL')
    tree_actual_df_test.to_excel(writer, sheet_name='TEST')

################################################
#        CLIP MAIN ORTHOMOSAIC BY AOI          #
################################################

# Read in dataframe of all trees
tree_point_calc_csv_path = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated.csv'
tree_pos_calc_df = pd.read_csv(tree_point_calc_csv_path)

# Read in dataframe of all trees (shifted) and assigned points to geometries
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

ortho = rxr.open_rasterio(ortho_raster_path, masked=True)
ortho_clipped = ortho.rio.clip(outer_trees_buffered_poly_df.geometry)
ortho_clipped_path = os.path.join("ortho & pointcloud gen","outputs","GT",
                            "ortho_corrected_no_compression_clipped.tif")
ortho_clipped.rio.to_raster(ortho_clipped_path)

################################################
#             OPEN ORTHOMOSAIC                 #
################################################

# Open orthomosaic as Numpy array (rgb & bgr)
with rasterio.open(ortho_clipped_path) as source:
    img = source.read() # Read raster bands as a numpy array
    transform_crs = source.transform
    crs = source.crs

img = img.astype(np.uint8)
img_rgb = np.moveaxis(img, 0, 2).copy()
img_bgr = img_rgb[...,::-1].copy()
plt.imshow(img_rgb)

################################################
# CLIP ORTHOMOSAIC INTO TRAIN, VAL & TEST SETS #
################################################

# Set tree points of boundary trees in train, validation and test sets
train_point_X_list = [802900.405, 802873.273, 802788.908, 802726.607, 802750.400]
train_point_Y_list = [6695320.204, 6695217.113, 6695239.194, 6695255.324, 6695359.291]

test_point_X_list = [802873.273, 802858.272, 802797.917, 802752.715, 802711.369, 802726.607, 802788.908]
test_point_Y_list = [6695217.113, 6695165.328, 6695181.150, 6695191.818, 6695203.817, 6695255.324, 6695239.194]

val_point_X_list = [802858.272, 802844.700, 802690.985, 802711.369, 802752.715, 802797.917]
val_point_Y_list = [6695165.328, 6695107.615, 6695143.427, 6695203.817, 6695191.818, 6695181.150]

# Create a buffer around each section (set) polygons
polygon_train_geom = Polygon(zip(train_point_X_list, train_point_Y_list)).buffer(0.3) 
polygon_test_geom = Polygon(zip(test_point_X_list, test_point_Y_list)).buffer(0.3) 
polygon_val_geom = Polygon(zip(val_point_X_list, val_point_Y_list)).buffer(0.3) 

# Set CRS and convert to GeoPandas DataFrame
crs = {'init': 'epsg:32735'}
polygon_train = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_train_geom])  
polygon_test = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_test_geom])
polygon_val = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_val_geom]) 

# Write polygons to shapefiles to visualise
polygon_train.to_file('data/QGIS and CC/train_poly.shp', driver='ESRI Shapefile')
polygon_test.to_file('data/QGIS and CC/test_poly.shp', driver='ESRI Shapefile')
polygon_val.to_file('data/QGIS and CC/val_poly.shp', driver='ESRI Shapefile')

# Clip orthomosaic using polygons
ortho_train_clip = ortho_clipped.rio.clip(polygon_train.geometry)
ortho_test_clip = ortho_clipped.rio.clip(polygon_test.geometry)
ortho_val_clip = ortho_clipped.rio.clip(polygon_val.geometry)

# Set filepaths for orthomosaic sections (sets)
train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "ortho_train_clip.tif")
test_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "ortho_test_clip.tif")
val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "ortho_val_clip.tif")

# Write clipped orthomosaics to GeoTiff files for later use
ortho_train_clip.rio.to_raster(train_clip_path)
ortho_test_clip.rio.to_raster(test_clip_path)
ortho_val_clip.rio.to_raster(val_clip_path)

################################################
#     CLIP CHM INTO TRAIN, VAL & TEST SETS     #
################################################

# Import CHM
chm_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_2_clipped.tif")
chm = rxr.open_rasterio(chm_path, masked=True).squeeze()

# Create polygons for CHM clipping WITHOUT buffering
polygon_train_geom_chm = Polygon(zip(train_point_X_list, train_point_Y_list))
polygon_test_geom_chm = Polygon(zip(test_point_X_list, test_point_Y_list))
polygon_val_geom_chm = Polygon(zip(val_point_X_list, val_point_Y_list))

# Set CRS and convert to GeoPandas DataFrame
crs = {'init': 'epsg:32735'}
polygon_train_chm = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_train_geom_chm])  
polygon_test_chm = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_test_geom_chm])
polygon_val_chm = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_val_geom_chm]) 

# Clip CHM using polygons
chm_train_clip = chm.rio.clip(polygon_train_chm.geometry)
chm_test_clip = chm.rio.clip(polygon_test_chm.geometry)
chm_val_clip = chm.rio.clip(polygon_val_chm.geometry)

# Set filepaths for CHM sections (sets)
chm_train_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_train_clip.tif")
chm_test_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_test_clip.tif")
chm_val_clip_path = os.path.join("ortho & pointcloud gen","outputs","GT", "chm_val_clip.tif")

# Write clipped CHMs to GeoTiff files for later use
chm_train_clip.rio.to_raster(chm_train_clip_path)
chm_test_clip.rio.to_raster(chm_test_clip_path)
chm_val_clip.rio.to_raster(chm_val_clip_path)

################################################
#   CREATE MERGE OF TRAIN & VAL CHMS FOR LM    #
################################################

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