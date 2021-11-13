import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import distance

# Import tree row boundary tree positions
tree_rows_path = os.path.join("ortho & pointcloud gen", "outputs", 
                                  "GT", "shape_files", 
                                  "tree_rows_2.shp")
tree_rows = gpd.read_file(tree_rows_path)
tree_rows = tree_rows[tree_rows['geometry'].isna() == False].reset_index(drop=True)

# Set tree position orientations
row_list = []
for idx in tree_rows.index:
    tree_id = tree_rows.loc[idx,'id']

    if tree_id in row_list: tree_rows.loc[idx,'orientation'] = 'E'

    else: tree_rows.loc[idx,'orientation'] = 'W'

    row_list.append(tree_id)

# Calculate row distances
trees_per_row = 72
for ids in tree_rows['id'].unique():

    west_idx = tree_rows[(tree_rows['id'] == ids) & (tree_rows['orientation'] == 'W')].index[0]
    east_idx = tree_rows[(tree_rows['id'] == ids) & (tree_rows['orientation'] == 'E')].index[0]
    pnt_e = tree_rows.loc[west_idx].geometry
    pnt_w = tree_rows.loc[east_idx].geometry

    distance = pnt_e.distance(pnt_w)

    tree_rows.loc[west_idx,'row_length'] = distance
    tree_rows.loc[east_idx,'row_length'] = distance
    tree_rows.loc[west_idx,'tree_spacing'] = distance / (trees_per_row - 1)
    tree_rows.loc[east_idx,'tree_spacing'] = distance / (trees_per_row - 1)

# Calculate tree positions
df_idx = 0
tree_locations_df = pd.DataFrame(columns = ['Row','Plot', 'Tree no', 'tree_easting', 'tree_northing'])
for row in tree_rows['id'].unique():
    
    west_idx = tree_rows[(tree_rows['id'] == row) & (tree_rows['orientation'] == 'W')].index[0]
    east_idx = tree_rows[(tree_rows['id'] == row) & (tree_rows['orientation'] == 'E')].index[0]
    pnt_e = tree_rows.loc[west_idx].geometry
    pnt_w = tree_rows.loc[east_idx].geometry  

    pnt_e_easting = pnt_e.coords[0][0]
    pnt_e_northing = pnt_e.coords[0][1]

    pnt_w_easting = pnt_w.coords[0][0]
    pnt_w_northing = pnt_w.coords[0][1]

    delta_easting = (pnt_w_easting - pnt_e_easting) / (trees_per_row - 1)
    delta_northing = (pnt_w_northing - pnt_e_northing) / (trees_per_row - 1)

    plot = row
    tree_in_plot = 1
    current_easting = pnt_e_easting
    current_northing = pnt_e_northing
    for tree_in_row in range(1,trees_per_row+1):

        if tree_in_row == 1: 
            tree_locations_df.loc[df_idx,'Row'] = row
            tree_locations_df.loc[df_idx,'Plot'] = row
            tree_locations_df.loc[df_idx,'Tree no'] = 1
            tree_locations_df.loc[df_idx,'tree_easting'] = pnt_e_easting
            tree_locations_df.loc[df_idx,'tree_northing'] = pnt_e_northing

        else: 
            current_easting = current_easting + delta_easting
            current_northing = current_northing + delta_northing
            tree_locations_df.loc[df_idx,'Row'] = row
            tree_locations_df.loc[df_idx,'Plot'] = plot
            tree_locations_df.loc[df_idx,'Tree no'] = tree_in_plot
            tree_locations_df.loc[df_idx,'tree_easting'] = current_easting
            tree_locations_df.loc[df_idx,'tree_northing'] = current_northing

        tree_in_plot += 1  
        df_idx += 1

        if tree_in_plot % 7 == 0: 
            plot = plot + 69
            tree_in_plot = 1

# Write tree positions to csv and shapefile
tree_locations_df['geometry'] = tree_locations_df.apply(lambda x: Point((float(x.tree_easting), float(x.tree_northing))), axis=1)
tree_locations_gp = gpd.GeoDataFrame(tree_locations_df, geometry='geometry', crs="EPSG:32735")
tree_locations_gp['tree_id'] = tree_locations_gp['Plot'].astype(str) + '_' + tree_locations_gp['Tree no'].astype(str)
shape_file_name = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_v2.shp'
csv_file_name = 'ortho & pointcloud gen/outputs/GT/shape_files/tree_points_calculated_v2.csv'

tree_locations_gp[['tree_id', 'geometry']].to_file(shape_file_name, driver='ESRI Shapefile')
tree_locations_gp.to_csv(csv_file_name, index=False)