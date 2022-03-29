# Test download and usage of osm.pbf files:
# https://pyrosm.readthedocs.io/en/latest/basics.html#Protobuf-file:-What-is-it-and-how-to-get-one?

import urllib.request 
from pyrosm import get_data
from pyrosm import OSM
from pyrosm.data import sources
import geopandas as gpd
from bs4 import BeautifulSoup
import requests
import re
import numpy as np
from urllib.parse import urljoin
import urllib
import os
from urllib.parse import urlparse
import time
import glob
import pandas as pd
from pathlib import Path
import pickle
import psutil
from zipfile import ZipFile
import gc
import matplotlib.pyplot as plt 
import rasterio
import rasterio.mask
from osgeo import gdal
from shutil import copyfile
import time
import config

#print("Sub-regions in Germany:", sources.subregions.germany.available)

def get_url_download_list(website, keyword="", extension=".zip"):
    '''
    Function scrapes download urls from a given website. 
    Inputs: 
            :params: website: url of the website from which download links should e scraped
                     extension: filtype of the download files (in case only certain files should be downloaded)
                     keyword: string that should be matched (can be used for url sub-selection). If empty: all urls with extension will be added
            :types: website, extension, keyword: string
    Output:
    	    :return: download_list: list of download urls

    '''
    response = requests.get(website)
    soup = BeautifulSoup(response.text, "lxml")
    #table = soup.find('table', {'id':'subregions'})
    download_list = [urljoin(website, a['href']) for a in soup.find_all("a", href=re.compile(fr".*{keyword}(.*){extension}$"))]

    return download_list


def download_files_from_list(urllist, directory):
    '''
    Function downloads all files from a url list to a specific directory
    Inputs:
            :params: urllist: list with urls
                     directory: path to store the files on your local machine
            :types: urllist: list
                     directory: string
    Output:  
            :return: None                 
    '''
    for url in urllist:
        time.sleep(5)
        filename = os.path.basename(urlparse(url).path)
        res = requests.get(url)

        with open(f"{directory}/{filename}", "wb") as file:
            file.write(res.content)

        #urllib.request.urlretrieve(url, f"{directory}/{filename}")
        #urllib.request.urlretrieve(url, f"{directory}")

    return None



def buffer_geoms(gdf_in, buffer, proj_epsg=25832):
    '''
    Function calculates a buffered GeoSeries from the geometry column of a geopandas DataFrame
    in a given meter based coordinate reference system (crs) and converts it back to the orginial crs
    '''
    init_crs = gdf_in.crs
    gdf_in = gdf_in.to_crs(proj_epsg)
    buffered = gpd.GeoDataFrame(geometry = gdf_in.geometry.buffer(buffer)).to_crs(init_crs)

    return buffered




def mask_raster(raster_path, shapes, output_path, filename, inversion=True, cropping=False, mask_by_intersection=False):
    '''
    '''
    with rasterio.open(raster_path) as src:
        print(src.meta)
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=cropping, invert=inversion, all_touched=mask_by_intersection)
        out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        out_image[out_image==0] = np.nan

        with rasterio.open(f'{output_path}/{filename}', "w", **out_meta) as dest:
            dest.write(out_image)

        return np.count_nonzero(~np.isnan(out_image[0]))


def create_nopa(param_dict):
    '''
    
    '''
    for key in param_dict["other_nopa"]:
        gdf = gpd.read_file(param_dict["other_nopa"][key])

        if isinstance(param_dict["nopa_distance"][key], str): # buffer shapes with distances from column in shapefile
            gdf_buff = gdf.copy().to_crs(param_dict["epsg_distance_calc"])
            gdf_buff['geometry'] = gdf_buff.apply(lambda x: x.geometry.buffer(int(x[param_dict["nopa_distance"][key]])), axis=1)
            gdf_buff = gdf_buff.to_crs(gdf.crs)

        elif param_dict["nopa_distance"][key] != 0:
            gdf_buff = buffer_geoms(gdf, param_dict["nopa_distance"][key], param_dict["epsg_distance_calc"])

        else: # dont buffer shapes
            gdf_buff = gdf


        nopa_shapes = gdf_buff.geometry


        if os.path.isfile(f'{param_dict["gwa_out_path"]}/gwa_masked_tmp.tif'):
            gwa_path = f'{param_dict["gwa_out_path"]}/gwa_masked_tmp.tif'
        else:
            gwa_path = f'{param_dict["gwa_in_file"]}.tif'

        if key == "region_border":
            cell_count = mask_raster(gwa_path, nopa_shapes, param_dict["gwa_out_path"], "gwa_masked_tmp.tif",
                                    inversion=param_dict["inversion"][key], cropping=param_dict["cropping"][key], mask_by_intersection=False)
        else:
            cell_count = mask_raster(gwa_path, nopa_shapes, param_dict["gwa_out_path"], "gwa_masked_tmp.tif",
                                    inversion=param_dict["inversion"][key], cropping=param_dict["cropping"][key], 
                                    mask_by_intersection=param_dict["intersection_mask"]) 


        print(f'Updated number of raster cells after dropping {key} data', cell_count)


    osmfiles = glob.glob(os.path.join(param_dict["osm_path"], '*.zip'))

    iteration = 0
    for key in param_dict["osm_keys"]:
        filename = param_dict["osm_file_names"][key]

        for z in osmfiles:
            print(f"Reading {key} data from {z}/{filename}")
            gdf = gpd.read_file(f"zip://{z}/{filename}").loc[:, ['fclass','geometry']]

            if key == "landuse":
                gdf_prot = gdf.loc[(gdf['fclass'] == 'nature_reserve') | (gdf['fclass'] == 'national_park')]
                gdf_res = gdf.loc[gdf['fclass'] == 'residential']

                buff_prot = buffer_geoms(gdf_prot, param_dict["nopa_distance"][key]["protected_areas"], param_dict["epsg_distance_calc"])
                buff_res = buffer_geoms(gdf_res, param_dict["nopa_distance"][key]["residential"], param_dict["epsg_distance_calc"])
                gdf_buff = gpd.GeoDataFrame(pd.concat([buff_prot, buff_res], ignore_index=True), crs=buff_prot.crs)

            elif key == "roads":
                gdf_major = gdf.loc[gdf['fclass'].isin(param_dict["osm_major_roads"])]
                gdf_minor = gdf.loc[gdf['fclass'].isin(param_dict["osm_minor_roads"])]
                gdf_small = gdf.loc[gdf['fclass'].isin(param_dict["osm_small_roads"])]

                buff_major = buffer_geoms(gdf_major, param_dict["nopa_distance"][key][0], param_dict["epsg_distance_calc"])
                buff_minor = buffer_geoms(gdf_minor, param_dict["nopa_distance"][key][1], param_dict["epsg_distance_calc"])
                buff_small = buffer_geoms(gdf_small, param_dict["nopa_distance"][key][2], param_dict["epsg_distance_calc"])
                gdf_buff = gpd.GeoDataFrame(pd.concat([buff_major, buff_minor, buff_small], ignore_index=True), crs=buff_major.crs)
                #gdf_buff = gpd.GeoDataFrame(pd.concat([buff_major, buff_minor], ignore_index=True), crs=buff_major.crs)

            else:
                gdf_buff = buffer_geoms(gdf, param_dict["nopa_distance"][key], param_dict["epsg_distance_calc"])


            nopa_shapes = gdf_buff.geometry


            if os.path.isfile(f'{param_dict["gwa_out_path"]}/gwa_masked_tmp.tif'):
                gwa_path = f'{param_dict["gwa_out_path"]}/gwa_masked_tmp.tif'
            else:
                gwa_path = f'{param_dict["gwa_in_file"]}.tif'

            cell_count = mask_raster(gwa_path, nopa_shapes, param_dict["gwa_out_path"], "gwa_masked_tmp.tif", 
                                    inversion=param_dict["inversion"][key], mask_by_intersection=param_dict["intersection_mask"])


        iteration += 1

        if param_dict["save_after"][key]:
            print(f"Saving file after masking {key} data (OSM masking step {iteration})...")
            copyfile(f'{param_dict["gwa_out_path"]}/gwa_masked_tmp.tif', f'{param_dict["gwa_out_path"]}/gwa_masked_{key}_{iteration}.tif')


        print(f'Updated number of raster cells after dropping {key} data', cell_count)


    # Drop tmp file and store masked_raster:
    os.replace(f'{param_dict["gwa_out_path"]}/gwa_masked_tmp.tif', f'{param_dict["gwa_out_path"]}/{param_dict["masked_file_name"]}.tif')

    return None


# param = config.general_settings()

# osmfiles = glob.glob(os.path.join(param["osm_path"], '*.zip'))

# for z in osmfiles:
#     print(z)
#     osm_gdf = gpd.read_file(f"zip://{z}/gis_osm_buildings_a_free_1.shp").loc[:, ['geometry']].to_crs(param["epsg_distance_calc"])
#     osm_gdf = osm_gdf.set_geometry(osm_gdf.centroid).to_crs(param["epsg_general"])
#     gdf = gpd.read_file(f"zip://{z}/gis_osm_landuse_a_free_1.shp").loc[:, ['fclass','geometry']]
#     gdf = gdf.loc[(gdf['fclass'] == 'residential')]
#     dfsjoin = gpd.clip(osm_gdf, gdf)
#     print(dfsjoin.head())
#     print(len(osm_gdf))
#     print(len(dfsjoin))
#     # fig, ax = plt.subplots(figsize=(15, 15))
#     # osm_gdf.plot(ax=ax, color='none', edgecolor='blue')
#     # dfsjoin.plot(ax=ax, color='none', edgecolor='red')
#     # gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=3)
#     # plt.show()
#     exit()
# exit()

def download_gwa(outpath, wind_layer, country='DEU', height=100):
    '''
    Download Data from the Global Wind Atlas Website 
    '''
    if 'IEC' in wind_layer:
        url = f'https://globalwindatlas.info/api/gis/country/{country}/{wind_layer}/'
        filename = f"{outpath}/{wind_layer.replace('-', '_')}_{country}_inp.tif"
            
        print(f"Downloading {wind_layer} layer ...")

    else: 
        url = f'https://globalwindatlas.info/api/gis/country/{country}/{wind_layer}/{height}'
        filename = f"{outpath}/{wind_layer.replace('-', '_')}_{country}_{height}_inp.tif"

        print(f"Downloading {wind_layer} layer at {height}m ...")

    res = requests.get(url)
    res.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(res.content)




def generate_gwa_input(param_dict, name_out, folder, iso3, layers, heights):
    """
    Function downloads files from the Global Wind Atlas web page and converts them into a multiband tif file
    Note: The band order corresponds to the alphabetical order!

    Parameters
    ----------

    name_out: str
        Name of the multiband raster that should be generated
    folder : str
        The directory where files shoukd be downloaded and where the Output file should be generated
    iso3 : str
        Country iso3 identification code
    layers : list
        List of layers for which data should be downloaded (eg.: ['air-density', 'wind-speed', 'power-density'])
    heights: list
        List of heights (in m) for which given layers should be downloaded  (eg: [100, 150])

    """

    # Download gwa layers:
    # for l in layers:
    #     if "IEC" in l:
    #         download_gwa(folder, l , country=iso3, height=h)
    #         time.sleep(10)

    #     else:
    #         for h in heights:
    #             download_gwa(folder, l , country=iso3, height=h)
    #             time.sleep(10)


    # stack gwa layers and create a multiband raster (important file list order influences band order!):
    gwa_list = glob.glob(f'{folder}/*inp.tif')

    # # Resample the Raster into 500x500m size
    # for file in gwa_list:
    #     filename = Path(file).stem
    #     gdal.Warp(f'{folder}/{filename}_proj.tif', file, dstSRS=f'EPSG:{param_dict["epsg_distance_calc"]}')
    #     gdal.Warp(f'{folder}/{filename}_res.tif', f'{folder}/{filename}_proj.tif', xRes=500, yRes=500, resampleAlg='bilinear')
    #     gdal.Warp(f'{folder}/{filename}_resampled.tif', f'{folder}/{filename}_res.tif', dstSRS=f'EPSG:{param_dict["epsg_general"]}')

    #     # Delete unnecessary files
    #     for temp in ['proj', 'res']:
    #         try:
    #             os.remove(f'{folder}/{filename}_{temp}.tif')
    #         except OSError:
    #             pass

    # # Convert into a multiband tiff file
    # gwa_list = glob.glob(f'{folder}/*resampled.tif')


    vrt = gdal.BuildVRT(f'{name_out}.vrt', gwa_list, separate=True)
    gdal.Translate(f'{name_out}.tif', vrt)





def raster2csv(raster_path, csv_path, col_names):
    """
    Function converts a GeoTiff raster file to a csv file with lon and lat columns representing the coordinates
    of the raster cell centre points. All raster bands will be added as individual columns of the DataFrame.
    
    Parameters:
        raster_path - Path to the raster file location
        csv_path    - Path to the Output location (with filename) of the csv file
        col_names   - list of column names attributed to the raster bands

    Returns:
        None

    # Helpful video for similar gdal application: https://www.youtube.com/watch?v=zLNLG0j13Cw
    """
    class Error(Exception):
        pass

    with rasterio.open(raster_path) as src:
        data = src.read(list(range(1, src.count + 1))) # read all bands of raster
        data = data.reshape(data.shape[0], -1)
        if len(col_names) != data.shape[0]: # Error check if number of bands and number of names match.
            raise Error("Number of column names does not match number of data bands in raster file.")
        data_dict = dict(zip(col_names, data)) # create dictionary that can be later used as input for pandas

        # Read in raster size x and y starting values and pixel width (res): 
        res = src.transform[0]
        xmin = src.transform[2]
        ymax = src.transform[5]
        xsize = src.width
        ysize = src.height

        # Get center of cell in upper left corner
        xstart = xmin + res/2
        ystart = ymax - res/2

        # Get centers of other cells
        x = xstart + np.arange(0, xsize) * res
        y = ystart - np.arange(0, ysize) * res

        x = np.tile(x, ysize)
        y = np.repeat(y, xsize)

        data_dict['lon'] = x
        data_dict['lat'] = y

        # Create and export DataFrame
        df = pd.DataFrame(data_dict)

        # Drop columns with no data values
        df.dropna(subset=list(set(df.columns).difference(["lon", "lat"])), inplace=True)

        df.to_csv(csv_path, index=False)

    return None


def get_gdf_raster_value(gdf, raster_file):
    '''
    Function obtains raster values from input raster_file for every location in a GeoDataFrame (gdf)

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Geodataframe containing the coordinates for which we want to check the raster value
    raster_file : str
        Path to the raster file of interest

    Returns
    -------
    values : list
        List of values for each location given in the GeoDataFrame

    '''
    gdf_in = gdf.copy()
    
    # Import raster:
    src  = rasterio.open(f"{raster_file}")
    gdf_in = gdf_in.to_crs(src.crs) # convert gdf to crs of the raster file
    
    # Creat list of coordinate tuples from gdf geometry:
    coords = [(x,y) for x, y in zip(gdf_in.geometry.x, gdf_in.geometry.y)]
    # Calculate altitude:
    values = [x[0] for x in src.sample(coords)]
    
    return values


def calc_house_impact_allocation(param_dict):
    '''
    '''
    osmfiles = glob.glob(os.path.join(param_dict["osm_path"], '*.zip'))
    filename = param_dict["osm_file_names"]["buildings"]
    #sub_files = np.array_split(osmfiles, 6)

    hp_gdf = gpd.read_file(f'{param_dict["house_prices"]}', crs=param_dict["epsg_general"])
    hp_gdf.loc[:, 'state'].replace({'-':'_', 'ü': 'ue', 'ä': 'ae', 'ö': 'oe'}, regex=True, inplace=True)
    hp_gdf.rename(columns={'avg_house_': 'hp', 'sqm_euro_v':'sqm_p', 'avg_apartm':'ap_size','no_appartm':'no_ap'}, inplace=True)

    print('Started calculating impact allocation')
    start = time.process_time()
    pa_gdf = gpd.read_file(f'{param_dict["gwa_placement_area"]}', crs=param_dict["epsg_general"])
    col_len = len(pa_gdf.columns)

    for buff in param_dict["impact_buffers"]:
        print(f'Calculating {buff}m impacts...')
        pa_buff = buffer_geoms(pa_gdf, buff , param_dict["epsg_distance_calc"])

        inter_list = []
        for z in osmfiles:
            # osm_list = []
            # for file in z:
            #print(f"Reading buildings data from {z}/{filename}")
            osm_gdf = gpd.read_file(f"zip://{z}/{filename}").loc[:,['geometry']].to_crs(param_dict["epsg_distance_calc"])
            osm_gdf = osm_gdf.set_geometry(osm_gdf.centroid).to_crs(param_dict["epsg_general"])
            #osm_list.append(osm_gdf)

            # # only consider residential areas
            gdf_res = gpd.read_file(f"zip://{z}/{param_dict['osm_file_names']['landuse']}").loc[:, ['fclass','geometry']]
            gdf_res = gdf_res.loc[(gdf_res['fclass'] == 'residential')]
            osm_gdf = gpd.clip(osm_gdf, gdf_res)

            #reg_gdf = pd.concat(osm_list)
            #dfsjoin = gpd.sjoin(pa_buff, reg_gdf, how="left") #Spatial join Points to polygons
            dfsjoin = gpd.sjoin(pa_buff, osm_gdf, how="left") #Spatial join Points to polygons
            counts = dfsjoin.groupby(dfsjoin.index)["index_right"].count().to_numpy()

            inter_list.append(counts) # add np.array of intersection count for file i to a list
        

        inter_vector = np.sum(inter_list, axis=0) # combine array list first to a single array and then sum over the relevant array column
        pa_gdf[f'imp_{buff}m'] = inter_vector
        print(f"Finished {buff}m impact zone. Time passed:", time.process_time() - start)

    # To avoid double counting of impacts subtract the larger buffer zone by the next smaller one
    # The smallest buffer zone impact count will simply be subtracted by 0 
    impacts_dc = pa_gdf.iloc[:, col_len:].copy().to_numpy()
    sub_array= np.hstack((np.zeros((impacts_dc.shape[0],1)), impacts_dc[:, :-1])) # adds zero column to numpy array in position 0
    corrected_impacts = impacts_dc - sub_array
    pa_gdf.iloc[:, col_len:] = corrected_impacts.astype(int)

    print('Adding house price information')
    joined_gdf = gpd.sjoin(pa_gdf, hp_gdf, how="inner", op="intersects")
    joined_gdf.drop(columns=['index_right'], inplace=True)
    joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep="first")] # drop duplicates keep first

    joined_gdf.to_file(f'{param_dict["optimization_file"]}', index=True)
    print('Finished generating impact allocation')
    print("Final Number of cells:", len(joined_gdf))

    return None


###############################################################################################################

param = config.general_settings()

###############################################################################################################

# Download OSM data:
# download_urls = ["https://download.geofabrik.de/europe/germany/baden-wuerttemberg.html", "https://download.geofabrik.de/europe/germany/bayern.html",
#                 "https://download.geofabrik.de/europe/germany/nordrhein-westfalen.html", "https://download.geofabrik.de/europe/germany.html"]

# print("Started downloading OSM data from https://download.geofabrik.de")
# for url in download_urls:
#     download_list = get_url_download_list(url, keyword="latest-free", extension=".zip")
#     download_files_from_list(download_list, param["osm_path"])

# print("Finished downloading OSM data")

###############################################################################################################

# Generate wind data input file:
# generate_gwa_input(param, param["gwa_in_file"], param["gwa_path"], "DEU", param['gwa_layers'], param['gwa_heights'])

###############################################################################################################

# # Create masked no-placement raster:
#create_nopa(param)

###############################################################################################################

# Convert to one single csv file
# Note Regrading iec layers: assumed turbine at 100m with 115m (IEC1), 126m (IEC2) and 136m (IEC3) radius

# band_names = ['ad_100', 'ad_150', 'iec1', 'iec2', 'iec3', 'pd_100', 'pd_150', 'ws_100', 'ws_150'] # bands are in alphabetical order
# raster2csv(f'{param["gwa_out_path"]}/{param["masked_file_name"]}.tif', f'{param["gwa_out_path"]}/{param["masked_file_name"]}.csv', band_names)

###############################################################################################################

# # Convert gwa_masked.csv DataFrame to a GeoDataFrame and clean locations using max altitude and min+max
# # wind speed requirements (also drop all empty cells)

# gwa_df = pd.read_csv(f'{param["gwa_out_path"]}/{param["masked_file_name"]}.csv')


# # Drop location if raster cell wind speed is not in the intervall where turbines are operational
# cut_in = 2.5
# cut_out = 34
# gwa_df = gwa_df.loc[(cut_in < gwa_df.ws_150) & (gwa_df.ws_150 < cut_out) & (cut_in < gwa_df.ws_100) & (gwa_df.ws_100 < cut_out)]

# # Generate GeoDataframe:
# gdf = gpd.GeoDataFrame(gwa_df, geometry=gpd.points_from_xy(gwa_df.lon, gwa_df.lat), crs=param["epsg_general"])

# # Drop turbines that are higher than a certain altitude (this altitude was obtained using the max altitude
# # of all existant turbine: value)

# max_alt = 1522
# gdf['altitude'] = get_gdf_raster_value(gdf, param["bkg_path"])
# print(len(gdf))
# gdf = gdf.loc[(gdf.altitude < max_alt)]
# print(len(gdf))
# gdf = gdf.drop(columns=["lon", "lat", "altitude"])

# gdf.to_file(f'{param["gwa_placement_area"]}', index=False)

###############################################################################################################

# Calculate cell impact numbers:
# calc_house_impact_allocation(param)

###############################################################################################################