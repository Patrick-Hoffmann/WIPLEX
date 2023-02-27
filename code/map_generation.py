import urllib.request
import os
os.environ["USE_PYGEOS"] = "0" 
import geopandas as gpd
from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import itertools
from urllib.parse import urljoin, urlparse
import time
import glob
import pandas as pd
from pathlib import Path
import rasterio
import rasterio.mask
from osgeo import gdal
from shutil import copyfile
import config

######################################################################################################

#-------------------- OSM download functions --------------------#

def get_url_download_list(website, keyword="", extension=".zip"):
    '''
    Function scrapes download urls from a given website.

    Parameters:
    ----------- 
        website : string
            url of the website from which download links should e scraped
        keyword: string
            string that should be matched (can be used for url sub-selection). If empty: all urls 
            with extension will be added
        extension: string
            filtype of the download files (in case only certain files should be downloaded)

    Returns:
    --------
        download_list : list
            list of download urls
    '''
    response = requests.get(website)
    soup = BeautifulSoup(response.text, "lxml")
    #table = soup.find('table', {'id':'subregions'})
    download_list = [urljoin(website, a['href']) for a in soup.find_all("a", href=re.compile(fr".*{keyword}(.*){extension}$"))]

    return download_list


def download_from_urllist(urllist, directory):
    '''
    Function downloads all files from a url list to a specific directory

    Parameters:
    ----------- 
        urllist : list
            list with urls to download
        directory: string
            file storage directory on local machine

    Returns:
    --------
        None          
    '''
    for url in urllist:
        time.sleep(6)
        filename = os.path.basename(urlparse(url).path)
        print(f"Downloading data from {url}...")
        try:
            res = requests.get(url)

            with open(f"{directory}/{filename}", "wb") as file:
                file.write(res.content)

        except Exception as ex:
            print(f"Error downloading {url}. Message: {ex}")

    return None

def download_OSM(urls, directory):
    """
    Function downloads zipped OSM shapefiles and places them in a given directory

    Parameters:
    ----------- 
        urls : list
            list with urls to download
        directory: string
            file storage directory on local machine

    """
    print("Started downloading OSM data from https://download.geofabrik.de")
    for url in urls:
        download_list = get_url_download_list(url, keyword="latest-free", extension=".zip")
        download_from_urllist(download_list, directory)
    print("Finished downloading OSM data")

#-------------------- GWA download and processing functions --------------------#

def download_gwa(outpath, wind_layer, country='DEU', height=100):
    '''
    Download Data from the Global Wind Atlas (GWA) Website 

    Parameters:
    ----------- 
        outpath : string
            directory for data storage
        wind_layer: string
            Name of the GWA wind layer that should be downloaded
        country : string
            iso3 code of the country for which to download the data
        height: int
            Number specifying the altitude for which to obtain a given wind layer (if applicable)
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
    for l in layers:
        if "IEC" in l:
            download_gwa(folder, l , country=iso3, height=h)
            time.sleep(10)

        else:
            for h in heights:
                download_gwa(folder, l , country=iso3, height=h)
                time.sleep(10)


    # stack gwa layers and create a multiband raster (important file list order influences band order!):
    gwa_list = glob.glob(f'{folder}/*inp.tif')

    # Resample the Raster into 500x500m size
    for file in gwa_list:
        filename = Path(file).stem
        gdal.Warp(f'{folder}/{filename}_proj.tif', file, dstSRS=f'EPSG:{param_dict["epsg_distance_calc"]}')
        gdal.Warp(f'{folder}/{filename}_res.tif', f'{folder}/{filename}_proj.tif', xRes=500, yRes=500, resampleAlg='bilinear')
        gdal.Warp(f'{folder}/{filename}_resampled.tif', f'{folder}/{filename}_res.tif', dstSRS=f'EPSG:{param_dict["epsg_general"]}')

        # Delete unnecessary files
        for temp in ['proj', 'res']:
            try:
                os.remove(f'{folder}/{filename}_{temp}.tif')
            except OSError:
                pass

    # Convert into a multiband tiff file
    gwa_list = glob.glob(f'{folder}/*resampled.tif')


    vrt = gdal.BuildVRT(f'{folder}/virtual_raster_tmp.vrt', gwa_list, separate=True)
    gdal.Translate(name_out, vrt)


#-------------------- Helper Functions for Map Generation --------------------#

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
    Function masks a raster file by a given GeoSeries of shapes using the inversion, cropping and intersection settings
    specified by the user.

    The parameters inversion, cropping and mask_by_intersection refer to options in the rasterio.mask module
    (https://rasterio.readthedocs.io/en/stable/api/rasterio.mask.html)

    '''
    with rasterio.open(raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=cropping, invert=inversion, all_touched=mask_by_intersection)
        out_meta = src.meta
        src.close() # has to be closed to allow overwriting the tmp file

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        out_image[out_image==0] = np.nan


        with rasterio.open(f'{output_path}/{filename}', "w", **out_meta) as dest:
            dest.write(out_image)

        return np.count_nonzero(~np.isnan(out_image[0]))


#-------------------- Main map generation function --------------------#

def generate_placement_area(param_dict, paths_dict):
    '''
    Function generates turbine placement area by masking the GWA raster with areas of non-placement defined by the
    user in param_dict and paths_dict. The remaining cells will be considered for turbine placement in the optimization exercise.

    Parameters
    ----------
        param_dict : dict
            parameter dictionary specified in the config.py file
        paths_dict : dict
            paths dictionary specified in the config.py file

    Returns
    -------
        None
    '''
    #------------- Mask raster with non-osm areas:

    for key in param_dict["other_nopa_keys"]:
        gdf = gpd.read_file(paths_dict["other_nopa"][key])

        if isinstance(param_dict["nopa_distance"][key], str): # buffer shapes with distances from column in shapefile (5D_Abstand)
            gdf_buff = gdf.copy().to_crs(param_dict["epsg_distance_calc"])
            gdf_buff['geometry'] = gdf_buff.apply(lambda x: x.geometry.buffer(int(x[param_dict["nopa_distance"][key]])), axis=1)
            gdf_buff = gdf_buff.to_crs(gdf.crs)

        elif param_dict["nopa_distance"][key] != 0:
            gdf_buff = buffer_geoms(gdf, param_dict["nopa_distance"][key], param_dict["epsg_distance_calc"])

        else: # dont buffer shapes
            gdf_buff = gdf


        nopa_shapes = gdf_buff.geometry

        # Use either existing temporary masked raster or initial raster file:
        if os.path.isfile(f'{paths_dict["gwa_out_path"]}/gwa_masked_tmp.tif'):
            gwa_path = f'{paths_dict["gwa_out_path"]}/gwa_masked_tmp.tif'
        else:
            gwa_path = f'{paths_dict["gwa_in_file"]}'

        # Mask the raster:
        if key == "region_border": 
            cell_count = mask_raster(gwa_path, nopa_shapes, paths_dict["gwa_out_path"], "gwa_masked_tmp.tif",
                                    inversion=param_dict["inversion"][key], cropping=param_dict["cropping"][key], 
                                    mask_by_intersection=False)
        else:
            cell_count = mask_raster(gwa_path, nopa_shapes, paths_dict["gwa_out_path"], "gwa_masked_tmp.tif",
                                    inversion=param_dict["inversion"][key], cropping=param_dict["cropping"][key], 
                                    mask_by_intersection=param_dict["intersection_mask"]) 

        print(f'Updated number of raster cells after dropping {key} data', cell_count)

    #------------- Mask raster with OSM areas:

    osmfiles = glob.glob(os.path.join(paths_dict["osm_path"], '*.zip'))

    for key in param_dict["osm_keys"]:
        filename = paths_dict["osm_file_names"][key]

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
                gdf_major = gdf.loc[gdf['fclass'].isin(param_dict["road_types"]["osm_major_roads"])]
                gdf_minor = gdf.loc[gdf['fclass'].isin(param_dict["road_types"]["osm_minor_roads"])]
                gdf_small = gdf.loc[gdf['fclass'].isin(param_dict["road_types"]["osm_small_roads"])]

                buff_major = buffer_geoms(gdf_major, param_dict["nopa_distance"][key][0], param_dict["epsg_distance_calc"])
                buff_minor = buffer_geoms(gdf_minor, param_dict["nopa_distance"][key][1], param_dict["epsg_distance_calc"])
                buff_small = buffer_geoms(gdf_small, param_dict["nopa_distance"][key][2], param_dict["epsg_distance_calc"])
                gdf_buff = gpd.GeoDataFrame(pd.concat([buff_major, buff_minor, buff_small], ignore_index=True), crs=buff_major.crs)

            else:
                gdf_buff = buffer_geoms(gdf, param_dict["nopa_distance"][key], param_dict["epsg_distance_calc"])


            nopa_shapes = gdf_buff.geometry


            if os.path.isfile(f'{paths_dict["gwa_out_path"]}/gwa_masked_tmp.tif'):
                gwa_path = f'{paths_dict["gwa_out_path"]}/gwa_masked_tmp.tif'
            else:
                gwa_path = f'{paths_dict["gwa_in_file"]}'

            cell_count = mask_raster(gwa_path, nopa_shapes, paths_dict["gwa_out_path"], "gwa_masked_tmp.tif", 
                                    inversion=param_dict["inversion"][key], cropping=param_dict["cropping"][key],
                                    mask_by_intersection=param_dict["intersection_mask"])


        print(f'Updated number of raster cells after dropping {key} data', cell_count)


    # Drop tmp file and store masked raster using the masked_file_name:
    os.replace(f'{paths_dict["gwa_out_path"]}/gwa_masked_tmp.tif', f'{paths_dict["gwa_out_path"]}/{paths_dict["masked_file_name"]}.tif')

    return None


#-------------------- Additonal Map generation functions: --------------------#

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

def create_placement_gdf(param_dict, paths_dict):
    """
    Function creates a geopandas.GeoDataFrame from a .csv file with lat and lon values. Additionally,
    data is cleaned by removes cells that lie above a certain maximum altitude and cells 
    with wind speeds above or below certain thresholds. 

    Parameters
    ----------
        param_dict : dict
            parameter dictionary specified in the config.py file
        paths_dict : dict
            paths dictionary specified in the config.py file
    """

    # Read in dataframe with masked turbine locations:
    gwa_df = pd.read_csv(f'{paths_dict["gwa_out_path"]}/{paths_dict["masked_file_name"]}.csv')

    # # Drop locations outside of cut-in and cut-out wind speed thresholds
    # gwa_df = gwa_df.loc[(param_dict["cut_in"] < gwa_df.ws_150) & (gwa_df.ws_150 < param_dict["cut_out"]) & 
    #          (param_dict["cut_in"] < gwa_df.ws_100) & (gwa_df.ws_100 < param_dict["cut_out"])]

    # Generate GeoDataframe:
    gdf = gpd.GeoDataFrame(gwa_df, geometry=gpd.points_from_xy(gwa_df.lon, gwa_df.lat), crs=param_dict["epsg_general"])

    # Drop turbines higher than a certain altitude
    gdf['altitude'] = get_gdf_raster_value(gdf, paths_dict["altitude_file"])
    gdf = gdf.loc[(gdf.altitude < param_dict["max_alt"])]
    gdf = gdf.drop(columns=["lon", "lat", "altitude"]) # drop unnecessary columns

    # Save file
    gdf.to_file(f'{paths_dict["gwa_placement_area"]}', index=False)

#-------------------- Impact allocation and house price functions --------------------#

# def add_house_impact_allocation(param_dict, paths_dict):
#     '''
#     Function calculates for a given set of impact zones (in meter) the number of affected houses per impact zone for each
#     cell in a given raster.

#     Parameters
#     ----------
#         param_dict : dict
#             parameter dictionary specified in the config.py file
#         paths_dict : dict
#             paths dictionary specified in the config.py file

#     Returns
#     -------
#         None
#     '''
#     osmfiles = glob.glob(os.path.join(paths_dict["osm_path"], '*.zip'))
#     filename = paths_dict["osm_file_names"]["buildings"]
#     #sub_files = np.array_split(osmfiles, 6)

#     # # Read in shapefile with house prices:
#     # hp_gdf = gpd.read_file(f'{paths_dict["house_prices"]}', crs=param_dict["epsg_general"])
#     # hp_gdf.loc[:, 'state'].replace({'-':'_', 'ü': 'ue', 'ä': 'ae', 'ö': 'oe'}, regex=True, inplace=True)
#     # hp_gdf.rename(columns={'avg_house_': 'hp', 'sqm_euro_v':'sqm_p', 'avg_apartm':'ap_size','no_appartm':'no_ap'}, inplace=True)

#     # Calculate number of affected houses per impact zone for each raster cell
#     print('Started calculating impact allocation')
#     start = time.process_time() # start time check for calculations
#     #pa_gdf = gpd.read_file(f'{paths_dict["inter_out"]}/opt_DEU_500_tmp.shp', crs=param_dict["epsg_general"])
#     #pa_gdf.drop(columns=["index"], inplace=True)
#     pa_gdf = gpd.read_file(f'{paths_dict["gwa_placement_area"]}', crs=param_dict["epsg_general"])
#     # col_order = list(pa_gdf_start.columns) + [item for item in list(pa_gdf.columns) if not item in list(pa_gdf_start.columns)]
#     # pa_gdf = pa_gdf[col_order]
#     col_len = len(pa_gdf.columns)

#     for buff in param_dict["impact_buffers"]:
#         print(f'Calculating {buff}m impacts...')
#         pa_buff = buffer_geoms(pa_gdf, buff , param_dict["epsg_distance_calc"])
#         print(f"Checkpoint Reached 0", time.process_time() - start)

#         inter_list = []
#         for z in osmfiles:
#             # osm_list = []
#             # for file in z:
#             #print(f"Reading buildings data from {z}/{filename}")
#             osm_gdf = gpd.read_file(f"zip://{z}/{filename}").loc[:,['geometry']].to_crs(param_dict["epsg_distance_calc"])
#             osm_gdf = osm_gdf.set_geometry(osm_gdf.centroid).to_crs(param_dict["epsg_general"])
#             print(f"Checkpoint Reached 1", time.process_time() - start)
#             #osm_list.append(osm_gdf)

#             # Only consider residential areas
#             gdf_res = gpd.read_file(f"zip://{z}/{paths_dict['osm_file_names']['landuse']}").loc[:, ['fclass','geometry']]
#             gdf_res = gdf_res.loc[(gdf_res['fclass'] == 'residential')]
#             osm_gdf = gpd.clip(osm_gdf, gdf_res)
#             print(f"Checkpoint Reached 2", time.process_time() - start)

#             #reg_gdf = pd.concat(osm_list)
#             #dfsjoin = gpd.sjoin(pa_buff, reg_gdf, how="left") #Spatial join Points to polygons
#             dfsjoin = gpd.sjoin(pa_buff, osm_gdf, how="left") #Spatial join Points to polygons
#             counts = dfsjoin.groupby(dfsjoin.index)["index_right"].count().to_numpy()
#             print(f"Checkpoint Reached 3", time.process_time() - start)
#             inter_list.append(counts) # add np.array of intersection count for file z to a list
        

#         inter_vector = np.sum(inter_list, axis=0) # combine array list first to a single array and then sum over the relevant array column
#         pa_gdf[f'imp_{buff}m'] = inter_vector # Add impact to output GeoDataFrame
#         pa_gdf.to_file(f'{paths_dict["inter_out"]}/opt_DEU_500_tmp.shp', index=True) # add saving after every  paths["inter_out"]
#         print(f"Finished {buff}m impact zone. Time passed:", time.process_time() - start)

#     # Correct impact double-counting:
#     # To avoid double counting of impacts subtract the larger buffer zone by the next smaller one
#     # The smallest buffer zone impact count will simply be subtracted by 0 
#     col_filter = [col for col in pa_gdf if col.startswith('imp_')]
#     impacts_dc =  pa_gdf.loc[:, col_filter].copy().to_numpy()
#     #impacts_dc = pa_gdf.iloc[:, col_len:].copy().to_numpy()
#     sub_array= np.hstack((np.zeros((impacts_dc.shape[0],1)), impacts_dc[:, :-1])) # adds zero column to numpy array in position 0
#     corrected_impacts = impacts_dc - sub_array
#     pa_gdf.loc[:, col_filter] = corrected_impacts.astype(int)
#     # pa_gdf.iloc[:, col_len:] = corrected_impacts.astype(int)

#     # # Add house price information to each cell
#     # print('Adding house price information')
#     # joined_gdf = gpd.sjoin(pa_gdf, hp_gdf, how="inner", op="intersects")
#     # joined_gdf.drop(columns=['index_right'], inplace=True)
#     # joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep="first")] # drop duplicates keep first

#     # Save file:
#     # joined_gdf.to_file(f'{paths_dict["optimization_file"]}', index=True)
#     pa_gdf.to_file(f'{paths_dict["optimization_file"]}')
#     #pa_gdf.to_file(f'{paths_dict["optimization_file"]}', index=True)
#     print('Finished generating impact allocation')

#     return None


# def add_house_impact_allocation(param_dict, paths_dict):
#     '''
#     Function calculates for a given set of impact zones (in meter) the number of affected houses per impact zone for each
#     cell in a given raster.

#     Parameters
#     ----------
#         param_dict : dict
#             parameter dictionary specified in the config.py file
#         paths_dict : dict
#             paths dictionary specified in the config.py file

#     Returns
#     -------
#         None
#     '''
#     building_files = glob.glob(os.path.join(paths_dict["osm_out_path"], '*.geojson'))

#     # Calculate number of affected houses per impact zone for each raster cell
#     print('Started calculating impact allocation')
#     start = time.process_time() # start time check for calculations
#     #pa_gdf = gpd.read_file(f'{paths_dict["inter_out"]}/opt_DEU_500_tmp.shp', crs=param_dict["epsg_general"])
#     #pa_gdf.drop(columns=["index"], inplace=True)
#     pa_gdf = gpd.read_file(f'{paths_dict["gwa_placement_area"]}', crs=param_dict["epsg_general"])
#     # col_order = list(pa_gdf_start.columns) + [item for item in list(pa_gdf.columns) if not item in list(pa_gdf_start.columns)]
#     # pa_gdf = pa_gdf[col_order]
#     col_len = len(pa_gdf.columns)

#     for buff in param_dict["impact_buffers"]:
#         print(f'Calculating {buff}m impacts...')
#         pa_buff = buffer_geoms(pa_gdf, buff , param_dict["epsg_distance_calc"])
#         print(f"Checkpoint Reached 0", time.process_time() - start)

#         inter_list = []
#         for z in building_files:
#             # osm_list = []
#             # for file in z:
#             #print(f"Reading buildings data from {z}/{filename}")
#             osm_gdf = gpd.read_file(z)
#             print(f"Checkpoint Reached 1", time.process_time() - start)
#             #osm_list.append(osm_gdf)

#             #reg_gdf = pd.concat(osm_list)
#             join_list = []
#             for i in dfs:
#                 dfsjoin = gpd.sjoin(i, osm_gdf, how="left") #Spatial join Points to polygons
#                 counts = dfsjoin.groupby(dfsjoin.index)["index_right"].count().to_numpy()
#                 join_list.append(counts)
#             dfsjoin = gpd.sjoin(pa_buff, osm_gdf, how="left") #Spatial join Points to polygons
#             counts = dfsjoin.groupby(dfsjoin.index)["index_right"].count().to_numpy()
#             print(f"Checkpoint Reached 3", time.process_time() - start)
#             inter_list.append(counts) # add np.array of intersection count for file z to a list
        

#         inter_vector = np.sum(inter_list, axis=0) # combine array list first to a single array and then sum over the relevant array column
#         pa_gdf[f'imp_{buff}m'] = inter_vector # Add impact to output GeoDataFrame
#         #pa_gdf.to_file(f'{paths_dict["inter_out"]}/opt_DEU_500_tmp.shp', index=True) # add saving after every  paths["inter_out"]
#         print(f"Finished {buff}m impact zone. Time passed:", time.process_time() - start)

#     # Correct impact double-counting:
#     # To avoid double counting of impacts subtract the larger buffer zone by the next smaller one
#     # The smallest buffer zone impact count will simply be subtracted by 0 
#     col_filter = [col for col in pa_gdf if col.startswith('imp_')]
#     impacts_dc =  pa_gdf.loc[:, col_filter].copy().to_numpy()
#     #impacts_dc = pa_gdf.iloc[:, col_len:].copy().to_numpy()
#     sub_array= np.hstack((np.zeros((impacts_dc.shape[0],1)), impacts_dc[:, :-1])) # adds zero column to numpy array in position 0
#     corrected_impacts = impacts_dc - sub_array
#     pa_gdf.loc[:, col_filter] = corrected_impacts.astype(int)
#     # pa_gdf.iloc[:, col_len:] = corrected_impacts.astype(int)

#     # # Add house price information to each cell
#     # print('Adding house price information')
#     # joined_gdf = gpd.sjoin(pa_gdf, hp_gdf, how="inner", op="intersects")
#     # joined_gdf.drop(columns=['index_right'], inplace=True)
#     # joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep="first")] # drop duplicates keep first

#     # Save file:
#     # joined_gdf.to_file(f'{paths_dict["optimization_file"]}', index=True)
#     pa_gdf.to_file(f'{paths_dict["optimization_file"]}')
#     #pa_gdf.to_file(f'{paths_dict["optimization_file"]}', index=True)
#     print('Finished generating impact allocation')

#     return None

def calculate_impact_zones(param_dict, paths_dict):
    '''
    Function counts the number of affected buildings for a GeoDataFrame with wind turbine locations given a set of 
    impact zones in meter. File locations and the set of impact zones should be specified in the config.py file before
    running this function.

    Parameters
    ----------
        param_dict : dict
            parameter dictionary specified in the config.py file
        paths_dict : dict
            paths dictionary specified in the config.py file

    Returns
    -------
        None
    '''
    print('Started calculating impact allocation')
    start = time.process_time() # start time check for calculations
    
    building_files = glob.glob(os.path.join(paths_dict["osm_out_path"], '*.geojson'))
    pa_gdf = gpd.read_file(f'{paths_dict["gwa_placement_area"]}', crs=param_dict["epsg_general"])
    

    for num, z in enumerate(building_files):
        print(f"Started calculating distance impacts for building batch {num + 1}/{len(building_files)}")
        osm_gdf = gpd.read_file(z)

        inter_list = []
        for buff in param_dict["impact_buffers"]:
            print(f"Evaluating buffer zone {buff}m...")
            pa_buff = buffer_geoms(pa_gdf, buff , param_dict["epsg_distance_calc"])

            dfs = np.array_split(pa_buff, 8)

            count_list = []
            for i in dfs:
                dfsjoin = gpd.sjoin(i, osm_gdf, how="left") #Spatial join Points to polygons
                counts = dfsjoin.groupby(dfsjoin.index)["index_right"].count().to_numpy()
                count_list.append(counts)
            count_list = list(itertools.chain(*count_list))
            inter_list.append(count_list)

        
        np.save(f'{paths_dict["osm_out_path"]}/impacts_{num + 2}', np.array(inter_list))
        #impact_list.append(np.array(inter_list))
        print(f"Finished {num + 1}. batch. Time passed:", time.process_time() - start)

def add_impact_zones(param_dict, paths_dict):
    impact_files = glob.glob(os.path.join(paths_dict["osm_out_path"], '*.npy'))
    impacts_list = [np.load(file) for file in impact_files]

    # Correct impact double-counting:
    # To avoid double counting of impacts subtract the larger buffer zone by the next smaller one
    # The smallest buffer zone impact count will simply be subtracted by 0 
    impacts_dc = np.sum(impacts_list, axis=0).T
    sub_array= np.hstack((np.zeros((impacts_dc.shape[0],1)), impacts_dc[:, :-1])) # adds zero column to numpy array in position 0
    corrected_impacts = impacts_dc - sub_array

    # Add impact zones to dataframe:
    names = [f"imp_{buff}" for buff in param_dict["impact_buffers"]]
    pa_gdf = gpd.read_file(f'{paths_dict["gwa_placement_area"]}', crs=param_dict["epsg_general"])
    pa_gdf[names] = corrected_impacts.astype(int)


    # Save file:
    pa_gdf.to_file(f'{paths_dict["optimization_file"]}')
    print('Finished generating impact allocation')
    

def add_house_price_info(param_dict, paths_dict):
    """
    Function adds house price information to optimization GeoDataFrame (ogdf) based on an inner spatial
    join with the ogdf rows (Expl.: Information is added to row if geometry of the ogdf is inside of the house price
    region of the house price gdf)

    Parameters
    ----------
        param_dict : dict
            parameter dictionary specified in the config.py file
        paths_dict : dict
            paths dictionary specified in the config.py file
    """
    # Read in shapefile with house prices:
    hp_gdf = gpd.read_file(f'{paths_dict["house_prices"]}', crs=param_dict["epsg_general"])
    hp_gdf.loc[:, 'state'].replace({'-':'_', 'ü': 'ue', 'ä': 'ae', 'ö': 'oe'}, regex=True, inplace=True)
    hp_gdf.rename(columns={'avg_house_': 'hp', 'sqm_euro_v':'sqm_p', 'avg_apartm':'ap_size','no_appartm':'no_ap'}, inplace=True)

    # Read in shapefile with turbine locations:
    pa_gdf = gpd.read_file(f'{paths_dict["optimization_file"]}', crs=param_dict["epsg_general"])

    # Add house price information to each raster cell
    print('Adding house price information')
    joined_gdf = gpd.sjoin(pa_gdf, hp_gdf, how="inner", predicate="intersects")
    joined_gdf.drop(columns=['index_right'], inplace=True)
    joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep="first")] # drop duplicates keep first

    # Save file:
    joined_gdf.to_file(f'{paths_dict["optimization_file"]}', index=True)
    print("Final Number of cells:", len(joined_gdf))

######################################################################################################

if __name__ == "__main__":

    #-------------------- Initialize param and paths dictionaries --------------------#

    WIPLEX_config = config.WIPLEX_settings()
    WIPLEX_config.initialize_config()
    param = WIPLEX_config.param
    paths = WIPLEX_config.paths

    #-------------------- Download OSM data --------------------#

    # download_urls = ["https://download.geofabrik.de/europe/germany/baden-wuerttemberg.html", "https://download.geofabrik.de/europe/germany/bayern.html",
    #                 "https://download.geofabrik.de/europe/germany/nordrhein-westfalen.html", "https://download.geofabrik.de/europe/germany.html"]
    
    # download_OSM(download_urls, paths["osm_path"])

    # Alternative: Use the Pyrosm package to download pbf files (Note: This requires changes in the create_nopa
    # and calc_house_impact_allocation() functions to read in protobuf instead of zip)
    # Link: https://pyrosm.readthedocs.io/en/latest/basics.html#Protobuf-file:-What-is-it-and-how-to-get-one?
    

    #-------------------- Generate GWA input file --------------------#

    # generate_gwa_input(param, paths["gwa_in_file"], paths["gwa_path"], "DEU", param['gwa_layers'], param['gwa_heights'])

    #-------------------- Create masked placement raster  --------------------#

    # generate_placement_area(param, paths)

    #-------------------- Convert .tif to .csv file  --------------------#
    # Note Regrading iec layers: assumed turbine at 100m with 115m (IEC1) radius, 126m (IEC2) and 136m (IEC3) radius

    # Name raster band columns in the .csv file (Note: Layers are in alphabetical order (based on param['gwa_layers']))
    # band_names = ['ad_100', 'ad_150', 'ad_200', 'ad_50', 'wb_A_100', 'wb_A_150', 'wb_A_200', 'wb_A_50', 'wb_k_100', 'wb_k_150', 'wb_k_200', 'wb_k_50', 'ws_100', 'ws_150', "ws_200", "ws_50"]
    # # band_names = ['ad_100', 'ad_150', 'ad_200', 'ad_50','iec1', 'iec2', 'iec3', 'ws_100', 'ws_150', "ws_200", "ws_50"] 
    # raster2csv(f'{paths["gwa_out_path"]}/{paths["masked_file_name"]}.tif', f'{paths["gwa_out_path"]}/{paths["masked_file_name"]}.csv', band_names)

    #-------------------- Create placement GeoDataFrame--------------------#
    # create_placement_gdf(param, paths)

    #-------------------- Calculate cell impact numbers --------------------#
    # calculate_impact_zones(param, paths) # Calculates impact zones in chunks
    # add_impact_zones(param, paths)

    #-------------------- Add house price data --------------------#
    add_house_price_info(param, paths)

    