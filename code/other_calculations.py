import glob
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import config
from tm_optimization import calc_power_cost

######################################################################################################

def buffer_geoms(gdf_in, buffer, proj_epsg=25832):
    '''
    Function calculates a buffered GeoSeries from the geometry column of a geopandas DataFrame
    in a given meter based coordinate reference system (crs) and converts it back to the orginial crs
    '''
    init_crs = gdf_in.crs
    gdf_in = gdf_in.to_crs(proj_epsg)
    buffered = gpd.GeoDataFrame(geometry = gdf_in.geometry.buffer(buffer)).to_crs(init_crs)

    return buffered

def add_house_impact_allocation(in_path, out_path, param_dict, paths_dict):
    '''
    Function calculates for a given set of impact zones (in meter) the number of affected houses per impact zone for each
    cell in a given raster.

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
    osmfiles = glob.glob(os.path.join(paths_dict["osm_path"], '*.zip'))
    filename = paths_dict["osm_file_names"]["buildings"]
    #sub_files = np.array_split(osmfiles, 6)

    # Calculate number of affected houses per impact zone for each raster cell
    print('Started calculating impact allocation')
    start = time.process_time() # start time check for calculations
    pa_gdf = gpd.read_file(in_path, crs=param_dict["epsg_general"])
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

            # Only consider residential areas
            gdf_res = gpd.read_file(f"zip://{z}/{paths_dict['osm_file_names']['landuse']}").loc[:, ['fclass','geometry']]
            gdf_res = gdf_res.loc[(gdf_res['fclass'] == 'residential')]
            osm_gdf = gpd.clip(osm_gdf, gdf_res)

            #reg_gdf = pd.concat(osm_list)
            #dfsjoin = gpd.sjoin(pa_buff, reg_gdf, how="left") #Spatial join Points to polygons
            dfsjoin = gpd.sjoin(pa_buff, osm_gdf, how="left") #Spatial join Points to polygons
            counts = dfsjoin.groupby(dfsjoin.index)["index_right"].count().to_numpy()

            inter_list.append(counts) # add np.array of intersection count for file z to a list
        

        inter_vector = np.sum(inter_list, axis=0) # combine array list first to a single array and then sum over the relevant array column
        pa_gdf[f'imp_{buff}m'] = inter_vector # Add impact to output GeoDataFrame
        print(f"Finished {buff}m impact zone. Time passed:", time.process_time() - start)

    # Correct impact double-counting:
    # To avoid double counting of impacts subtract the larger buffer zone by the next smaller one
    # The smallest buffer zone impact count will simply be subtracted by 0 
    impacts_dc = pa_gdf.iloc[:, col_len:].copy().to_numpy()
    sub_array= np.hstack((np.zeros((impacts_dc.shape[0],1)), impacts_dc[:, :-1])) # adds zero column to numpy array in position 0
    corrected_impacts = impacts_dc - sub_array
    pa_gdf.iloc[:, col_len:] = corrected_impacts.astype(int)

    # Save file:
    # joined_gdf.to_file(f'{paths_dict["optimization_file"]}', index=True)
    pa_gdf.to_file(out_path, index=True)
    print('Finished generating impact allocation')

    return None

def add_house_price_info(in_path, out_path, param_dict, paths_dict):
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
    hp_gdf = gpd.read_file(f'{paths_dict["house_price_file"]}', crs=param_dict["epsg_general"])
    hp_gdf.loc[:, 'state'].replace({'-':'_', 'ü': 'ue', 'ä': 'ae', 'ö': 'oe'}, regex=True, inplace=True)
    hp_gdf.rename(columns={'avg_house_': 'hp', 'sqm_euro_v':'sqm_p', 'avg_apartm':'ap_size','no_appartm':'no_ap'}, inplace=True)

    # Read in shapefile with turbine locations:
    pa_gdf = gpd.read_file(in_path, crs=param_dict["epsg_general"])

    # Add house price information to each raster cell
    print('Adding house price information')
    joined_gdf = gpd.sjoin(pa_gdf, hp_gdf, how="inner", op="intersects")
    joined_gdf.drop(columns=['index_right'], inplace=True)
    joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep="first")] # drop duplicates keep first

    # Save file:
    joined_gdf.to_file(out_path, index=True)
    print("Final Number of cells:", len(joined_gdf))

def round_down(num, divisor):
    '''
    Function for rounding down a number (num) depening on a divisor.
    (Eg. With divisor 10 the number is rounded down to the nearest 10: 26-->20)
    '''
    return num - (num%divisor)

def sound_pressure_level_uj(sound_power, distances, hub_height):
    '''
    Function calculating the sound pressure level as defined in the paper 
    by Jensen et al. (2014)
    '''
    return sound_power - 10*np.log10(np.square(distances) + np.square(hub_height)) - 11 + 1.5 - (2/1000) * np.sqrt((np.square(distances) + np.square(hub_height)))


def calculate_externalities(gdf_path, study):
    """
    """

    gdf = gpd.read_file(gdf_path, crs=param["epsg_general"])

    sqm_p, ap_size, no_ap = gdf['sqm_p'].to_numpy(), gdf['ap_size'].to_numpy(), gdf['no_ap'].to_numpy()
    

    imp_list = [f"imp_{i}m" for i in list(range(500, 2750, 250))]

    impacts = gdf.loc[:, imp_list].to_numpy()

    hp = sqm_p * ap_size * no_ap # simple average sqm price 2077 per m^2 (replace for relevant calculation)


    impacts_hp = impacts * hp.reshape(-1, 1) # calculates total of house prices for the affected buildings in each zone


    if study == 'jensen':
        dmg = {0: 0.0, 10: 0.0, 20: 0.0307, 30: 0.055, 40: 0.0669, 50: 0.0669, 60: 0.0669}
        
        distance = np.array(range(500, 2750, 250)) - 125 # average distance of a cell within a certain distance range
        # hub_height = np.array([[100], [150]])
        # sound_power = np.array([[105.5], [106.1]]) #maximum sound power of E-115 and E-126EP3

        # Calculation with existing turbines:
        hub_height = gdf.loc[:, "hub_height"].to_numpy().reshape(-1,1)
        sound_power = 98


        DB = round_down(sound_pressure_level_uj(sound_power, distance, hub_height), 10)

        
        cn = DB.copy()
        for noise_level in list(dmg.keys()):
            cn[cn == noise_level] = dmg[noise_level]

        cost_vis = {'500': (0.0315 + 22.5 * 0.0024),
                    '750': (0.0315 + 20.0 * 0.0024),
                    '1000': (0.0315 + 17.5 * 0.0024),
                    '1250': (0.0315 + 15.0 * 0.0024),
                    '1500': (0.0315 + 12.5 * 0.0024),
                    '1750': (0.0315 + 10.0 * 0.0024),
                    '2000': (0.0315 + 7.5 * 0.0024),
                    '2250': (0.0315 + 5.0 * 0.0024),
                    '2500': (0.0315 + 2.5 * 0.0024),
                    } #'2750': (0.0315 + 0.0 * 0.0024), '3000': (0.0315 + 0.0 * 0.0024)

        cv = np.array(list(cost_vis.values()))
        cost_ext = cn + cv

    elif study == 'dk':
        dk_dmg = {250: 0.026, 500: 0.026, 750: 0.025, 1000: 0.021, 1250: 0.019, 1500: 0.019, 1750: 0.015, 2000: 0.0, 2250: 0.0}
        cost_ext = np.array([list(dk_dmg.values())])
        #cost_ext = np.array([list(dk_dmg.values()), list(dk_dmg.values())])

    # install_cost = np.array([[(990 + 387 + 56 ) * 3000.0, 
    #                           (1180 + 387 + 56) * 4200.0]]) # from windguard study

    install_cost = np.array([[(990 + 387 + 56 ) * 3000.0]]) # from windguard study

    if study == "noext":
        external_cost = np.zeros((len(gdf), 2))
    else:
        external_cost = np.round(impacts_hp @ cost_ext.T, 2)

    total_cost = external_cost + install_cost

    #print(impacts_hp[0] * cost_ext)
    installed_cap = np.sum(gdf["p_rated"])
    #print(external_cost[0])
    return external_cost, installed_cap, total_cost


if __name__ == "__main__":
    
    #-------------------- Initialize param and paths dictionaries --------------------#

    WIPLEX_config = config.WIPLEX_settings()
    WIPLEX_config.initialize_config()
    param = WIPLEX_config.param
    paths = WIPLEX_config.paths

    #----------------------- Calculate Existing Turbine Externalities -----------------------#
    out_file = "E:/Master_Thesis/WIPLEX/Database/Intermediate Outputs/ecost_existing/existing_externalities.shp"

    # add_house_impact_allocation("E:/Master_Thesis/WIPLEX/Database/Input Files/existing_turbines/existing_turbines_extended.shp", out_file, param, paths)
    # add_house_price_info(out_file, out_file, param, paths)


    # Calculate Externalities:
    gdf = gpd.read_file(out_file, crs=param["epsg_general"])
    #print(gdf.columns)    

    turbine_costs, installed_cap, total_cost = calculate_externalities(out_file, "jensen")
    ext_estimate = np.sum(turbine_costs, axis=0)[0] /1000000000
    total_cost_estimate = np.sum(total_cost, axis=0)[0] /1000000000
    print("Externalities (EUR Billion): ", ext_estimate)
    print("Total Cost (EUR Billion): ", total_cost_estimate)
    print("Externality Share:", ext_estimate/total_cost_estimate)