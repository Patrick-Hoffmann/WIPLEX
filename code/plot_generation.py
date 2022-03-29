import rasterio
import rasterio.mask
import rasterio.plot
import numpy as np 
import geopandas as gpd 
import matplotlib.pyplot as plt 
import matplotlib
import glob
#import pickle
import config
from pathlib import Path
import pickle5 as pickle
import pandas as pd
from tm_optimization import calc_power_cost
from mpl_toolkits.axes_grid1 import make_axes_locatable


def buffer_geoms(gdf_in, buffer, proj_epsg=25832):
    '''
    Function calculates a buffered GeoSeries from the geometry column of a geopandas DataFrame
    in a given meter based coordinate reference system (crs) and converts it back to the orginial crs
    '''
    init_crs = gdf_in.crs
    gdf_in = gdf_in.to_crs(proj_epsg)
    buffered = gpd.GeoDataFrame(geometry = gdf_in.geometry.buffer(buffer)).to_crs(init_crs)

    return buffered

def mask_raster(raster_path, shapes, output_path, filename, inversion=True, cropping=False, mask_by_intersection=True):
    '''
    '''
    with rasterio.open(raster_path) as src:
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



param = config.general_settings()
region_border = gpd.read_file(param["gadm_path"])
state_border = gpd.read_file(r"D:\Master_Thesis\Database\Input Files\regions\gadm36_DEU_1.shp")
county_border = gpd.read_file(r"D:\Master_Thesis\Database\Input Files\regions\gadm36_DEU_2.shp")

layer_name = "wind_speed_DEU_100_inp_resampled"

# #Clip raster layers of interest
# reg_shape = region_border.geometry

# raster = mask_raster(f'{param["gwa_path"]}/{layer_name}.tif', reg_shape, param["gwa_clipped"], f"{layer_name}_clipped.tif",
#                     inversion=param["inversion"]["region_border"], cropping=param["cropping"]["region_border"], 
#                     mask_by_intersection=False)


# wind_layer = rasterio.open(f'{param["gwa_clipped"]}/{layer_name}_clipped.tif')



# my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","greenyellow", "red"])

# #######################################################################################################

# im_ratio = wind_layer.shape[0]/wind_layer.shape[1]

# fig, ax = plt.subplots(figsize=(15, 15))
# raster_plot = rasterio.plot.show(wind_layer, ax=ax, cmap=my_cmap)
# region_border.plot(ax=ax, color='none', edgecolor='none')
# image_hidden = ax.get_images()[0]
# fig.colorbar(image_hidden, ax=ax, fraction=0.046*im_ratio, pad=0.04, aspect=40) # .set_label('Label name',size=18)
# ax.set_title('Wind speed in Germany (in m/s, height: 100m)', fontsize=20, pad=40, weight='bold')
# ax.axis('off')
# plt.savefig(f'{param["plot_path"]}/{layer_name}.png', dpi=300, bbox_inches = 'tight')

# ######################################################################################################

ex_turbines = gpd.read_file(param["mastr_path"])
turbines_plot = buffer_geoms(ex_turbines, 250, proj_epsg=param["epsg_distance_calc"])

fig, ax = plt.subplots(figsize=(15, 15))
turbines_plot.plot(ax=ax, color='red', markersize=0.01)
region_border.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8)
state_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.8)
county_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.6)
ax.set_title(f'Existing wind turbines in Germany (count: {len(ex_turbines)})', fontsize=20, pad=30, weight='bold')
ax.axis('off')
plt.savefig(f'{param["plot_path"]}/existing_turbines.png', dpi=300, bbox_inches = 'tight')

# ######################################################################################################

# placement_area = gpd.read_file(r"D:\Master_Thesis\Database\Intermediate Outputs\opt_DEU_500_nointer_res.shp")
# pa_plot = buffer_geoms(placement_area, 250, proj_epsg=param["epsg_distance_calc"])
# fig, ax = plt.subplots(figsize=(15, 15))
# pa_plot.plot(ax=ax, facecolor='blue')
# region_border.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
# state_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=1)
# county_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.8)
# ax.set_title(f'Placement cell locations (count: {len(placement_area)})', fontsize=20, pad=30, weight='bold')
# ax.axis('off')
# plt.savefig(f'{param["plot_path"]}/placement_cells.png', dpi=300, bbox_inches = 'tight')

#######################################################################################################
##################################### House Price Plots ###############################################
#######################################################################################################

# my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow", "orange","red"])
# gdf_house = gpd.read_file(param["house_prices"], crs=param["epsg_general"])
# gdf_house["ap_mil"] = gdf_house["avg_house_"] * 1e-6

# fig, ax = plt.subplots(figsize=(15, 15))
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=-2)
# region_border.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
# state_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.5)
# county_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.2)
# houses = gdf_house.plot(ax=ax, column="ap_mil", cmap=my_cmap, legend=True, cax=cax)
# #ax.set_title("Average $\mathregular{m^2}$ price for residential properties \n in all German counties (in €)", fontsize=20, pad=30, weight='bold')
# ax.axis('off')
# plt.savefig(f'{param["plot_path"]}/house_prices.png', dpi=300, bbox_inches = 'tight')


#######################################################################################################
##################################### Population Graphs ###############################################
#######################################################################################################

# my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkgreen", "green", "lightgreen", "yellowgreen", "yellow", "gold", "orange", "darkorange",
#     "red", "darkred"])
# gdf_pop = gpd.read_file("D:/Master_Thesis/Database/Input Files/population/county_population.shp", crs=param["epsg_general"])

# fig, ax = plt.subplots(figsize=(15, 15))
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=-2)

# bounds = [0, 100, 200, 400, 600, 800, 1000, 2000, 3000, 4000]
# norm = matplotlib.colors.BoundaryNorm(bounds, my_cmap.N)

# op_dens = gdf_pop.plot(ax=ax, column="pop_d_km2", cmap=my_cmap, legend=True, cax=cax, norm=norm)
# region_border.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
# state_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.5)
# county_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.2)
# #ax.set_title("Population Density (per $\mathregular{km^2}$) by county", fontsize=20, pad=30, weight='bold')
# ax.axis('off')
# plt.savefig(f'{param["plot_path"]}/population_density.png', dpi=300, bbox_inches = 'tight')

#######################################################################################################
##################################### Expansion Graphs ################################################
#######################################################################################################

def generate_expansion_maps(gdf, opt_file, df_final, region_border, plot_dir):
    '''
    Function genenerates location maps with all optimized turbine locations given in the opt_file.

    Parameters
    ----------
        gdf: geopandas.GeoDataFrame
            Dataframe containing all possible wind turbine locations

        opt_file: str
            path to the optimization file with the selected wind turbine locations for each scenario
        
        param: dict
            dictionary with configuration settings. Relevant for the Output location of the plots

    '''
    opt_result = pickle.load(open(file, "rb"))
    filename_split = Path(file).stem.split('_')
    power_calc = filename_split[1]
    ext_type = filename_split[2]

    fig, ax = plt.subplots(figsize=(15, 15))
    for scenario in [25000, 50000, 100000]: 
        rated_power = df_final[(df_final["Expansion Goal"] == scenario) & (df_final["Power Calculation"] == power_calc) & 
                    (df_final["Externality Assumption"] == ext_type)].iloc[0]["Expansion Goal (rated)"]
        selection = np.sum(opt_result[scenario], axis= 1).astype(bool)
        gdf_sel = gdf[selection]
        gdf_plot = buffer_geoms(gdf_sel, 250, proj_epsg=param["epsg_distance_calc"])
        gdf_plot.plot(ax=ax, facecolor='blue')
        region_border.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
        state_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=1)
        county_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.5)
        # ax.set_title(f'Placement locations {scenario}MW ({rated_power}MW rated power) expansion target \n Scenario: {ext_type} cost assumption with {power_calc} calculation', 
        #     fontsize=20, pad=30, weight='bold')
        ax.axis('off')
        #print(f"Saving to file: {plot_dir}/locations_{power_calc}_{ext_type}_{scenario}MW.png")
        plt.savefig(f'{plot_dir}/locations_{power_calc}_{ext_type}_{scenario}MW.png', dpi=300, bbox_inches = 'tight')
        plt.cla()

# location_file = "D:/Master_Thesis/Database/Intermediate Outputs/opt_DEU_500_nointer_nores.shp"
# final_result = pd.read_excel("D:/Master_Thesis/Database/Final Outputs/Optimization_results_nores_500_03_2022/optimization_results_nores_500.xlsx")

# gdf_in = gpd.read_file(location_file, crs=param["epsg_general"])
# gdf_in.rename(columns = {'sqm_euro_v':'sqm_p', 'avg_apartm':'ap_size','no_appartm':'no_ap'}, inplace = True)
# opt_files = glob.glob('D:/Master_Thesis/Database/Final Outputs/Optimization_results_nores_500_03_2022/*.pickle')
# plot_path = "D:/Master_Thesis/Database/Final Outputs/Optimization_results_nores_500_03_2022/plots"

# for file in opt_files:
#     if "wpf" in file:
#         generate_expansion_maps(gdf_in, file, final_result, region_border, plot_path)

#######################################################################################################
##################################### Externality Share Graphs ########################################
#######################################################################################################

def externality_county_map(gdf, file, house_gdf, border_region, scenarios, plot_dir, max_scale, per_capita):
    '''
    '''
    opt_result = pickle.load(open(file, "rb"))
    filename_split = Path(file).stem.split('_')
    power_calc = filename_split[1]
    ext_type = filename_split[2]

    if ext_type=="noext":
        ec = False
        scale_value = max_scale[0]
    else:
        ec = True
        scale_value = max_scale[0]

    total_cost, wpc, ext_cost, proj_cost = calc_power_cost(gdf, power_calc, study="jensen", ec=True)

    for scenario in scenarios:
        selection = np.sum(opt_result[scenario], axis= 1).astype(bool)
        # Calculate externality cost for each chosen cell:
        cell_ext_cost = np.sum(opt_result[scenario] * ext_cost, axis= 1)
        cell_ext_cost = cell_ext_cost[selection]

        # Select chosen cells from gdf:
        gdf_sel = gdf.copy()[selection]

        # Add externality cost to selection gdf:
        #gdf_sel["ext_cost_share"] = cell_ext_cost/np.sum(cell_ext_cost)
        gdf_sel["ext_cost_share"] = cell_ext_cost * 1e-9

        gdf_sum = gdf_sel.groupby("county")["ext_cost_share"].sum()
        gdf_plot = house_gdf.merge(gdf_sum, on="county", how="left").fillna(0)

        if per_capita:
            gdf_plot["ext_cost_share"] = gdf_plot["ext_cost_share"] * 1e9/ gdf_plot["pop_count"]
            extension ="_pc"
        else:
            extension =""


        fig, ax = plt.subplots(figsize=(15, 15))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=-1.5)

        gdf_plot.plot(ax=ax, column="ext_cost_share", cmap=my_cmap, legend=True, cax=cax, vmax=scale_value)
        region_border.plot(ax=ax, color='none', edgecolor='black', linewidth=0.4)
        state_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.3)
        county_border.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.2)
        # ax.set_title(f"Externality Distribution in Germany (in € per capita) \n scenario: {scenario}MW, ext: {ext_type}, calc: {power_calc}", 
        #     fontsize=20, pad=30, weight='bold')
        ax.axis('off')
        plt.savefig(f'{plot_dir}/ext_distribution_{power_calc}_{ext_type}_{scenario}MW{extension}.png', dpi=300, bbox_inches = 'tight')


# house_gdf = gpd.read_file(param["house_prices"], crs=param["epsg_general"])[["county", "pop_count", "geometry"]]

# gdf_in = gpd.read_file("D:/Master_Thesis/Database/Intermediate Outputs/opt_DEU_500_nointer_nores.shp", crs=param["epsg_general"])
# gdf_in.rename(columns = {'sqm_euro_v':'sqm_p', 'avg_apartm':'ap_size','no_appartm':'no_ap'}, inplace = True)
# opt_files = glob.glob('D:/Master_Thesis/Database/Final Outputs/Optimization_results_nores_500_03_2022/*.pickle')

# my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#fdee73", "orange","red"])
# plot_path = "D:/Master_Thesis/Database/Final Outputs/Optimization_results_nores_500_03_2022/plots"

# for file in opt_files:
#     if "wpf" in file:
#         externality_county_map(gdf_in, file, house_gdf, region_border, [25000, 50000, 100000], plot_path, max_scale=[80], per_capita=False)
#     else:
#         pass

#######################################################################################################
##################################### Supply Curve Graphs #############################################
#######################################################################################################

def plot_supply_curve(df, column_x, column_y, title, xlab, ylab, color_dict, out_path, out_name):
    '''
    Function generates a line curve plot for two columns of a given pandas DataFrame. If subset is True the diagram
    will generate the plot for a subset of the dataframe according the the subset_col subset_val combination.

    Parameters:
    -----------
        df : pandas.DataFrame
            DataFrame containing the plotted data in columns
        column_x : string
            String containing the name of the column that should be plotted on the x-axis
        column_y : string
            String containing the name of the column that should be plotted on the y-axis
        subset_col: string
            String containing the name of the column that is used for subsetting (None if no subset)
        subset_val: string, int
            Value of the column that defines the subset (can be string or integer depending on the column type)
        subset: bolean
            Bolean value defining if subsetting should be used or not (default True)

    '''
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(15,10))

    for sub_df in df:
        sub_df[1].plot(x=column_x, y=column_y, ax=ax, label=sub_df[0], color=color_dict[sub_df[0]])
        #sub_df[1].plot(x=column_x, y="Project Cost (€/kW rated)", ax=ax, label=sub_df[0], color=color_dict[sub_df[0]])
        # if sub_df[0] == "jensen":
        #     interp_yval = np.interp(100000, sub_df[1][column_x],sub_df[1][column_y])
        #     print(interp_yval)
        #     plt.axvline(x = 100000, color = 'grey', linestyle='dashed')
        #     plt.axhline(y = interp_yval, color = 'grey', linestyle='dashed')
        # if sub_df[0] == "noext":
        #     interp_yval = np.interp(100000, sub_df[1][column_x],sub_df[1][column_y])
        #     print(interp_yval)
        #     plt.axvline(x = 100000, color = 'grey', linestyle='dashed')
        #     plt.axhline(y = interp_yval, color = 'grey', linestyle='dashed')

    # Setting axis and plot labels:
    ax.set_title(title, fontsize=25, pad=15, weight='bold')
    ax.set_xlabel(xlab, fontsize=15, labelpad=15)
    ax.set_ylabel(ylab, fontsize=15, labelpad=10)

    # Other options:
    fig.tight_layout()
    ax.legend(loc='lower right')
    ax.margins(x=0)
    plt.xlim(right=300000)
    plt.ylim(top=750, bottom=0)

    plt.savefig(f"{out_path}/{out_name}.png", dpi=300, bbox_inches = 'tight')

# in_path_opt = r"D:\Master_Thesis\Database\Final Outputs\Optimization Results_res_500"

# opt_df = pd.read_excel(f"{in_path_opt}/optimization_results_res_500.xlsx")
# opt_df['Total Cost'] = opt_df['Total Cost'] / 1e9

# color_dict = {'dk': 'blue', 'jensen': 'red', 'noext': 'black'}
# #opt_subset = opt_df[(opt_df['Power Calculation'] == 'cf') & (opt_df['Externality Assumption'] == 'dk')]
# #opt_subset = opt_df[(opt_df['Power Calculation'] == 'cf') & (opt_df['Expansion Goal (rated)'] < 300000)]
# opt_subset = opt_df[(opt_df['Power Calculation'] == 'wpf')]
# opt_sub_group = opt_subset.groupby('Externality Assumption')


# plot_supply_curve(opt_sub_group, column_x='Expansion Goal (rated)', column_y='Total Cost', 
#                   title='Wind power supply curves for Germany under different externality scenarios (wpf calculation)',
#                   xlab='Expansion Goal (in MW rated)', ylab='Expansion Cost (in Billion €)', color_dict=color_dict,
#                   out_path=param["plot_path"], out_name="wp_supply_curve_wpf_res_500")