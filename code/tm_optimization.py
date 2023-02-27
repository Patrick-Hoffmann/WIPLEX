# Package Imports:

import config # Load WIPLEX configuration
import geopandas as gpd
import numpy as np
import pandas as pd 
import pickle
import glob
import os
import pulp as plp #package used to solve Mixed Integer Programming Problem # https://pyomo-simplemodel.readthedoc
import time
from pathlib import Path
import itertools
from multiprocessing import Pool, freeze_support, Manager
from itertools import repeat
import ast
import signal
import math



#############################################################################################################
################################################ Function Hub ###############################################


# ----------------------------------------- Additional helper Functions ------------------------------------------ #

def up_down_round(x, base=50, up=True):
    if up:
        return base * math.ceil(x/base)
    else:
        return base * math.floor(x/base)



def interpolate_heights(wpc_array, t_height, gwa_heights = [50, 100, 150, 200]):
    lower_layer = up_down_round(t_height, up=False)
    upper_layer = up_down_round(t_height, up=True)
    lower_index = gwa_heights.index(lower_layer)
    upper_index = gwa_heights.index(upper_layer)

    return wpc_array[:,lower_index] + (wpc_array[:,upper_index] - wpc_array[:,lower_index]) * (t_height - lower_layer)/50

# ----------------------------------------- Wind power functions ------------------------------------------ #

def weibull_wind_pdf(x, k, A):
    return k/A * (x/A)**(k-1) * np.exp(-(x/A)**k)


def wind_power_function(ws, radius, ad, cp):
    # Calculating costs using wind power function
    return (np.pi/2) * ws**3 * radius**2 * ad * cp * 1e-6


def weibull_power_output(gdf, gwa_heights, cp_df, types_df):
    x = np.arange(3, 26, 1)
    wb_A = gdf[[f"wb_A_{x}" for x in gwa_heights]].to_numpy()
    wb_k = gdf[[f"wb_k_{x}" for x in gwa_heights]].to_numpy()
    ad_array = gdf[[f"ad_{x}" for x in gwa_heights]].to_numpy()
    blade_radius = types_df["Rotor diameter (m)"].to_numpy()/2
    
    array_list = []
    for ws in x:
        probs = weibull_wind_pdf(ws, wb_k, wb_A)
        cp_array = cp_df[cp_df["Wind Speed"] == ws].loc[:,cp_df.columns != 'Wind Speed'].to_numpy()
        res = wind_power_function(ws, blade_radius, ad_array, cp_array) * probs
        array_list.append(res)
    return np.array(array_list)


# ----------------------------------------- Location Cost Functions ------------------------------------------ #

def get_avg_property_value(gdf, impact_range):
    '''
    Function calulates the aggregate property value of buildings in each impact zone for all cells in the gdf.
    Parameters
    ----------
        gdf :  geopandas.GeoDataFrame
            GeoDataFrame containing house price and property impact information (columns: eg: imp_500m,..., hp )
        impact_range : list
            list of integers specifying which impact ranges to use for the analysis
    Returns:
    --------
        impacts_hp : numpy.array
            Array containing aggregate property values for each cell and impact zone (cell x zone)
    '''
    # Define impact matrix
    imp_list = [f"imp_{i}" for i in impact_range]
    impacts = gdf.loc[:, imp_list].to_numpy()
    
    # Calculate aggregate  property values per impact zone:
    impacts_hp = impacts * gdf["hp"].to_numpy().reshape(-1, 1)
    
    return impacts_hp

def calculate_location_costs(gdf, damage_df, t_types_df, impact_range, externality=True):
    '''
    Function for calculating location specific costs for each of the turbines in the dataset
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the all location information (placement locations, impacts, house prices...)
    damage_df: pandas.DataFrame
        DataFrame with the damage assumptions for each turbine (turbines as collumns and impact zones as rows)
    t_type_df: pandas.DataFrame
        DataFrame with information on each turbine used (name, height, rated power, ...)
    impact_range: list
        List containing all impact zones that should be used (maximum is the max number of impacts calculated)
    externality: Bolean
        Parameter indicating if wind power externalities should be considered (Default=True)

    Returns
    -------
        total_cost, ext_cost, const_cost: np.arrays
            Arrays containing information on total, externality and construction costs for each location and turbine
    '''
    # Definition of property damage matrix:
    damage_selection = damage_df.loc[damage_df["Upper"].isin(impact_range)]
    prop_damages = damage_selection[list(t_types_df["Name"])].to_numpy()

    # Obtain house price values for each location
    prop_values = get_avg_property_value(gdf, impact_range)/1000000
    
    # Caclulation of externalities:
    if not externality:
        ext_cost = np.zeros((len(gdf), len(t_types)))
    else:
        ext_cost = np.round(prop_values @ prop_damages, 2)

    # Calculation of construction costs:
    const_cost = t_types["Total construction cost (million €)"].to_numpy().reshape(1,-1)

    # Calc total costs:
    total_cost = ext_cost + const_cost
    
    return total_cost, ext_cost, const_cost

# ----------------------------------------- Optimization Functions ------------------------------------------ #

def minimization_problem(total_cost, wpc, expansion_goal):
    '''
    Main optimization function of the Wind turbine placement algorithm

    Parameters
    ----------
    total_cost : np.array
        Array containing the total cost (project + externality cost) for each wind turbine
        and placement location.
    wpc : np.array
        Array containing the wind power capacity values for each turbine and location.
    expansion_goal : int
        Value representing the expansion goal (in MW).

    Returns
    -------
    turbines_opt : np.arrays
        Array containing Bolean values (0,1) for each turbine and location. It indicates
        which turbine is built in which location (array element=1)

    '''
    # Create model
    m = plp.LpProblem("TC_Germany", plp.LpMinimize)


    # Choice Variables:
    t_list = []
    for i in range(total_cost.shape[1]):
        turbine = plp.LpVariable.dicts(f't{i}', range(len(total_cost)) , lowBound=0, upBound=1, cat=plp.LpBinary)
        t_list.append(list(turbine.values()))
    
    #turbines = np.array([list(turbines_100.values()), list(turbines_150.values())]).T
    turbines = np.array(t_list).T

    # # Objective
    #print('Setting up Objective Function...')
    m += plp.lpSum(total_cost * turbines)

    # Constraint
    #print('Adding constraints...')
    m += plp.lpSum(wpc * turbines) >= expansion_goal, "Expansion Constraint"

    for i in range(len(turbines)):
        m += plp.lpSum(turbines[i,:]) <= 1, f"Site constraint {i}"


    # Define the solver:
    if "GUROBI_CMD" in plp.listSolvers(onlyAvailable=True):  
        solver = plp.GUROBI_CMD(msg=0, options=[("MIPGap", 1e-3)])
    else:
        #solver = plp.PULP_CBC_CMD(msg=0)
        solver = plp.GLPK_CMD(path='C:\\Program Files\\glpk-4.65\\w64\\glpsol.exe', msg=0, options = ["--mipgap", "0.01"])  #,"--tmlim", "2000"
    # use other solver: plp.PULP_CBC_CMD(msg=0) ; plp.GLPK_CMD(path='C:\\Program Files\\glpk-4.65\\w64\\glpsol.exe', msg=0, options = ["--mipgap", "0.001","--tmlim", "1000"])


    # Optimize
    #print('Started optimization')
    m.solve(solver)


    # Print the status of the solved LP
    print(f"Status = {plp.LpStatus[m.status]}")

    # Print the value of the objective
    print(f"Objective value = {plp.value(m.objective)}")

    # Print the value of the constraint:
    sel = []
    for constraint in m.constraints:
        if constraint == "Expansion_Constraint":
            constraint_sum = 0
            for var, coefficient in m.constraints[constraint].items():
                constraint_sum += var.varValue * coefficient
                sel.append(var.varValue)
            print(m.constraints[constraint].name, constraint_sum)
        else:
            pass
    
    turbine_opt_list = []
    for turbine_type in t_list:
        turbine_results = [t.varValue for t in turbine_type]
        turbine_opt_list.append(turbine_results)
            
    turbines_opt = np.array(turbine_opt_list).T
    
    return turbines_opt


def run_optimization(expansion_goal, tot_cost_array, wpc_array, storage_dict):
    print(f'Started optimization for expansion goal {expansion_goal}')
    if expansion_goal in storage_dict.keys():
        print(f"Expansion goal {expansion_goal}MW already calculated! Proceeding... ")
    else:
        start = time.time()
        storage_dict[expansion_goal] = minimization_problem(tot_cost_array, wpc_array, expansion_goal)
        print(f'Finished optimization for expansion goal {expansion_goal}m!...solving time:', time.time() - start)


def optimization_loop(gdf, damage_df, t_type_df, cp_df, impact_range, scenario, gwa_heights, 
                      overwrite, expansion_goals, out_dir, multiprocessing=False):
    '''
    Main optimization loop to evaluate different expansion scenarios.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the all information (placement locations, impacts, house prices...)
    damage_df: pandas.DataFrame
        DataFrame with the damage assumptions for each turbine (turbines as collumns and impact zones as rows)
    t_type_df: pandas.DataFrame
        DataFrame with information on each turbine used (name, height, rated power, ...)
    impact_range: list
        List containing all impact zones that should be used (maximum is the max number of impacts calculated)
    scenario: string
        Name of the Scenario that should be used. This should be identical to the sheet name of the damage scenario in the assumptions file
    gwa_heights: list
        list of integers specifying which wind layers (height) of the Global wind Atlas should be used
    overwrite : Bolean
        Bolean value that defines whether new expansion scenarios should just be added to an existing file or if old 
        scenarios should be overwritten.
    expansion_goals : list
        List of of integer values defining the expansion scenarios for which to run the optimization.
    out_dir : string
        Directory where the output files should be saved to.

    Returns
    -------
    None.

    '''
    print("Started optimization process...")
  
    # Calculate Wind Power Capacity
    wpc = weibull_power_output(gdf, gwa_heights, cp_df, t_type_df).sum(axis=0)

    # Interpolate power Outputs using turbine height:
    wpc = np.vstack([interpolate_heights(wpc, t_height) for t_height in list(t_type_df["Hub height (m)"])]).T

    # Calculate Location Specific Costs:
    total_lsc, ext_lsc, const_lsc = calculate_location_costs(gdf, damage_df, t_type_df, impact_range, 
                                                             externality=True)

    
    if os.path.isfile(f'{out_dir}_{scenario}.pickle') and not overwrite:
        with open(f'{out_dir}_{scenario}.pickle', "rb") as input_file:
            supply_fct = pickle.load(input_file)
    else:
        supply_fct = {}
    

    if multiprocessing:
        manager = Manager()
        supply_fct = manager.dict(supply_fct)

        p = Pool()                                   # Create a multiprocessing Pool
        p.starmap(run_optimization, zip(expansion_goals, repeat(total_lsc), repeat(wpc), repeat(supply_fct)))
        p.close()
        p.join()

    else:
        for i in expansion_goals:
            run_optimization(i, total_lsc, wpc, supply_fct)

    # Pickle results:
    supply_fct = dict(sorted(supply_fct.items(), key=lambda item: item[0])) # sort keys
    print(f'Saving to file: {out_dir}_{scenario}.pickle')
    with open(f'{out_dir}_{scenario}.pickle', 'wb') as handle:
        pickle.dump(supply_fct, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ------------------------------------------ Output Functions -------------------------------------------- #

def calc_cost(turbines, total_cost, external_cost, const_cost):
    '''
    Function calculating total, external project and marginal cost 
    as well as the externality cost share for an array of selected turbines.
    '''
    marginal_cost = np.max(np.sum(total_cost * turbines, axis=1))
    p_cost = np.sum(const_cost * turbines) 
    ex_cost = np.sum(external_cost * turbines) 
    cost_total = p_cost + ex_cost
    ext_share = ex_cost/cost_total

    
    return [cost_total, ex_cost, p_cost, marginal_cost, ext_share]

def calc_other(turbines, expansion_goal, rated_power_array):
    '''
    Function calculates the aggregate rated power, average FLH and the  number of turbines
    build for each type for a given array of selected trubines
    '''
    turbine_count = np.sum(turbines, axis=0)
    rated_power_agg = np.sum(turbine_count *rated_power_array)
    flh = expansion_goal/rated_power_agg *8760

    return [rated_power_agg, flh] + list(turbine_count)

def generate_results_df(geo_df, model, damage_df, t_type_df, impact_range, ext=True):
    '''
    Function calculates the cost predictions for each expansion scenario of a given model using the optimization results.

    Parameters
    ----------
        geo_df : geopandas.GeoDataFrame
            Input DataFrame containing the cell data from the Global Wind Atlas for all valid placement cells
        model : dict
            Dictionary containing arrays with the optimization results (which cells are chosen and which turbine is built in a chosen cell)
        damage_df: pandas.DataFrame
            DataFrame with the damage assumptions for each turbine (turbines as collumns and impact zones as rows)
        t_type_df: pandas.DataFrame
            DataFrame with information on each turbine used (name, height, rated power, ...)
        impact_range: list
            List containing all impact zones that should be used (maximum is the max number of impacts calculated)
        ext: Bolean (default: True)
            Bolean indicating if externality cost should be considered or not. 

    Returns
    -------
        results_df : pandas.DataFrame
            Pandas DataFrame containing the cost predictions (total cost, externality cost, project cost and marginal cost)
            for each expansion scenario of the selected model

    '''
    # Calculate Location Specific Costs:
    total_lsc, ext_lsc, const_lsc = calculate_location_costs(geo_df, damage_df, t_type_df, impact_range, externality=True)
    
    results = []
    for i in list(model.keys()):
        results.append([i] + calc_other(model[i], i, np.array([t_type_df["Rated power (MW)"]])) + calc_cost(model[i], 
                                                                                                    total_lsc, ext_lsc, const_lsc))
    results_df = pd.DataFrame(results, columns=['Expansion Goal', 'Expansion Goal (rated)', 
                                                'FLH'] +  list(t_types.index) + ['Total Cost', 'Externality Cost', 
                                                'Project Cost', 'Marginal Cost', 'Ext. Share'])

    return results_df

def generate_optimization_output(gdf_in, opt_file_dir, damage_df,t_type_df, impact_range, out_file):
    '''
    Function calculates cost predictions for each file in a given opt_file_dir and generates a .xlsx table containing all cost predictions
    in this directory.

    Parameters
    ----------
        gdf_in: geopandas.GeoDataFrame
            DataFrame used for the optimization process
        damage_df: pandas.DataFrame
            DataFrame with the damage assumptions for each turbine (turbines as collumns and impact zones as rows)
        t_type_df: pandas.DataFrame
            DataFrame with information on each turbine used (name, height, rated power, ...)
        impact_range: list
            List containing all impact zones that should be used (maximum is the max number of impacts calculated)
        opt_file_dir: str
            String containing a path to the direcotory with the optimization results
        out_file: str
            Name of the output .xlsx file
    '''
    opt_files = glob.glob(f'{opt_file_dir}/*.pickle')

    df_list = []
    for file in opt_files:
        # Infer optimization settings from filename:
        filename_split = Path(file).stem.split('_')
        scenario = filename_split[-1]
        
        
        if scenario == "noext":
            externality = False
        else:
            externality = True
            
        # Load in optimized turbine location files:
        opt_result = pickle.load(open(file, "rb"))
        opt_result= dict(sorted(opt_result.items(), key=lambda item: item[0]))


        opt_df = generate_results_df(gdf_in, opt_result, damage_df, t_type_df, impact_range, ext=externality)
        opt_df['Scenario'] = scenario
        opt_df['Externality Cost (€/kW rated)'] = opt_df['Externality Cost'] / opt_df['Expansion Goal (rated)'] *1e-3
        opt_df['Project Cost (€/kW rated)'] = opt_df['Project Cost'] / opt_df['Expansion Goal (rated)'] *1e-3
        df_list.append(opt_df)

    df_out = pd.concat(df_list)
    df_out.to_excel(f'{out_file}.xlsx', index=False)

#############################################################################################################
#############################################################################################################


if __name__ == "__main__":
    freeze_support()

    # -- Initialize Config:
    WIPLEX_config = config.WIPLEX_settings()
    WIPLEX_config.initialize_config()
    param = WIPLEX_config.param
    paths = WIPLEX_config.paths

    # -- Load Datasets:

    # Main geodataframe with Wind data, impacted buildings and house prices:
    gdf_in = gpd.read_file(paths["optimization_file"])

    # Turbine Types:
    t_types = pd.read_excel(paths["assumptions_file"], sheet_name="Turbine_Types")
    t_types = t_types.set_index("Variable").T #.reset_index(drop=True)

    # Power Coefficients
    cps = pd.read_excel(paths["assumptions_file"], sheet_name="Power_Coefficient", skiprows=3)

    # Load in damage assumption:
    damages = pd.read_excel(paths["assumptions_file"], sheet_name=param["Scenario"], skiprows=2)


    gdf_in_sub = gdf_in.copy().iloc[:6000]
    

    optimization_loop(gdf_in_sub, damages, t_types, cps, param["impact_range"], param["Scenario"], param["gwa_heights"], 
                overwrite=param["overwrite"], expansion_goals=param["expansion_scenarios"], out_dir=paths['results_file'], multiprocessing=param["multiprocessing"])

    # Generate a summary .xlsx file containing all Optimization results:
    
    generate_optimization_output(gdf_in_sub, paths["output_folder"], damages, t_types, param["impact_range"], paths['results_file'])


    print('Finished script')