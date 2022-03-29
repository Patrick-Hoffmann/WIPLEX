import geopandas as gpd
import numpy as np
import pandas as pd 
import pickle
import glob
import os
import pulp as plp #package used to solve Mixed Integer Programming Problem # https://pyomo-simplemodel.readthedoc
import time
from pathlib import Path
import config
import itertools
from multiprocessing import Pool

def power_scaling(wpc_in, flh_const=4000):
    '''
    Function is used to scale the wind power capacity array to a value 
    in line with a maximum FLH constraint of flh_const

    Parameters
    ----------
    wpc_in : np.array
        Numpy array containing the Wind power capacity values per turbine and location.
    flh_const: int
        Integer indicating the Maxiumum FLH constraint used for defining the scaling factor

    Returns
    -------
    wpc_out : np.array
        Array containing the scaled Wind power capacity values.

    '''
    # Scale Full load hours by the maxiumum FLH of the smaller turbine is 4000:
    flh_max = np.max(wpc_in[:, 0]) * 8760/3
    if flh_max > 4000:
        scaling = 4000/flh_max
    else:
        scaling = 1
    
    wpc_out = wpc_in * scaling
    
    return wpc_out

def sound_pressure_level(sound_power, distances, hub_height):
    '''
    Function that calculates the sound pressure level at several distances 
    depending on sound_ower and hub_height
    '''
    return sound_power - abs(10* np.log10(1/(4*np.pi* np.square(distances))))

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


def calc_power_cost(gdf, calc_type, study, ec=True):
    '''
    Function calculates relevant cost and power data for each cell in the dataset

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        DataFrame containing all trubine placement locations and cell specific information
        (such as wind speed, capacity factor, impacted houses at 500m, ...).
    calc_type : string
        String defining the wind power calculation type used. Can be either wpf (based on
        the wind power function), cf (based on rated power and capacity factor) or pc (based
        on the power curve values given by the producer)
    study : string
        String used to define the externality cost assumption used for the cost calculation.
        dk: Droes & Koster (2016) and jensen: Jensen et al. (2014)
    ec : Bolean
        Bolean value which indicates if externalities should be used for the cost calculation
        (default=True).

    Returns
    -------
    total_cost : np.array
        Array containing the total cost (externality + installation cost) of each turbine
        in a specific placement location.
    wpc : np.array
        Array containing the wind power capacity values for each turbine and 
        placement location.
    external_cost : np.array
        Array containing the externality cost of each turbine 
        in a specific placement location.
    install_cost : np.array
        Array containing the installation cost of each turbine
        in a specific placement location.

    '''
    
    ws_100, ws_150 = gdf['ws_100'].to_numpy(), gdf['ws_150'].to_numpy()

    ws = np.array([ws_100, ws_150]).T
    
    if calc_type == "cf":
        cf_1, cf_t2 = gdf['iec1'].to_numpy(), gdf['iec2'].to_numpy()
        cf = np.array([cf_1, cf_t2]).T


    if calc_type == 'wpf':
        ad_100, ad_150 = gdf['ad_100'].to_numpy(), gdf['ad_150'].to_numpy()
        ad = np.array([ad_100, ad_150]).T
    sqm_p, ap_size, no_ap = gdf['sqm_p'].to_numpy(), gdf['ap_size'].to_numpy(), gdf['no_ap'].to_numpy()
    

    imp_list = [f"imp_{i}m" for i in list(range(500, 2750, 250))]

    impacts = gdf.loc[:, imp_list].to_numpy()


    hp = sqm_p * ap_size * no_ap # simple average sqm price 2077 per m^2 (replace for relevant calculation)
    

    impacts_hp = impacts * hp.reshape(-1, 1) # calculates total of house prices for the affected buildings in each zone

    if study == 'jensen':
        dmg = {0: 0.0, 10: 0.0, 20: 0.0307, 30: 0.055, 40: 0.0669, 50: 0.0669, 60: 0.0669}
        
        distance = np.array(range(500, 2750, 250)) - 125 # average distance of a cell within a certain distance range
        hub_height = np.array([[100], [150]])
        sound_power = np.array([[105.5], [106.1]]) #maximum sound power of E-115 and E-126EP3


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
        cost_ext = np.array([list(dk_dmg.values()), list(dk_dmg.values())])


    install_cost = np.array([[(990 + 387 + 56 ) * 3000.0, 
                              (1180 + 387 + 56) * 4200.0]]) # from windguard study

    
    if study == "noext":
        external_cost = np.zeros((len(gdf), 2))
    else:
        external_cost = np.round(impacts_hp @ cost_ext.T, 2)
    
    total_cost = external_cost + install_cost

    # Calculate Wind power capacity:

    blade_radius = np.array([58, 63.5]).T # .reshape(1,2)

    # Calculate power coefficients:

    # Write down efficiency factor cp for the specific turbine at different wind speed (keys)
    enercon_e115_cp = {1: 0.0, 2: 0.058, 3: 0.227 , 4: 0.376, 5: 0.421, 6: 0.451 , 7: 0.469, 8: 0.470, 9: 0.445, 10: 0.401, 11: 0.338, 12: 0.270, 13: 0.212, 14: 0.170 , 15: 0.138,
                        16: 0.114, 17: 0.095, 18: 0.080, 19: 0.068, 20: 0.058, 21: 0.050, 22: 0.044, 23: 0.038, 24: 0.034, 25: 0.030}

    # nominal power 3000kW

    enercon_e126_cp = {1: 0.00, 2: 0.00, 3: 0.28, 4: 0.37, 5: 0.41 , 6: 0.44, 7: 0.45, 8: 0.45, 9: 0.43, 10: 0.40, 11: 0.35, 12: 0.30, 13: 0.24, 14: 0.20, 15: 0.16,
                     16: 0.13, 17: 0.11, 18: 0.09, 19: 0.08, 20: 0.07, 21: 0.06, 22: 0.05, 23: 0.04, 24: 0.04, 25: 0.03}

    # nominal power 4200kW

    enercon_e115_power = {1: 0.0, 2: 3.0, 3: 48.5 , 4: 155.0, 5: 339.0 , 6: 627.5 , 7: 1035.5, 8: 1549.0, 9: 2090.0, 10: 2580.0, 
                       11: 2900.0, 12: 3000.0, 13: 3000.0, 14: 3000.0, 15: 3000.0, 16: 3000.0, 17: 3000.0, 18: 3000.0, 19: 3000.0,
                        20: 3000.0, 21: 3000.0, 22: 3000.0, 23: 3000.0, 24: 3000.0, 25: 3000.0} 

    enercon_e126_power= {1: 0.0, 2: 0.0, 3: 58.0, 4: 185.0, 5: 400.0, 6: 745.0, 7: 1200.0, 8: 1790.0, 9: 2450.0 , 10: 3120.0, 
                         11: 3660.0, 12: 4000.0, 13: 4150.0 , 14: 4200.0 , 15: 4200.0 , 16: 4200.0, 17: 4200.0, 18: 4200.0, 
                         19: 4200.0, 20: 4200.0 , 21: 4200.0, 22: 4200.0, 23: 4200.0, 24: 4200.0 , 25:  4200.0}


    ##########################################################################################################################

    cp_100 = np.round(ws_100.copy())

    for wind_speed in list(enercon_e115_cp.keys()):
        cp_100[cp_100 == wind_speed] = enercon_e115_cp[wind_speed]

    cp_150 = np.round(ws_150.copy())

    for wind_speed in list(enercon_e126_cp.keys()):
        cp_150[cp_150 == wind_speed] = enercon_e126_cp[wind_speed]

    cp = np.vstack((cp_100, cp_150)).T

    ###########################################################################################################################
    wp_100 = np.round(ws_100.copy())

    for wind_speed in list(enercon_e115_power.keys()):
        wp_100[wp_100 == wind_speed] = enercon_e115_power[wind_speed]

    wp_150 = np.round(ws_150.copy())

    for wind_speed in list(enercon_e126_power.keys()):
        wp_150[wp_150 == wind_speed] = enercon_e126_power[wind_speed]

    wp = np.vstack((wp_100, wp_150)).T


    ###########################################################################################################################
    
    if calc_type == 'pc':
        # Caclulate wpc based on enercon power curve info
        wpc = wp * 0.001

    elif calc_type == 'wpf':
        # Calculate wpc based on wind speed and air density formula:
        wpc = (np.pi/2) * ws**3 * blade_radius**2 * ad * cp * 1e-6  # the 1e-6 are needed to convert from W to MW
    
    elif calc_type == "cf":
        rated = np.tile(np.array([3.0, 4.2]), (len(gdf), 1))     
        wpc = rated * cf
        
    else:
        print('Add valid calculation type')
    
        
    return total_cost, wpc , external_cost, install_cost


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
        Value representing the expansion goal.

    Returns
    -------
    turbines_opt : np.arrays
        Array containing Bolean values (0,1) for each turbine and location. It indicates
        which turbine is built in which location (array element=1)

    '''
    # Create model
    m = plp.LpProblem("TC_Germany", plp.LpMinimize)


    # Choice Variables: 
    turbines_100 = plp.LpVariable.dicts('t1', range(len(total_cost)) , lowBound=0, upBound=1, cat=plp.LpBinary) #plp.LpInteger
    turbines_150 = plp.LpVariable.dicts('t2', range(len(total_cost)) , lowBound=0, upBound=1, cat=plp.LpBinary)

    turbines = np.array([list(turbines_100.values()), list(turbines_150.values())]).T

    # # Objective
    #print('Setting up Objective Function...')
    m += plp.lpSum(total_cost * turbines)

    # Constraint
    #print('Adding constraints...')
    m += plp.lpSum(wpc * turbines) >= expansion_goal, "Expansion Constraint"

    for i in range(len(turbines)):
        m += plp.lpSum(turbines[i,:]) <= 1, f"Site constraint {i}"


    # Define the solver:
        
    solver = plp.GUROBI_CMD(msg=0, options=[("MIPGap", 1e-3)])
    #solver = plp.PULP_CBC_CMD(msg=0)
    # solver = plp.GLPK_CMD(path='C:\\Program Files\\glpk-4.65\\w64\\glpsol.exe', msg=0, options = ["--mipgap", "0.01"])  #,"--tmlim", "2000"
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

    t1 = [t.varValue for t in list(turbines_100.values())]
    t2 = [t.varValue for t in list(turbines_150.values())]

    turbines_opt = np.array([t1, t2]).T
    
    return turbines_opt

def optimization_loop(ext_type, power_calc, gdf, ec, overwrite, expansion_goals, out_dir):
    '''
    Main optimization loop to evaluate different expansion scenarios.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        DESCRIPTION.
    ext_type : string
        String Indicating the externality assumption used (can be either dk, jensen or noext at the moment).
    power_calc : string
        String indicating the power calculation used (can be either wpf (wind power function), pc (power curve) or cf (capacity factor) based).
    ec : Bolean
        Bolean indicating whether externality cost should be used.
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
    total_cost, wpc, ext_cost, proj_cost = calc_power_cost(gdf, power_calc, study=ext_type, ec=ec)
    
    opt_file_name = f"results_{power_calc}_{ext_type}_stm_scaled.pickle"
    
#    if power_calc != "cf":
    wpc = power_scaling(wpc)
    
    if os.path.isfile(f'{out_dir}/{opt_file_name}') and not overwrite:
        with open(f'{out_dir}/{opt_file_name}', "rb") as input_file:
            supply_fct = pickle.load(input_file)
    else:
        supply_fct = {}
    
    print(f'Started Optimization with externality assumption: "{ext_type}" and power calculation method: "{power_calc}"')
    for i in expansion_goals:
        print(f'Started optimization for expansion goal {i}')
        if i in supply_fct.keys():
            print(f"Expansion goal {i}m already calculated! Proceeding... ")
        else:
            start = time.time()
            supply_fct[i] = minimization_problem(total_cost, wpc, i)
            print('Finished optimization!...solving time:', time.time() - start)
    
            # Pickle results:
            supply_fct = dict(sorted(supply_fct.items(), key=lambda item: item[0])) # sort keys
            print(f'Saving to file: {out_dir}/{opt_file_name}')
            with open(f'{out_dir}/{opt_file_name}', 'wb') as handle:
                pickle.dump(supply_fct, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
def calc_cost(turbines, total_cost, external_cost, project_cost):
    '''
    Function calculating total, external project and marginal cost 
    as well as the externality cost share for an array of selected turbines.
    '''
    marginal_cost = np.max(np.sum(total_cost * turbines, axis=1))
    p_cost = np.sum(project_cost * turbines) 
    ex_cost = np.sum(external_cost * turbines) 
    cost_total = p_cost + ex_cost
    ext_share = ex_cost/cost_total

    
    return [cost_total, ex_cost, p_cost, marginal_cost, ext_share]

def calc_other(turbines, expansion_goal, rated_power):
    '''
    Function calculates the agrregate rated power, average FLH and the  number of turbines
    build for each type for a given array of selected trubines
    '''
    turbine_count = np.sum(turbines, axis=0)
    rated_power_agg = np.sum(turbine_count * np.array([3.0, 4.2]))
    flh = expansion_goal/rated_power_agg *8760

    return [rated_power_agg, flh] + list(turbine_count)


def generate_results_df(geo_df, model, power_calc, stdy, ext=True):
    '''
    Function calculates the cost predictions for each expansion scenario of a given model using the optimization results.

    Parameters
    ----------
        geo_df : geopandas.GeoDataFrame
            Input DataFrame containing the cell data from the Global Wind Atlas for all valid placement cells
        model : dict
            Dictionary containing arrays with the optimization results (which cells are chosen and which turbine is built in a chosen cell)
        power_calc : str
            Name of the power calculation type used in the optimization of the model results
        stdy: str
            String refering to the externality cost assumption ("dk": Droes & Koster (2016), "jensen": Jensen et al (2014))
        ext: Bolean (default: True)
            Bolean indicating if externality cost should be considered or not. 

    Returns
    -------
        results_df : pandas.DataFrame
            Pandas DataFrame containing the cost predictions (total cost, externality cost, project cost and marginal cost)
            for each expansion scenario of the selected model

    '''
    total_cost, wpc, ext_cost, proj_cost = calc_power_cost(geo_df, power_calc, study=stdy, ec=ext)
            
    results = []
    for i in list(model.keys()):
        results.append([i] + calc_other(model[i], i, [3.0, 4.2]) + calc_cost(model[i], total_cost, ext_cost, proj_cost))
    results_df = pd.DataFrame(results, columns=['Expansion Goal', 'Expansion Goal (rated)', 'FLH (scaled)', 'Type1', 'Type2', 
                                                'Total Cost', 'Externality Cost', 'Project Cost', 'Marginal Cost', 'Ext. Share'])

    return results_df


def generate_optimization_output(gdf_in, opt_file_dir, out_file):
    '''
    Function calculates cost predictions for each file in a given opt_file_dir and generates a .xlsx table containing all cost predictions
    in this directory.

    Parameters
    ----------
        gdf_in: geopandas.GeoDataFrame
            DataFrame used for the optimization process
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
        power_calc = filename_split[1]
        ext_type = filename_split[2]

        if ext_type == "noext":
            externality = False
        else:
            externality = True

        # Load in optimized turbine location files:
        opt_result = pickle.load(open(file, "rb"))
        opt_result= dict(sorted(opt_result.items(), key=lambda item: item[0]))


        opt_df = generate_results_df(gdf_in, opt_result, power_calc, stdy=ext_type, ext=externality)
        opt_df['Power Calculation'] = power_calc
        opt_df['Externality Assumption'] = ext_type
        opt_df['Externality Cost (€/kW rated)'] = opt_df['Externality Cost'] / opt_df['Expansion Goal (rated)'] *1e-3
        opt_df['Project Cost (€/kW rated)'] = opt_df['Project Cost'] / opt_df['Expansion Goal (rated)'] *1e-3
        df_list.append(opt_df)

    df_out = pd.concat(df_list)
    df_out.to_excel(f'{opt_file_dir}/{out_file}.xlsx', index=False)




##############################################################################################################
##############################################################################################################
from itertools import repeat
import signal

def initializer():
    '''
    Used to get catch KeyboardInterrupts for multiprocessing
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == "__main__":
    param = config.optimization_config()
    gdf_in = gpd.read_file(param["input_file"])
    
    gdf_in.rename(columns = {'sqm_euro_v':'sqm_p', 'avg_apartm':'ap_size','no_appartm':'no_ap'}, inplace = True)
    
    # Run the optimization loop for a set of expansion scenarios:
    arg_list = list(zip(*itertools.product(param["externality_types"], param["power_calcs"])))
    
    if param["multiprocessing"]:
        try:
            p = Pool(param["num_workers"], initializer)
            
            p.starmap(optimization_loop, zip(arg_list[0], arg_list[1], repeat(gdf_in), repeat(param["externality_cost"]), 
                              repeat(param["overwrite"]), repeat(param["expansion_scenarios"]), repeat(param["output_folder"])))

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt! Exiting process...")
            p.terminate()
            p.join()
            
        else:
            p.close()
            p.join()
    else:
        for ex_type in param["externality_types"]:
            for power_calc in param["power_calcs"]:
                optimization_loop(ext_type=ex_type, power_calc=power_calc, gdf=gdf_in, ec=param["externality_cost"], 
                                  overwrite=param["overwrite"], expansion_goals=param["expansion_scenarios"], out_dir=param["output_folder"])

    
    # Generate a summary .xlsx file containing all Optimization results:
    
    generate_optimization_output(gdf_in, param["output_folder"], param['output_name'])

    print('Finished script')