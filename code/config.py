# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 14:49:06 2022

@author: hoffmannp
"""
import os
from pathlib import Path

def optimization_config():
    '''
    Function creates a parameter dictionary with all optimization parameters for the wind turbine placement optimization.

    Returns
    -------
    param : dictionary
        Dictionary with all optimization parameters.

    '''
    sep = os.path.sep
    param = {}
    
    # Directory Paths:
    param["root_dir"] = "G:\Projekte\FAUST\Papers\Windpower_PHMM\Thesis_Paper\Database"
    param["input_file"] = param["root_dir"] + sep + "Intermediate Outputs" + sep + "Opt_500_nointer_nores" + sep + "opt_DEU_500_nointer_nores.shp"
    param['output_name'] = "optimization_results_nores_500"
    param["output_folder"] = param["root_dir"] + sep + "Final Outputs" + sep + param['output_name']

    # Main Optimization parameters:
    param["expansion_scenarios"] = list(range(5000, 105000, 5000)) # list(5000, 10000)
    param["externality_types"] = ["jensen", "noext"] #"dk"
    param["power_calcs"] = ["wpf"] #"cf", "pc"
    param["externality_cost"] = True
    param["overwrite"] = True
    
    # Multiprocessing Parameters
    param["multiprocessing"] = True
    param["num_workers"] = 9
    
    # Generate Output Path if it does not exist yet:
    Path(param["output_folder"]).mkdir(parents=True, exist_ok=True)
    
    return param
