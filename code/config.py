import os
from pathlib import Path


class WIPLEX_settings:
    """
    This class initializes the "param" and "paths" dictionaries containing information on directory paths and parameters
    for the map_generation.py and tm_optimization.py script.
    """
    def __init__(self):
        self.param = {} # initializes Parameter dictionary 
        self.paths = {} # initializes Paths dictionary
        self.sep = os.path.sep # path separator (enables functionality on different systems)

    def initialize_folder_structure(self):
        """
        Intitalizes required folder structure in a given root directory.
        """

        # Get folder of the config file
        current_folder = os.path.dirname(os.path.abspath(__file__))

        # Create root directory (Default:= Folder Created in the parent directory of the config file)
        self.paths["root_dir"] = str(Path(current_folder).parent)

        # Create an assumptions folder for turbine and cost assumptions:
        self.paths["assumptions"] = self.paths["root_dir"] + self.sep + "assumptions"

        # Create a Database input folder
        self.paths["database"] = self.paths["root_dir"] + self.sep + "Database"

        #---------------------- Input Folders -------------------------
        # Create folders for input files:
        self.paths["input_files"] = self.paths["database"] + self.sep + "Input Files"

        # Folders for Required data:
        self.paths["osm_path"] = self.paths["input_files"] + self.sep + "osm"
        self.paths["gwa_path"] = self.paths["input_files"] + self.sep + "gwa"
        self.paths["house_prices"] = self.paths["input_files"] + self.sep + "house_prices"
        self.paths["region_boundaries"] = self.paths["input_files"] + self.sep + "region_boundaries"
        self.paths["turbine_types"] = self.paths["input_files"] + self.sep + "turbine_types"

        # Folders of optional data (comment out if not available)
        self.paths["altitude_path"] = self.paths["input_files"] + self.sep + "altitude_data"
        self.paths["existing_turbines"] = self.paths["input_files"] + self.sep + "existing_turbines"

        #------------------- Intermediate Output Folders ---------------------
        # Create folder for intermediate outputs (generated by the map_generation.py script)
        self.paths["inter_out"] = self.paths["database"] + self.sep + "Intermediate Outputs"
        self.paths["osm_out_path"] = self.paths['inter_out'] + self.sep + "osm"
        self.paths["gwa_out_path"] = self.paths['inter_out'] + self.sep + "gwa"

        #----------------------- Final Results Folders ------------------------
        self.paths["final_results"] = self.paths["database"] + self.sep + "Final Outputs"
        self.paths["output_folder"] = self.paths["final_results"] + self.sep + "Optimization_Results"
        self.paths["plot_path"] = self.paths["final_results"] + self.sep + "Plots"


    def create_folder_structure(self):
        """
        Function calls self.initialize_folder_structure() and creates the folders if they do not exist
        """
        self.initialize_folder_structure()
        for path in self.paths:
            Path(self.paths[path]).mkdir(parents=True, exist_ok=True)

    def set_file_paths(self):
        """
        Set paths to specific input files. Should be run after create_folder_structure().
        """
        self.initialize_folder_structure()

        #---------------------- Input Files -------------------------
        # Generated by script:
        # Path to the resampled (500x500m) multiband GWA raster file
        self.paths["gwa_in_file"] = self.paths["gwa_path"] + self.sep + "gwa_total_DEU.tif"


        # Supplied by User:
        self.paths["region_file"] = self.paths["region_boundaries"] + self.sep + "gadm36_DEU_0.shp"
        self.paths["house_price_file"] = self.paths["house_prices"] + self.sep + "house_prices_germany_coords.shp"
        self.paths["existing_turbines_file"] = self.paths["existing_turbines"] + self.sep + "existing_turbines.shp"
        self.paths["turbine_types_file"] = self.paths["turbine_types"] + self.sep + "turbine_types.xlsx"
        self.paths["altitude_file"] = self.paths["altitude_path"] + self.sep + "dgm200_utm32s.asc"
        self.paths["osm_file_names"] = {"buildings": "gis_osm_buildings_a_free_1.shp",
                  "waterways": "gis_osm_waterways_free_1.shp", 
                  "railways": "gis_osm_railways_free_1.shp", 
                  "water": "gis_osm_water_a_free_1.shp",
                  "roads": "gis_osm_roads_free_1.shp",
                  "landuse": "gis_osm_landuse_a_free_1.shp"
                  } # OSM filenames correspond to shapefiles in the ziped Geofabrik download (added here in case of naming changes)

        # Create dictionary with additional (other than OSM) no-placement areas. Additional non-placement area file paths should be 
        # added here.
        self.paths["other_nopa"] = {"region_border": self.paths["region_file"], 
                                    "ex_turbines": self.paths["existing_turbines_file"]
                                    }
        #---------------------- Assumption Files -----------------------

        # Path to file with turbine type and cost assumptions
        self.paths["assumptions_file"] = self.paths["assumptions"] + self.sep + "assumptions.xlsx"

        #---------------------- Intermediate Output Files -----------------------
        # Path to the Output file of the masking process (no extension since the code produces both 
        #a .tif and a .csv file for further calculations)
        self.paths["masked_file_name"] = "gwa_masked_500_new"

        # Path to the file with final placement locations (after masking high altitude areas and areas with very high or low 
        # wind speed)
        self.paths["gwa_placement_area"] = self.paths["gwa_out_path"] + self.sep + "placement_locations_500_new.shp"

        # Path to the optimization file (includes house price information and number of affected houses per distance range)
        self.paths["optimization_file"] = self.paths["inter_out"] + self.sep + "opt_DEU_500_new.shp"

        #------------------------ Final Output Files ----------------------------
        # Path to the File with the final optimization results (Scenario and .pickle ending will be added on file save) 
        self.paths['results_file'] = self.paths["output_folder"] + self.sep + "optimization_results_DEU_500_new"



    def map_generation_settings(self):
        """
        Function defines the parameters needed in the map_generation.py script.
        """

        #------------------- GWA options ---------------------
        # Define the Global Wind Atlas layers required for download (Note: Downloading only subset of layers can cause problems
        # in the optimization. The parameter should not be changed unless the user also adjusts the necessary calculations in
        # the tm optimization file)

        self.param['gwa_layers'] = ['air-density', 'wind-speed', "combined-Weibull-A", "combined-Weibull-k"] #'capacity-factor_IEC1', 'capacity-factor_IEC2', 'capacity-factor_IEC3',
        # Select for which heights wind layers should be obtained
        self.param['gwa_heights'] = [50, 100, 150, 200]

        #------------------- OSM options ---------------------

        # Define road types considered for the analysis. Names in the lists should coincide with the OSM naming tags (see OSM website)
        # Different types are used to to allow for different minimum distance requirements (see param["nopa_distance"])
        self.param["road_types"] = {"osm_major_roads" : ["motorway", "trunk", "primary", "secondary", "tertiary", 
                                                    "motorway_link", "trunk_link", "primary_link",
                                                    "secondary_link"],
                               "osm_minor_roads" : ["unclassified", "residential", "living_street", "pedestrian"],
                               "osm_small_roads" : ["service", "track",  "track_grade1", "track_grade2"]
                               }

        #------------------- CALC options --------------------
        
        # Define which osm areas should be used for restricting the placement raster 
        # (Note: Landuse is comprised of sub-areas! To omit specific sub areas the user would have to adjust the 
        # generate_placement_area() function in the map_generation.py file, by commenting out the respective lines and 
        # by adjusting the dataframe concatenation accordingly)
        self.param["osm_keys"] = ["roads", "railways", "water", "waterways"] #"landuse", "buildings", "roads", "railways", "water", "waterways"

        # Define which other areas should be used for restricting the placement raster (Name should coincide with the
        # respective keys in self.paths["other nopa"])
        self.param["other_nopa_keys"] = ["region_border", "ex_turbines"] #, "ex_turbines"

        # Set minimum distance requirements (in meter) for certain areas of interest:
        # 1. : For Roads the user can set different distance requirements for major roads (first element), minor roads
        #      and small roads as defined in param["road_types"]
        # 2. : The distance for existing turbines (ex_turbines) can either be in meter or "5D_Abstand" which is equivalent.
        #      to a minimum distance of 5*rotor_diameter
        # 3. : region_border sets the minimum distance to the border of the analysed region (eg. Germany)

        self.param["nopa_distance"] = {"buildings": 250, "landuse": {"protected_areas": 200, "residential": 250}, 
                                  "waterways": 65, "railways": 250, "water": 65,
                                  "roads": [100, 80, 1], "ex_turbines": "5D_Abstand", "region_border": 0
                                  }

        # Define restriction parameters for maximum altitude and cut-in + cut-out wind speeds:
        self.param["cut_in"] = 2.5
        self.param["cut_out"] = 34
        self.param["max_alt"] = 1522 # threshold obtained from maximum altitude in MaStR dataset

        # Select EPSG coordinate system for distance calculations (in meter and appropriate for the Region of interest)
        self.param["epsg_distance_calc"] = 25832

        # Select EPSG coordinate system for general file storage and display (leave as EPSG:4326 if possible)
        self.param["epsg_general"] = 4326

        # Choose list of buffer regions. This parameter defines what impact zones (in meter) should be calculated
        # for each of the turbine locations:
        self.param["impact_buffers"] = list(range(500, 5750, 250)) #[500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]


        # The parameters inversion, cropping and intersection_mask refer to options in the rasterio.mask module
        # (https://rasterio.readthedocs.io/en/stable/api/rasterio.mask.html)

        # inversion := Select if pixels inside or outside of a shape should be masked. Eg: All cells inside the non-placement
        # area for buildings should be omitted (hence True). However, all cells outside the border of our defined region should
        # be dropped as well (hence Falls) 

        self.param["inversion"] = {"buildings": True, "landuse": True, "waterways": True, "railways": True, "water": True,
                              "roads": True, "ex_turbines": True, "region_border": False}

        # cropping := Crop the raster to the newly masked shape. This makes only sense for the area outside of our region,
        # to omit irrelevant cells.
        self.param["cropping"] = {"buildings": False, "landuse": False, "waterways": False, "railways": False, "water": False,
                                  "roads": False, "ex_turbines": False, "region_border": True}

        # intersection_mask (called all_touched in rasterio) := Select wether pixel (raster cell) should be masked if the cell
        # border intersects with the mask region (intersection_mask = True) or only if the cell center is in the mask region
        # (intersection_mask = False). Default is False (less restrictive) since turbines are placed in the cell center.
        self.param["intersection_mask"] = False



    def optimization_settings(self):
        """
        Function adds optimization parameters to the parameter dictionary (param).
        """
        #------------------- Main Optimization parameters --------------------

        # Main Optimization parameters:
        # Define a list of expansion scenarios of interest (in MW)
        self.param["expansion_scenarios"] = list(range(100, 300, 50)) # list(5000, 10000)

        # State which externality damage calculations should be included 
        # (Existing: "jensen": Jensen et. al. 2014; "dk": Droes & Koster 2016; "noext": No externality case)
        #self.param["externality_types"] = ["jensen", "noext"] #"dk"

        # State which calculation of the the location wind power output should be used. Refer for user Manual and optimization
        # file for more information (Implemented: "wpf": Wind power function; "cf": Calculation based on capacity factor; 
        # "pc": Calculation based on power curve of producer)
        self.param["power_calcs"] = ["wpf"] #"cf", "pc"

        # Define the cost Scenario to be used (Scenarios are set in the assumptions.xlsx file). param["Scenario"] should be set to the sheet
        # name of the required scenario in the assumptions file
        self.param["Scenario"] = "Sze1"

        # Set the impact range to consider (impacts can be generated up to the impact buffers generated see: self.param["impact_buffers"]
        # and the impacts costs defined in self.paths["assumptions_file"]). This option can be used to set a lower impact range for the
        # optimization procedure (if not required set it to the same range as self.param["impact_buffers"])
        self.param["impact_range"] = list(range(500, 5750, 250))

        # 
        self.param["externality_cost"] = True

        # Decide weather existing results file should be overwritten (True) or new scenarios simply added to the results (False)
        self.param["overwrite"] = True
        
        #--------------------- Multiprocessing parameters ---------------------

        # Select if multiprocessing should be used:
        self.param["multiprocessing"] = False
        # Decide upon the number of processes to be run in parallel
        self.param["num_workers"] = 3


    def initialize_config(self):
        """
        Function calls other class functions to initialize the paths and param dictionaries
        """
        self.create_folder_structure() # comment out when existing
        self.set_file_paths()
        self.map_generation_settings()
        self.optimization_settings()



# test = WIPLEX_settings()
# test.initialize_config()
# print(test.param["num_workers"])
# print(test.param["inversion"])