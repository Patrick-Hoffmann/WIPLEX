# WIPLEX

The **WI**nd turbine **PL**acement and **EX**ternality assessment model (WIPLEX) was developed to assess optimal wind turbine locations for Germany considering negative turbine externalities. In its current iteration the model optimizes turbine placement by minimizing overall costs (project and externality cost) under an expansion constrained (new MW installed). For more refer to the WIPLEX working paper (LINK)



## Data Sources

#### Wind Data


#### Buildings Data
Buildings data was obtained from OpenStreetMap via geofabrik (LINK). By default the download of the files is integrated in the map generation process (working version February 2023).
Should the download not work, download the zip archives manually and place them in the osm folder in Database/Input Files. The extraction of the zip archives is not required.


#### House Price Data


#### Other Data Sources

## User Guide

### Installation

### Running the model


### Outputs

## Citation


### Regional Coverage
For the moment the model is calibrated and used to calculate wind turbine placements for Germany. 
An extension of the model to other countries, is possible but requires a dataset on location specific house prices. For minimal required changes, the dataset should be a shapefile with a "hp" column with the average regional house price and a geometry column for the region coordinates. The file path should be specified accordingly in the config.py file.
To test a general house price (one price for the whole country/region) comment out the house price addition function in the map generation part and adjust the aggregate_property_values() function of the tm_optimization.py script.