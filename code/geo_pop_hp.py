import pandas as pd
import geopandas as gpd 
from shapely.ops import unary_union

pop_df = pd.read_excel("04-kreise.xlsx", sheet_name="Kreisfreie Städte u. Landkreise", skiprows=4, usecols =["Unnamed: 0", "insgesamt", "je km2"]).dropna(how="any").drop(0)
pop_df = pop_df.rename(columns={"insgesamt": "pop_count", "je km2": "pop_d_km2"})

region_border = gpd.read_file("../regions/gadm36_DEU_2.shp", encoding="utf-8") 
region_border["CC_2"] = region_border["CC_2"].replace("03152", "03159") # correct classifier fpr Göttingen

region_border.loc[region_border["NAME_2"] == "Göttingen", "geometry"]= gpd.GeoSeries([region_border[region_border["NAME_2"]=="Osterode am Harz"].iloc[0]["geometry"], 
    region_border[region_border["NAME_2"]=="Göttingen"].iloc[0]["geometry"]]).unary_union

region_pop = pd.merge(region_border, pop_df, left_on="CC_2", right_on="Unnamed: 0").drop("Unnamed: 0", axis=1)
region_pop.to_file("county_population.shp", index=False)


# # Add population to house price gdf:
# test = pd.read_excel("D:/Master_Thesis/Database/Input Files/house_prices/house_prices_germany.xlsx")
# test["county"] = test["county"].replace({" \(LK\)": "", "\(Stadt\)": "(Kreisfreie Stadt)"}, regex=True)

# gdf_pop = gpd.read_file("D:/Master_Thesis/Database/Input Files/population/county_population.shp", 
#     crs=param["epsg_general"])[["NAME_2", "CC_2", "pop_count", "pop_d_km2", "geometry"]]

# gdf_house_pop = pd.merge(gdf_pop, test, left_on="NAME_2", right_on="county").drop("county", axis=1)
# gdf_house_pop.to_file(param["house_prices"], index=False)