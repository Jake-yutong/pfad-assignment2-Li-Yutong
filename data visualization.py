import geopandas as gpd
import matplotlib.pyplot as plt

# 读取SHP文件
gdf = gpd.read_file('/Users/liyutong/Desktop/Gradedhistoricbuildingswithimageandgrading_SHP/HBG_20250703_141956.shp')

# 可视化
gdf.plot()
plt.show()