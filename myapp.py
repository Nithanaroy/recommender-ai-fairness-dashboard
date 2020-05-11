# # myapp.py
 
# import pandas as pd
# import geopandas as gpd
# import json
# from bokeh.models import GeoJSONDataSource 
# from bokeh.plotting import figure, curdoc
# from bokeh.layouts import column
 
 
# # Read the country borders shapefile into python using Geopandas 
# shapefile = 'data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'
# gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

# # Rename the columns
# gdf.columns = ['country', 'country_code', 'geometry']
 

# # Convert the GeoDataFrame to GeoJSON format so it can be read by Bokeh
# merged_json = json.loads(gdf.to_json())
# json_data = json.dumps(merged_json)
# geosource = GeoJSONDataSource(geojson=json_data)


# # Make the plot
# TOOLTIPS = [
# ('UN country', '@country')
# ]

# p = figure(title='World Map', plot_height=600 , plot_width=950, tooltips=TOOLTIPS,
# x_axis_label='Longitude', y_axis_label='Latitude')

# p.patches('xs','ys', source=geosource, fill_color='white', line_color='black',
# hover_fill_color='lightblue', hover_line_color='black')
 
# # This final command is required to launch the plot in the browser
# curdoc().add_root(column(p))

import warnings, logging

import param
import panel as pn

import holoviews as hv

import bokeh
from bokeh.plotting import show
from bokeh.io import output_notebook

from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from holoviews.streams import RangeXY
from holoviews.operation import decimate
from holoviews import opts, Cycle

hv.extension('bokeh')
renderer = hv.renderer('bokeh')
output_notebook()

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s %(filename)s] %(message)s')


symbol = pn.widgets.RadioButtonGroup(options=['A', 'B'])

@pn.depends(symbol=symbol.param.value)
def load_symbol_cb(symbol):
    if symbol == "A":
        return hv.Points(range(10))
    else:
        return hv.Points(range(10, 0, -1))

dmap = hv.DynamicMap(load_symbol_cb)

dashboard = pn.Row(pn.WidgetBox('## Stock Explorer', symbol), dmap.opts(width=500, framewise=True))


dashboard.servable(title="Hello World Dashboard")




