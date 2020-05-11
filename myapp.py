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




