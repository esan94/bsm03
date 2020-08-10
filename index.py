"""
Este es el archivo que se usa para ejecutar la aplicación.
"""
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import constants as c
from app import app
from home import layout_home
from update import layout_update
from analysis import layout_analysis
from strategy_one import layout_strategyone

# Creación del HTML principal de la app
app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], style={'backgroundColor': c.PALETTE['black']})

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    """
    En esta función se manejan las diferentes rutas del punto
    web.

    Parámetros:
    ----------
        pathname:
            ruta relativa del punto web.

    Return:
    ------
        String: ruta en uso.
    """
    if pathname == '/':
         return layout_home
    elif pathname == '/update':
        return layout_update
    elif pathname == '/analysis':
        return layout_analysis
    elif pathname == '/strategyone':
        return layout_strategyone
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)
