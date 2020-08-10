"""
Este script es para actualizar los datos que tenemos guardados.
"""
import logging
from multiprocessing import Pool
from functools import partial

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash import callback_context

import constants as c
from app import app
import update_db.utils as db_u

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

# Creación del HTML para esta vista
layout_update = html.Div([
    html.Br(),
    html.Div([
        html.Button('Update daily DB', id='btn-1', style={'color': c.PALETTE['green']}),
        html.Button('Update technical DB', id='btn-2', style={'color': c.PALETTE['green']}),
        html.Br(),
        html.Div(id='container-button-timestamp')], style={
            'columnCount': 1, 'textAlign': 'center',
            'color': c.PALETTE['grey']}),
    html.Br(),
    html.Div(style={
        'textAlign': 'center'
        },
        children=dcc.Link('Home', href='/', style={
            'color': c.PALETTE['red']
            })),
    html.Div(style={
        'textAlign': 'center'
        },
        children=dcc.Link('Analysis', href='/analysis', style={
            'color': c.PALETTE['red']
            })),
    html.Div(style={
        'textAlign': 'center'
        },
        children=dcc.Link('Strategy One', href='/strategyone', style={
            'color': c.PALETTE['red']
            }))
    ])

@app.callback(Output('container-button-timestamp', 'children'),
              [Input('btn-1', 'n_clicks'),
               Input('btn-2', 'n_clicks')])
def displayClick(btn1, btn2):
    """
    Actualizar los datos bursátiles.

    Parámetros:
    ----------
        btn1:
            Click del botón 1.
        btn2:
            Click del botón 2.

    Return:
    -------
        Mensaje de confirmación de datos actualizados.
    """

    symbols = c.SYMBOLS
    # symbols = db_u.get_symbols()

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    # Actualizar los datos de precios.
    if 'btn-1' in changed_id:
        for symbol in symbols:
            try:
                db_u.get_daily(symbol, is_save=True)
            except Exception as exp:
                LOGGER.warning(f"Ticker {symbol} ha fallado. Fallo: {exp}")
#                db_u.failed_symbols(symbol)
                continue

        msg = f'Daily prices updated!'

    # Actualizar los indicadores técnicos.
    elif 'btn-2' in changed_id:

#        for symbol in symbols:
        with Pool(3) as p:
            gt_partial = partial(db_u.get_technical, is_save=True)
            p.map(gt_partial, symbols)
#            try:
#                db_u.get_technical(symbol, is_save=True)
#            except Exception as exp:
#                LOGGER.warning(f"Ticker {symbol} ha fallado. Fallo: {exp}")
#                continue
        msg = 'Technical indicators updated!'

    else:
        msg = 'None of the buttons have been clicked yet'

    return html.Div(msg)
