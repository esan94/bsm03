"""
Este script es donde se hace todo el análisis de las curvas.
"""
from datetime import datetime
from dateutil.relativedelta import relativedelta

from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import plotly.graph_objects as go
import numpy as np

import tools.read as t_read
import constants as c
from app import app

# Crear el HTML para esta vista
layout_analysis = html.Div([
    html.Div([
        html.H1(' '),
        html.Br(),
        dcc.Dropdown(
            id='symbols',
            options=[{'label': ticker, 'value': ticker} for ticker in c.SYMBOLS],
            multi=False,
            value='AAPL',
            style={'width': '40%',
                   'margin':'auto',
                   'color': c.PALETTE['grey']}
        ),

        dcc.RadioItems(
            id='time-filter',
            options=[
                {'label': '10Y', 'value': '10Y'},
                {'label': '5Y', 'value': '5Y'},
                {'label': '3Y', 'value': '3Y'},
                {'label': '1Y', 'value': '1Y'},
                {'label': '6M', 'value': '6M'},
                {'label': '3M', 'value': '3M'},
                {'label': '1M', 'value': '1M'}
            ],
            value='1M', style={
                'textAlign': 'center',
                'color': c.PALETTE['grey'],
                'margin':'auto',
                'columnCount': 7
            }
        )], style={
            'display': 'flex',
            'textAlign': 'center',
            'color': c.PALETTE['grey']}, className='row'),

    html.Div(children=[
            dcc.Dropdown(
                id='tech_indc',
                options=[
                    {'label': k, 'value': k} for k in c.TECH.keys()
                ],
                multi=True,
                value='CLOSE',
                style={'width': '45%',
                       'margin':'auto',
                       'margin-right': '50px',
                       'color': c.PALETTE['grey']}
            )]),

    html.Div(style={'backgroundColor': c.PALETTE['black'], 'width': '100%'},
        children=[
            dcc.Graph(style={'display': 'inline-block', 'backgroundColor': c.PALETTE['black'], 'width': '50%'}, id='candle-graphic'),
            dcc.Graph(style={'display': 'inline-block', 'backgroundColor': c.PALETTE['black'], 'width': '50%'}, id='linear-graphic')
            ]),
    html.Div(dcc.RadioItems(
        id='shift-filter',
        options=[
            {'label': '1', 'value': '1'},
            {'label': '3', 'value': '3'},
            {'label': '7', 'value': '7'}
        ],
        value='1', style={
            'textAlign': 'center',
            'color': c.PALETTE['grey'],
            'margin':'auto',
            'columnCount': 9
        }
    )),
    html.Div(style={'backgroundColor': c.PALETTE['black'], 'width': '100%'},
        children=[
            dcc.Graph(style={'display': 'inline-block', 'backgroundColor': c.PALETTE['black'], 'width': '50%'}, id='pct-graphic'),
            dcc.Graph(style={'display': 'inline-block', 'backgroundColor': c.PALETTE['black'], 'width': '50%'}, id='box-graphic')]),
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
        children=dcc.Link('Update', href='/update', style={
            'color': c.PALETTE['red']
            })),
    html.Div(style={
        'textAlign': 'center'
        },
        children=dcc.Link('Strategy One', href='/strategyone', style={
            'color': c.PALETTE['red']
            }))
    ], style={'color': c.PALETTE['green']})

@app.callback(
    Output('box-graphic', 'figure'),
    [Input('symbols', 'value'),
     Input('time-filter', 'value'),
     Input('shift-filter', 'value')])
def pct_box_change(symbols, year_value, shift_val):
    """
    Box plot del porcentaje de cambio de un rango temporal dado.

    Parámetros:
    ----------
        symbols str:
            Nombre de la acción en bolsa.
        year_value str:
            Filtrado de datos por la fecha.
        shift_val str:
            Días sobre los que calcular el porcentaje de cambio.

    Return:
    -------
        dict:
            Figura
    """

    df = t_read.read_data(symbols, year_value)
    df.sort_values(by=['date'], ascending=True, inplace=True)
    df['pct_change'] = df['close'].pct_change(int(shift_val))
    fig = go.Figure([go.Box(x=df['pct_change'] * 100, marker_color=c.PALETTE['yellow'], name='')])
    fig.update_layout(
                   xaxis_showgrid=False,
                   plot_bgcolor=c.PALETTE['black'],
                   paper_bgcolor=c.PALETTE['black'],
                   font={'color': c.PALETTE['grey']},
                   )
    fig.update_xaxes(showline=False, showgrid=False, gridcolor=c.PALETTE['grey'], zerolinecolor=c.PALETTE['grey'])
    fig.update_yaxes(showline=False, showgrid=False, gridcolor=c.PALETTE['grey'], zerolinecolor=c.PALETTE['grey'], linecolor=c.PALETTE['grey'])
    return fig

@app.callback(
    Output('pct-graphic', 'figure'),
    [Input('symbols', 'value'),
     Input('time-filter', 'value'),
     Input('shift-filter', 'value')])
def pct_change(symbols, year_value, shift_val):
    """
    Plot del porcentaje de cambio de un rango temporal dado.

    Parámetros:
    ----------
        symbols str:
            Nombre de la acción en bolsa.
        year_value str:
            Filtrado de datos por la fecha.
        shift_val str:
            Días sobre los que calcular el porcentaje de cambio.

    Return:
    -------
        dict:
            Figura
    """
    df = t_read.read_data(symbols, year_value)
    df.sort_values(by=['date'], ascending=True, inplace=True)
    df['pct_change'] = df['close'].pct_change(int(shift_val))
    fig = go.Figure([go.Scatter(line=dict(color=c.PALETTE['green']), x=df['date'], y=df['pct_change'] * 100)])
    fig.update_layout(
                   xaxis_showgrid=False,
                   xaxis_title='Date',
                   plot_bgcolor=c.PALETTE['black'],
                   paper_bgcolor=c.PALETTE['black'],
                   font={'color': c.PALETTE['grey']},
                   )
    fig.update_yaxes(showgrid=True, zerolinecolor=c.PALETTE['grey'], gridcolor=c.PALETTE['grey'])
    return fig

@app.callback(
    Output('linear-graphic', 'figure'),
    [Input('tech_indc', 'value'),
     Input('symbols', 'value'),
     Input('time-filter', 'value')])
def line_graphs(tech_indc, symbols, year_value):
    """
    Plot de los indicadores técnicos.

    Parámetros:
    ----------
        tech_indc any: Lista de indicadores técnicos a mostrar
        o un str del indicador técnico a mostrar.
        symbols str:
            Nombre de la acción en bolsa.
        year_value str:
            Filtrado de datos por la fecha.

    Return:
    -------
        dict:
            Figura
    """
    df = t_read.read_data(symbols, year_value, True)
    cols = set(df.columns)
    df_melt = pd.melt(df, id_vars=['date'],
                      value_vars=list(cols - set(['date', 'symbol'])))

    fig = go.Figure()
    if isinstance(tech_indc, str):
        tech_indc = [tech_indc]
    for ti in tech_indc:
        if len(c.TECH[ti]) == 1:
            df_fig = df_melt[df_melt['variable'].isin(c.TECH[ti])]
            fig.add_trace(go.Scatter(x=df_fig['date'], y=df_fig['value'],
                        mode='lines',
                        name=ti))
        else:
            for t in c.TECH[ti]:
                df_fig = df_melt[df_melt['variable'] == t]
                fig.add_trace(go.Scatter(x=df_fig['date'], y=df_fig['value'],
                            mode='lines',
                            name=t))
    fig.update_layout(
                   xaxis_showgrid=False,
                   xaxis_title='Date',
                   plot_bgcolor=c.PALETTE['black'],
                   paper_bgcolor=c.PALETTE['black'],
                   font={'color': c.PALETTE['grey']},
                   )
    fig.update_yaxes(showgrid=True, gridcolor=c.PALETTE['grey'], zerolinecolor=c.PALETTE['grey'])
    return fig

@app.callback(
    Output('candle-graphic', 'figure'),
    [Input('symbols', 'value'),
     Input('time-filter', 'value')])
def update_graph(symbols, year_value):
    """
    Gráfico de velas.

    Parámetros:
    ----------
        symbols str:
            Nombre de la acción en bolsa.
        year_value str:
            Filtrado de datos por la fecha.

    Return:
    -------
        dict:
            Figura
    """
    df = t_read.read_data(symbols, year_value)
    return {
        'data': [{
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'x': df['date'],
            'type': 'candlestick',
            'increasing': {'fillcolor': c.PALETTE['green'], 'line': {'color': c.PALETTE['green']}},
            'decreasing': {'fillcolor': c.PALETTE['red'], 'line': {'color': c.PALETTE['red']}}
        }],
        'layout': {
                'plot_bgcolor': c.PALETTE['black'],
                'paper_bgcolor': c.PALETTE['black'],
                'font': {
                    'color': c.PALETTE['grey']
                },
                'xaxis': {'title': 'Date','rangeslider': {'visible': False}},
                'yaxis': {'title': 'Price', 'color': c.PALETTE['grey']}
            }
    }
