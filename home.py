"""
Pantalla principal de la aplicación.
"""
import dash_core_components as dcc
import dash_html_components as html

import constants as c

# Creación del HTML mediante DASH
layout_home = html.Div(style={'backgroundColor': c.PALETTE['black']}, children= [
    html.Br(),
    html.H1(
        children='Beating Stock Markets',
        style={
            'textAlign': 'center',
            'color': c.PALETTE['yellow']
        }),

    html.Div(children='version :: 0.3', style={
        'textAlign': 'center',
        'color': c.PALETTE['grey']
    }),

    html.Div(children='Esteban Sánchez', style={
        'textAlign': 'center',
        'color': c.PALETTE['grey']
    }),
    html.Br(),
    html.Div(style={
        'textAlign': 'center'
        },
        children=dcc.Link('Update Values', href='/update', style={
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
            })),

    html.Br()
        ])
