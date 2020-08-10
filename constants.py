"""
En este archivo se encuentran constantes usadas dentro de todo el proyecto.
"""
import os
import random

ALPHA_VAN_KEY = "ENTER_YOUR_API_KEY_HERE"
COLUMNS = ['symbol', 'date', 'close', 'volume', 'open', 'high', 'low']
COLUMNS_TECH = ['symbol', 'date', 'MACD_Signal', 'MACD_Hist', 'MACD', 'SlowK', 'SlowD',
       'Chaikin A/D', 'OBV', 'RSI21', 'ADX21',
       'CCI21', 'Aroon Up21', 'Aroon Down21',
       'RSI28', 'ADX28', 'CCI28', 'Aroon Down28', 'Aroon Up28',
       'Real Lower Band28', 'Real Upper Band28', 'Real Middle Band28',
       'SMA50', 'RSI50', 'ADX50', 'CCI50', 'Aroon Up50',
       'Aroon Down50']
COLS_PAST_DATA = ['tag_y', 'PC 1', 'PC 2', 'PC 3', 'PC 4']
MODEL_COLUMNS = ['close', 'volume', 'MACD_Signal', 'MACD_Hist', 'MACD', 'SlowK', 'SlowD',
                 'Chaikin A/D', 'OBV', 'RSI21', 'ADX21', 'CCI21', 'Aroon Up21', 'Aroon Down21',
                 'RSI28', 'ADX28', 'CCI28', 'Aroon Down28', 'Aroon Up28', 'Real Lower Band28',
                 'Real Upper Band28', 'Real Middle Band28', 'SMA50', 'RSI50', 'ADX50', 'CCI50',
                 'Aroon Up50', 'Aroon Down50']
TECH = {
    'CLOSE': ['close'],
    'VOLUME': ['volume'],
    'MACD': ['MACD', 'MACD_Hist', 'MACD_Signal'],
    'STOCH': ['SlowK', 'SlowD'],
    'AD': ['Chaikin A/D'],
    'OBV': ['obv'],
    'SMA 50': ['SMA50'],
    'RSI 21': ['RSI21'],
    'RSI 28': ['RSI28'],
    'RSI 50': ['RSI50'],
    'CCI 21': ['CCI21'],
    'CCI 28': ['CCI28'],
    'CCI 50': ['CCI50'],
    'AROON 21': ['Aroon Up21', 'Aroon Down21'],
    'AROON 28': ['Aroon Up28', 'Aroon Down28'],
    'AROON 50': ['Aroon Up50', 'Aroon Down50'],
    'ADX 21': ['ADX21'],
    'ADX 28': ['ADX28'],
    'ADX 50': ['ADX50'],
    'BBANDS 28': ['Real Lower Band28', 'Real Middle Band28', 'Real Upper Band28'],
    }
TIME_PERIOD = [21, 28, 50]
PATH_SAVE_CLOSE = './data/close/'
PATH_SAVE_TECH = './data/tech/'
PATH_PAST_VAL = './data/group/'
PATH_MODEL = './data/model/'
PATH_SYMBOL = './data/symbol/'
PALETTE = {
    'black': '#272727',
    'yellow': '#FFE400',
    'red': '#FF652F',
    'green': '#14A76C',
    'blue': '#2260bf',
    'pink': '#C5386F',
    'grey': '#747474'
    }
random.seed(32)
SYMBOLS = random.sample([s.split('.csv')[0] for s in os.listdir(f"{PATH_SAVE_TECH}") if '.L' not in s], 250)
