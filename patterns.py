#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Candlestick Pattern Recognition Indicators

https://discuss.tradewave.net/t/candlestick-pattern-recognition-indicators-ie-dragonfly-doji/167

Modified by lewk to use the talib abstract Function API.

[-1] included on each to convert array to "last" value
max() and [:-5] could be used to see if any of last 5 were '1' output
int() function used to convert int32 data to integers for plotting
/100 because default talib outputs are 0 or 100
Default parameters used where "penetration" is required input

Most Return:  Binary [0, 1] or Ternary [-1, 0, 1]
Hikkaki Returns:  Quinary [-2, -1, 0, 1, 2]
'''
# Modified by mda@pyrosome.com to use Quandl prices

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from datetime import date
import pandas as pd
import quandl
from talib.abstract import Function
import config

matplotlib.rcParams.update({'font.size': 8})

quandl.ApiConfig.api_key = config.API_CONFIG_KEY

PATTERNS = [
    #    '2CROWS',            # Two Crows
    '3BLACKCROWS',       # Three Black Crows
    #    '3INSIDE',           # Three Inside Up/Down
    '3LINESTRIKE',       # Three-Line Strike
    #    '3OUTSIDE',          # Three Outside Up/Down
    #    '3STARSINSOUTH',     # Three Stars In The South
    '3WHITESOLDIERS',    # Three Advancing White Soldiers
    'ABANDONEDBABY',     # Abandoned Baby
    #    'ADVANCEBLOCK',      # Advance Block
    #    'BELTHOLD',          # Belt-hold
    'BREAKAWAY',         # Breakaway
    'CLOSINGMARUBOZU',   # Closing Marubozu
    #    'CONCEALBABYSWALL',  # Concealing Baby Swallow
    #    'COUNTERATTACK',     # Counterattack
    #    'DARKCLOUDCOVER',    # Dark Cloud Cover
    #    'DOJI',              # Doji
    #    'DOJISTAR',          # Doji Star
    #    'DRAGONFLYDOJI',     # Dragonfly Doji
    'ENGULFING',         # Engulfing Pattern
    'EVENINGDOJISTAR',   # Evening Doji Star
    'EVENINGSTAR',       # Evening Star
    #    'GAPSIDESIDEWHITE',  # Up/Down-gap side-by-side white lines
    'GRAVESTONEDOJI',    # Gravestone Doji
    #    'HAMMER',            # Hammer
    #    'HANGINGMAN',        # Hanging Man
    #    'HARAMI',            # Harami Pattern
    #    'HARAMICROSS',       # Harami Cross Pattern
    #    'HIGHWAVE',          # High-Wave Candle
    #'HIKKAKE',           # Hikkake Pattern
    #'HIKKAKEMOD',        # Modified Hikkake Pattern
    #'HOMINGPIGEON',      # Homing Pigeon
    #'IDENTICAL3CROWS',   # Identical Three Crows
    #'INNECK',            # In-Neck Pattern
    #'INVERTEDHAMMER',    # Inverted Hammer
    #'KICKING',           # Kicking
    #'KICKINGBYLENGTH',   # Kicking - bull/bear determined by the longer marubozu
    #'LADDERBOTTOM',      # Ladder Bottom
    #'LONGLEGGEDDOJI',    # Long Legged Doji
    #'LONGLINE',          # Long Line Candle
    #'MARUBOZU',          # Marubozu
    #'MATCHINGLOW',       # Matching Low
    #'MATHOLD',           # Mat Hold
    #'MORNINGDOJISTAR',   # Morning Doji Star
    'MORNINGSTAR',       # Morning Star
    #'ONNECK',            # On-Neck Pattern
    #'PIERCING',          # Piercing Pattern
    #'RICKSHAWMAN',       # Rickshaw Man
    #'RISEFALL3METHODS',  # Rising/Falling Three Methods
    #'SEPARATINGLINES',   # Separating Lines
    'SHOOTINGSTAR',      # Shooting Star
    #'SHORTLINE',         # Short Line Candle
    #'SPINNINGTOP',       # Spinning Top
    #'STALLEDPATTERN',    # Stalled Pattern
    #'STICKSANDWICH',     # Stick Sandwich
    #'TAKURI',            # Takuri (Dragonfly Doji with very long lower shadow)
    #'TASUKIGAP',         # Tasuki Gap
    #'THRUSTING',         # Thrusting Pattern
    #'TRISTAR',           # Tristar Pattern
    #'UNIQUE3RIVER',      # Unique 3 River
    #'UPSIDEGAP2CROWS',   # Upside Gap Two Crows
    #'XSIDEGAP3METHODS',  # Upside/Downside Gap Three Methods
]
PATTERN_WINDOW = 250

today = date.today()
start = (today.day, today.month, today.year - 1)
start = str(start[0]) + '-' + str(start[1]) + '-' + str(start[2])


def plotstock(stock, cname):
    '''
    Plots the selected patterns of the stock argument (pd.DataFrame)
    '''
    def candlesrg(index, open_price, close_price, low, high):
        return 'r' if open_price[index] > close_price[index] else 'g'

    candle = stock[-60:]
    O = candle['Open'].as_matrix()
    H = candle['High'].as_matrix()
    L = candle['Low'].as_matrix()
    C = candle['Close'].as_matrix()

    x = np.arange(len(candle))
    fig = plt.figure()
    ax0 = fig.add_subplot(111, axisbg='#f5f5f5')
    ax0.grid(False)
    ax0.set_title(cname)

    oc_min = pd.concat(
        [candle['Open'], candle['Close']], axis=1).min(axis=1)
    oc_max = pd.concat(
        [candle['Open'], candle['Close']], axis=1).max(axis=1)

    x_midline = x + 0.4  # Approximate left of the centerline of each bar
    bar_width = 0.8
    # Hack: shrink bar width as we increase the number of
    # bars to prevent edge overlap
    if len(x) > 100:
        bar_width = 0.5
        x_midline = x + 0.3
    candle_colors = [candlesrg(i, O, C, L, H) for i in x]
    edge_colors = ['k' if c == 'w' else c for c in candle_colors]
    ax0.bar(x, oc_max - oc_min, bottom=oc_min, color=candle_colors,
            edgecolor=edge_colors, width=bar_width, linewidth=1)
    ax0.vlines(x_midline, L, oc_min, color=edge_colors,
               linewidth=1)  # low wick
    ax0.vlines(x_midline, oc_max, H, color=edge_colors,
               linewidth=1)  # high wick

    plt.xticks(x, [date for date in candle.index], rotation=45)

    for pattern in PATTERNS:
        func = Function('CDL%s' % pattern)
        output = func(dict(open=O, high=H, low=L, close=C)) / 10
        ax0.plot(x, output, '1', markeredgewidth=1.2,
                 label=pattern, linewidth=1.3)
    handles1, labels1 = ax0.get_legend_handles_labels()
    ax0.legend(loc='best', shadow=False, fancybox=True)
    plt.show()


stock = quandl.get('WIKI/NFLX', start_date=start,
                   end_date=today.strftime('%d-%m-%Y'))

plotstock(stock, 'NFLX')
