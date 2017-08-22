#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 23:23:23 201
@author: mda

Based on the Quantopian notebook "Plot Candlestick Charts in Reserach"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
from matplotlib.finance import quotes_historical_yahoo_ochl
from datetime import date
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
import matplotlib
from pattern_analysis import plotstock

matplotlib.rcParams.update({'font.size': 8})

stocklist = ['EZJ', 'AMZN', 'GOOGL', 'SPY',
             'TSLA', 'FB', 'GLD', 'NVDA', 'GILD',
             'NESN', 'MSFT', 'BBY', 'INTC', 'LOGI', 'SLV', 'AMD', 'IMGN',
             'HBI', 'DIS', 'MEET', 'INTU', 'INVN', 'ERII',
             'MCD', 'VHC', 'HIMX', 'WHR', 'TWTR', 'UAA',
             'NFLX', 'NXPI', 'BABA', 'T', 'KHC', 'JNJ',
             'PG', 'MA', 'WMT', 'IBM', 'AAPL', 'BBBY',
             'WSM', 'MRK', 'XBI', 'GSBD', 'RGLD', 'CRM',
             'SA', 'FN', 'NSA', 'HP', 'ZNGA', 'CVA', 'TTDKY',
             'ETRM', 'BK', 'PYPL', 'MU', 'JCP', 'BBRY', 'YHOO',
             'RF', 'FOXA', 'MS', 'BAC', 'XL', 'SGEN', 'CL',
             'FOSL', 'RGR', 'SWHC', 'TASR', 'MMM', 'TECK', 'KNSL',
             'AMTD', 'LOCK', 'AXP', 'V', 'WEB', 'DRYS', 'CSGN',
             'NWN', 'SGYP', 'MVIS', 'LPL', 'VRX', 'SNE', 'CNAT', 'XON', 'S',
             'MGNX', 'BCG', 'GME', 'NYLD', 'INAP', 'PTLA', 'VOD', 'VZ',
             'GM', 'F', 'TM', 'ILMN', ]

shortlist = ['MMM', 'CVX', 'PTEN', 'NVDA', 'TDG', 'JNJ']

today = date.today()
start = (today.year - 1, today.month, today.day)
mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')      # e.g., 12


def rsiFunc(prices, n=14):

    # pdeltas = prices
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi


def movingaverage(values, window):
    weigths = np.repeat(1.0, window) / window
    smas = np.convolve(values, weigths, 'valid')
    return smas  # as a numpy array


def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def computeMACD(x, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow


# Define the MACD function
def MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9):
    '''
    Function to return the difference between the most recent
    MACD value and MACD signal. Positive values are long
    position entry signals

    optional args:
        fastperiod = 12
        slowperiod = 26
        signalperiod = 9

    Returns: macd - signal
    '''
    macd, hist, signal = talib.MACD(prices,
                                    fastperiod=fastperiod,
                                    slowperiod=slowperiod,
                                    signalperiod=signalperiod)
    # return macd[-1] - signal[-1]
    return macd, hist, signal


def plot_candles(pricing, title=None,
                 timezone='US/Eastern',
                 max_x_ticks=15,
                 volume_bars=False,
                 color_function='redgreen',
                 overlays=None,
                 technicals=None,
                 technicals_titles=None):
    '''
    Plots a candlestick chart using quantopian pricing data.
    Author: Daniel Treiman
    Args:
    pricing: A pandas dataframe with columns 
    ['open_price', 'close_price', 'high', 'low', 'volume']

    title: An optional title for the chart

    timezone: timezone to use for formatting intraday data.

    max_x_ticks: The maximum number of X ticks with labels, 
    to keep the X axis readable.

    volume_bars: If True, plots volume bars

    color_function: ('redgreen', 'hollow') or a function which, 
    given a row index and price data, returns a candle color.

    overlays: A list of additional data series to overlay on top of pricing.
    Must be the same length as pricing.

    technicals: A list of additional data series to display as subplots.

    technicals_titles: A list of titles to display 
    for each technical indicator.
    '''

    overlays = overlays or []
    technicals = technicals or []
    technicals_titles = technicals_titles or []

    # ---- Builtin color functions ---
    def color_function_red_green(index, open_price, close_price, low, high):

        return 'r' if open_price[index] > close_price[index] else 'g'

    def color_function_filled_hollow(index,
                                     open_price, close_price, low, high):

        return 'k' if open_price[index] > close_price[index] else 'w'

    if color_function == 'redgreen':
        color_function = color_function_red_green
    elif color_function == 'hollow':
        color_function = color_function_filled_hollow

    open_price = pricing['open_price']
    close_price = pricing['close_price']
    low = pricing['low']
    high = pricing['high']
    oc_min = pd.concat(
        [open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat(
        [open_price, close_price], axis=1).max(axis=1)

    # ---- Set up plot layout ----
    subplot_count = 4

    if technicals:
        subplot_count += len(technicals)

    ratios = np.insert(np.full(subplot_count - 1, 1), 0, 3)

    fig, subplots = plt.subplots(subplot_count, 1, sharex=True,
                                 gridspec_kw={'height_ratios': ratios})

    ax1 = subplots[0]

    if title:
        ax1.set_title(title)

    # ---- Plot bars and wicks ----
    x = np.arange(len(pricing))
    x_midline = x + 0.4  # Approximate left of the centerline of each bar
    bar_width = 0.8
    # Hack: shrink bar width as we increase the number of
    # bars to prevent edge overlap
    if len(x) > 100:
        bar_width = 0.5
        x_midline = x + 0.3
    candle_colors = [color_function(i, open_price, close_price, low, high) or
                     color_function_filled_hollow(i, open_price,
                                                  close_price,
                                                  low, high) for i in x]
    edge_colors = ['k' if c == 'w' else c for c in candle_colors]
    ax1.bar(x, oc_max - oc_min, bottom=oc_min, color=candle_colors,
            edgecolor=edge_colors, width=bar_width, linewidth=1)
    ax1.vlines(x_midline, low, oc_min, color=edge_colors,
               linewidth=1)  # low wick
    ax1.vlines(x_midline, oc_max, high, color=edge_colors,
               linewidth=1)  # high wick

    plt.xticks(x, [date for date in pricing.index], rotation=90)
    plt.ylabel('Stock price')

    ax1.xaxis.grid(False)
    ax1.xaxis.set_tick_params(
        which='major', length=3.0, direction='in', top='off')
    labels_x = x_midline
    labels = np.array([date for date in pricing.index])

    if len(x) > max_x_ticks:
        if len(x) / 2 < max_x_ticks:  # Label every other day
            labels_x = labels_x[::2]
            labels = labels[::2]
    plt.xticks(labels_x, labels, ha='center')

    # ---- Plot all overlays ----
    for overlay in overlays:
        ax1.plot(x, overlay)

    # ---- Plot RSI ----
    ax2 = subplots[1]
    rsi = rsiFunc(close_price)
    # rsi = talib.RSI(pricing['close_price'].as_matrix())
    # SP = len(pricing.index[-60:])

    rsiCol = '#FF6600'
    posCol = '#386d13'
    negCol = '#8f2020'

    ax2.plot(x, rsi, rsiCol, linewidth=1.5, label='RSI')

    ax2.axhline(70, color=negCol, label='>70: Overbought')
    ax2.axhline(30, color=posCol, label='<30: Oversold')

    ax2.fill_between(x, rsi, 70,
                     where=(rsi >= 70),
                     facecolor=negCol,
                     edgecolor=negCol, alpha=0.5)
    ax2.fill_between(x, rsi, 30,
                     where=(rsi <= 30),
                     facecolor=posCol,
                     edgecolor=posCol, alpha=0.5)

    ax2.set_yticks([30, 70])
    ax2.yaxis.label.set_color("b")
    ax2.spines['bottom'].set_color("w")
    ax2.spines['top'].set_color("w")
    ax2.spines['left'].set_color("w")
    ax2.spines['right'].set_color("w")
    ax2.tick_params(axis='y', colors='b')
    ax2.tick_params(axis='x', colors='b')
    ax2.set_ylim(0, 100)
    # plt.ylabel('RSI')

    # ---- Plot MACD ----

    ax3 = subplots[2]
    # compute the MACD indicator
    fillcolor = '#3771C8'
    emaslow = 26
    emafast = 12
    nema = 9
    emaslow, emafast, macd = computeMACD(close_price)
    ema9 = ExpMovingAverage(macd, nema)
    ax3.axhline(0)
    ax3.plot(x, macd, color='#FF0066', lw=1.5, label='MACD')
    ax3.plot(x, ema9, color='blue', lw=1, label='EMA')
    ax3.bar(x, macd - ema9, 0.95, alpha=0.5,
            facecolor=fillcolor, edgecolor=fillcolor, label='Signal')

    handles1, labels1 = ax2.get_legend_handles_labels()
    ax2.legend(loc='best', shadow=True, fancybox=True)

    handles1t, labels1t = ax3.get_legend_handles_labels()
    ax3.legend(loc='lower left', shadow=True, fancybox=True)

    if volume_bars:
        ax4 = subplots[3]
        volume = pricing['volume']
        volume_scale = None
        scaled_volume = volume
        if volume.max() > 1e6:
            volume_scale = 'M'
            scaled_volume = volume / 1e6
        elif volume.max() > 1e3:
            volume_scale = 'K'
            scaled_volume = volume / 1e3
        ax4.bar(x, scaled_volume, color=candle_colors)
        volume_title = 'Volume'
        if volume_scale:
            volume_title = 'Volume (%s)' % volume_scale
        ax4.set_title(volume_title)
        ax4.xaxis.grid(False)

    # ---- Plot additional technical indicators in subplots ----
    for (i, technical) in enumerate(technicals):
        # Technical indicator plots are shown last
        ax = subplots[i - len(technicals)]
        ax.plot(x, technical)
        if i < len(technicals_titles):
            ax.set_title(technicals_titles[i])

    plt.show()
    fig.savefig(stock + '.png', facecolor=fig.get_facecolor())


for stock in stocklist:

    print("Fetching", stock)

    columns = ['date', 'open_price', 'close_price', 'high', 'low', 'volume']
    ticker = quotes_historical_yahoo_ochl(stock, start, today)

    list1 = []
    for i in range(0, len(ticker)):
        x = date.fromordinal(int(ticker[i][0]))
        y = date.strftime(x, '%Y-%m-%d')
        list1.append(y)

    day_pricing = pd.DataFrame(ticker, index=list1, columns=columns)
    # day_pricing = day_pricing.drop(['date'], axis=1)
    last_hour = day_pricing[-100:]

    openp = last_hour['open_price'].as_matrix()
    highp = last_hour['high'].as_matrix()
    lowp = last_hour['low'].as_matrix()
    closep = last_hour['close_price'].as_matrix()
    volp = last_hour['volume'].as_matrix()

    rsi1 = talib.RSI(last_hour['close_price'].as_matrix())
    macd = talib.MACD(last_hour['close_price'].as_matrix())

    if rsi1[-1:] < 30 or rsi1[-1:] > 70:
        print('Currently analyzing ', stock, rsi1[-1])
        upper, middle, lower = talib.BBANDS(
            last_hour['close_price'].as_matrix())
        EMAL = talib.EMA(last_hour['close_price'].as_matrix(), 50)
        EMAS = talib.EMA(last_hour['close_price'].as_matrix(), 21)
        gravestone = talib.CDLGRAVESTONEDOJI(openp, highp, lowp, closep)
        obv = talib.OBV(closep, volp)

        plot_candles(last_hour, title=stock, volume_bars=True,
                     overlays=[EMAL, EMAS],
                     technicals=[obv], technicals_titles=['OBV'])
