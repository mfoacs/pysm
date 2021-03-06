#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Author : MDA (mda@pyrosome.com)
    Combining Mean Reversion and MACD learning.
    TODO: 
        1. cut_losses shall be applied `before_trading_start` (performance)
        2. integrate correlations of stock_map.py (covariance matrix)
"""

# Zipline imports
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import AverageDollarVolume, Returns
from zipline.api import (order, record, symbol,
                         set_commission, attach_pipeline,
                         pipeline_output,
                         schedule_function,
                         time_rules, date_rules, order_target_percent)
from zipline.finance import commission

# External imports
from datetime import date
import numpy as np
from sklearn import svm
from talib import MACD

# On Quantopian we just used talib.MACD
# With CONDA we import our own from studies because talib isn't available (Python 3.6*).

# from studies import macd


def initialize(context):
    ''' Initialize global vars'''
    context.long_leverage = 0.1
    context.short_leverage = -0.9
    context.returns_lookback = 16
    context.pct_per_stock = 0.5
    
    context.fastperiod = 12
    context.slowperiod = 26
    context.signalperiod = 9
    context.bar_count = 90

    set_commission(commission.PerShare(cost=0.0014, min_trade_cost=1))
    
    # Rebalance on the first trading day of each week at 12AM.
    schedule_function(rebalance, date_rules.week_start(days_offset=0),time_rules.market_open(hours=0.5))
    
    # Rebalance mid-week
    schedule_function(cut_losses, date_rules.week_start(days_offset=2),time_rules.market_open(hours=0.5))

    # Record tracking variables at the end of each day.
    schedule_function(record, date_rules.every_day(),time_rules.market_open(minutes=1))


    # Create and attach our pipeline (dynamic stock selector), defined below.
    attach_pipeline(make_pipeline(context),
                    'mean_reversion_macd_learning')
    

def make_pipeline(context):
    """
    A function to create our pipeline (dynamic stock selector). The pipeline is used
    to rank stocks based on different factors, including builtin factors, or custom
    factors that you can define. Documentation on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """
    # Create a pipeline object.

    # Create a dollar_volume factor using default inputs and window_length.
    # This is a builtin factor.
    dollar_volume = AverageDollarVolume(window_length=1)

    # Define high dollar-volume filter to be the top 2% of stocks by dollar
    # volume.
    high_dollar_volume = dollar_volume.percentile_between(95, 100)

    # Create a recent_returns factor with a 5-day returns lookback for all securities
    # in our high_dollar_volume Filter. This is a custom factor defined below (see
    # RecentReturns class).
    recent_returns = Returns(
        window_length=16, mask=high_dollar_volume)

    # Define high and low returns filters to be the bottom 1% and top 1% of
    # securities in the high dollar-volume group.
    low_returns = recent_returns.percentile_between(0, 5)
    high_returns = recent_returns.percentile_between(95, 100)

    # Define a column dictionary that holds all the Factors
    pipe_columns = {
        'low_returns': low_returns,
        'high_returns': high_returns,
        'recent_returns': recent_returns,
        'dollar_volume': dollar_volume
    }

    # Add a filter to the pipeline such that only high-return and low-return
    # securities are kept.
    # pipe_screen = (low_returns & liquidity_filter | high_returns & vol_filter)
    pipe_screen = (low_returns | high_returns)

    # Create a pipeline object with the defined columns and screen.
    pipe = Pipeline(columns=pipe_columns, screen=pipe_screen)

    return pipe



def before_trading_start(context, data):
    """
    Called every day before market open. This is where we get the securities
    that made it through the pipeline.
    """

    # Pipeline_output returns a pandas DataFrame with the results of our factors
    # and filters.
    context.output = pipeline_output('mean_reversion_macd_learning')

    # Sets the list of securities we want to long as the securities with a 'True'
    # value in the low_returns column.
    context.long_secs = context.output[context.output['low_returns']]

    # Sets the list of securities we want to short as the securities with a 'True'
    # value in the high_returns column.
    context.short_secs = context.output[context.output['high_returns']]

    # A list of the securities that we want to order today.
    context.security_list = context.long_secs.index.union(context.short_secs.index).tolist()

    # A set of the same securities, sets have faster lookup.
    context.security_set = set(context.security_list)

    print(context.output)


def compute_weights(context):
    """
    Compute weights to our long and short target positions.
    """

    # Set the allocations to even weights for each long position, and even weights
    # for each short position.
    long_weight = context.long_leverage / len(context.long_secs)
    short_weight = context.short_leverage / len(context.short_secs)

    return long_weight * 1.1, short_weight * 1.1


def rebalance(context, data):
    """
    This rebalancing function is called according to our
    schedule_function settings.
    """

    long_weight, short_weight = compute_weights(context)

    # For each security in our universe, order long or short
    # positions according to our context.long_secs and context.short_secs lists.
    for stock in context.security_list:
        if data.can_trade(stock):
            if stock in context.long_secs.index:
                order_target_percent(stock, long_weight)
                print('Long: ', stock)
            elif stock in context.short_secs.index:
                order_target_percent(stock, short_weight)
                print('Short: ', stock)

    # Sell all previously held positions not in our new context.security_list.
    for stock in context.portfolio.positions:
        if stock not in context.security_set and data.can_trade(stock):
            order_target_percent(stock, 0)


def cut_losses(context,data):
    """ 
    Sells/Invert positions according to returns. 

    """
    
    for stock in context.portfolio.positions:
        
        #current_position = context.portfolio.positions[stock].amount
        prices = data.history(stock,fields='close',
                              bar_count=context.bar_count,
                              frequency='1m')
        
        hist = data.history(stock,
                            fields=['price', 'open', 'high', 'low', 'close', 'volume'],
                            bar_count=context.bar_count,
                            frequency='1d')

        try:
            macd = MACD(prices,
                        fastperiod=context.fastperiod,
                        slowperiod=context.slowperiod, 
                        signalperiod=context.signalperiod)
            
            price = context.portfolio.positions[stock].last_sale_price            
            predicted = predict_prices(hist['close'], prices, context.returns_lookback)
            
            # print('Current price: Predicted price', price, predicted)
            if stock in context.long_secs.index:
                if macd < 0 and predicted < 1.2*price: 
                    order_target_percent(stock, 0)
            else:
                if macd > 0 and 1.1*predicted > price: 
                    order_target_percent(stock, 0)
        except Exception as e:
            print(e)


def predict_prices(dates, prices, x):
    '''
    SVM RBF regression
    '''
    dates = np.reshape(dates,(len(dates),1))
    svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(dates,prices)

    return svr_rbf.predict(x)[0]

