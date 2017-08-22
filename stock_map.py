#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
=======================================
Visualizing the stock market structure
=======================================

This example employs several unsupervised learning techniques to extract
the stock market structure from variations in historical quotes.

The quantity that we use is the daily variation in quote price: quotes
that are linked tend to cofluctuate during a day.

.. _stock_market:

Learning a graph structure
--------------------------

We use sparse inverse covariance estimation to find which quotes are
correlated conditionally on the others. Specifically, sparse inverse
covariance gives us a graph, that is a list of connection. For each
symbol, the symbols that it is connected too are those useful to explain
its fluctuations.

Clustering
----------

We use clustering to group together quotes that behave similarly. Here,
amongst the :ref:`various clustering techniques <clustering>` available
in the scikit-learn, we use :ref:`affinity_propagation` as it does
not enforce equal-size clusters, and it can choose automatically the
number of clusters from the data.

Note that this gives us a different indication than the graph, as the
graph reflects conditional relations between variables, while the
clustering reflects marginal properties: variables clustered together can
be considered as having a similar impact at the level of the full stock
market.

Embedding in 2D space
---------------------

For visualization purposes, we need to lay out the different symbols on a
2D canvas. For this we use :ref:`manifold` techniques to retrieve 2D
embedding.


Visualization
-------------

The output of the 3 models are combined in a 2D graph where nodes
represents the stocks and edges the:

- cluster labels are used to define the color of the nodes
- the sparse covariance model is used to display the strength of the edges
- the 2D embedding is used to position the nodes in the plan

This example has a fair amount of visualization-related code, as
visualization is crucial here to display the graph. One of the challenge
is to position the labels minimizing overlap. For this we use an
heuristic based on the direction of the nearest neighbor along each
axis.
"""
print(__doc__)

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause
# Modified by MDA (mda@pyrosome.com)

from datetime import datetime
from datetime import date
import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urlencode
from sklearn import cluster, covariance, manifold
import quandl
from quandl_config import API_CONFIG_KEY

quandl.ApiConfig.api_key = API_CONFIG_KEY

rcParams.update({'font.size': 8})


###############################################################################
# Retrieve the data from Internet

def quotes_historical_google(symbol, date1, date2):
    """Get the historical data from Google finance.
    Parameters
    ----------
    symbol : str
        Ticker symbol to query for, for example ``"DELL"``.
    date1 : datetime.datetime
        Start date.
    date2 : datetime.datetime
        End date.
    Returns
    -------
    X : array
        The columns are ``date`` -- datetime, ``open``, ``high``,
        ``low``, ``close`` and ``volume`` of type float.
    """
    params = urlencode({
        'q': symbol,
        'startdate': date1.strftime('%b %d, %Y'),
        'enddate': date2.strftime('%b %d, %Y'),
        'output': 'csv'
    })
    url = 'http://www.google.com/finance/historical?' + params
    with urlopen(url) as response:
        # print(response, symbol)
        dtype = {
            'names': ['date', 'open', 'high', 'low', 'close', 'volume'],
            'formats': ['object', 'f4', 'f4', 'f4', 'f4', 'f4']
        }
        converters = {0: lambda s: datetime.strptime(s.decode(), '%d-%b-%y')}
        return np.genfromtxt(response, delimiter=',', skip_header=1,
                             dtype=dtype, converters=converters,
                             missing_values='-', filling_values=-1)


# We are looking for 2y max


d1 = datetime(2017, 1, 1)
d2 = datetime(2017, 7, 10)

# d1 = today = datetime.datetime(2016, 1, 1)
# d1 = today = date.today()
# d2 = start = datetime.datetime(2017, 5, 10)

# start = (today.day, today.month, today.year - 1)
# start = str(start[0]) + '-' + str(start[1]) + '-' + str(start[2])


def make_symbol_dic(stocks, **kwargs):

    symbol_dict = {}
    if not kwargs:
        F = open(stocks, 'r')
        stocks = set(F.readlines())
    for symbol in stocks:
        try:
            qnd_symbol = 'WIKI/' + symbol.strip()
            stock = quandl.Dataset(qnd_symbol).name
            symbol_dict.setdefault(symbol.strip(), stock.split(' (', 1)[0])
        except:
            pass
    return symbol_dict


shortlist = [
    'MMM', 'CVX', 'PTEN', 'NVDA', 'TDG',
    'JNJ', 'TSLA', 'F', 'NFLX', 'AMD', 'MU',
    'AMZN', 'AAPL', 'FB', 'BA', 'TDG',
    'BA', 'GOOGL', 'STX', 'WDC', 'TXN', 'AMS', 'WLM',
    'BK', 'WMT'
]

longlist = [
    'AMZN', 'GOOGL', 'TSLA', 'FB', 'GLD', 'NVDA', 'GILD',
    'NESN', 'MSFT', 'BBY', 'INTC', 'AMD', 'IMGN',
    'HBI', 'DIS', 'MEET', 'MCD', 'TWTR',
    'NFLX', 'NXPI', 'BABA', 'JNJ', 'T', 'PG', 'MA',
    'WMT', 'IBM', 'AAPL', 'BBBY', 'MRK', 'GS', 'CRM',
    'MS', 'TDG', 'BA', 'MU', 'MMM', 'ILMN', 'LPL',
    'SNE', 'BAC', 'FOXA', 'GPRO', 'GILD', 'SNAP', 'FIT',
    'TXN', 'STX', 'WDC'
]

pipeline = ['LB', 'FL', 'COL','VXX', 'TVIX', 'XIV', 'SVXY', 'UVXY', 'SNAP', 'AMD']

symbol_dict = make_symbol_dic(pipeline, t=list)
symbols, names = np.array(list(symbol_dict.items())).T

'''
quotes = []
for symbol in symbols:
    try:
        quote = quotes_historical_google(symbol, d1, d2)
        quotes.append(quote)
    except Exception as e:
        print(e)

close_prices = np.stack([q['close'] for q in quotes])
open_prices = np.stack([q['open'] for q in quotes])
'''

for symbol in symbols:
    try:
        quotes = [quandl.get('WIKI/' + symbol, start_date=d1,
                             end_date=d2, returns='numpy')
                  for symbol in symbols]
    except Exception as e:
        print(e)

close_prices = np.stack([q['Close'] for q in quotes])
open_prices = np.stack([q['Open'] for q in quotes])


# The daily variations of the quotes are what carry most information
variation = close_prices - open_prices

###############################################################################
# Learn a graphical structure from the correlations
edge_model = covariance.GraphLassoCV()

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
X = variation.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

###############################################################################
# Cluster using affinity propagation

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

###############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T

###############################################################################
# Visualization
plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.show()
