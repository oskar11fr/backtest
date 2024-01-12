from engine.utils import load_pickle
from engine.performance import performance_measures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# btc = load_pickle('strategy_obj/btc_start.obj')
bbwidth_pe = load_pickle('strategy_obj/bbwidth_pe_strat.obj')
syst_yield = load_pickle('strategy_obj/syst_yield_strat.obj')
vix_strat = load_pickle('strategy_obj/vix_strat.obj')
pead_strat = load_pickle("strategy_obj/pead_strat.obj")
weights = np.array([0.3333,0.3333,0.13333,0.2])

strategies = pd.concat([bbwidth_pe,syst_yield,vix_strat,pead_strat],axis=1)
strategies.columns = ['bbwidth_pe', 'systematic yield', 'vix', "pead"]
portfolio = pd.Series(np.dot(strategies.values,weights),index=strategies.index).fillna(0.)
# (1+strategies.loc['2014-01-01':]*weights).cumprod(axis=0).apply(np.log).plot()
# (1+portfolio.loc['2014-01-01':]).cumprod(axis=0).apply(np.log).plot()
# plt.show();exit()
performance_measures(r = portfolio[portfolio!=0.], plot=True)
# import quantstats as qs
# qs.reports.html(portfolio[portfolio!=0.], "SPY", output="images/portfolio.html")
