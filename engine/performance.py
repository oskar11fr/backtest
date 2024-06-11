import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
plt.style.use("seaborn-dark-palette")

def performance_measures(r, plot=False, path="/images", market=None, show=False):
    moment = lambda x,k: np.mean((x-np.mean(x))**k)
    stdmoment = lambda x,k: moment(x,k)/moment(x,2)**(k/2)
    cr = np.cumprod(1 + r)
    lr = np.log(cr)
    mdd=cr/cr.cummax() - 1
    rdd_fn = lambda cr,pr: cr/cr.rolling(pr).max() - 1
    rmdd_fn = lambda cr,pr: rdd_fn(cr,pr).rolling(pr).min()
    srtno = np.mean(r.values)/np.std(r.values[r.values<0])*np.sqrt(253)
    shrpe = np.mean(r.values)/np.std(r.values)*np.sqrt(253)
    mu1 = np.mean(r)*253
    med = np.median(r)*253
    stdev = np.std(r)*np.sqrt(253)
    var = stdev**2
    skw = stdmoment(r,3)
    exkurt = stdmoment(r,4)-3
    cagr_fn = lambda cr: (cr[-1]/cr[0])**(1/len(cr))-1
    cagr_ann_fn = lambda cr: ((1+cagr_fn(cr))**253) - 1
    cagr = cagr_ann_fn(cr)
    rcagr = cr.rolling(5*253).apply(cagr_ann_fn,raw=True)
    calmar = cr.rolling(3*253).apply(cagr_ann_fn,raw=True) / rmdd_fn(cr=cr,pr=3*253)*-1
    var95 = np.percentile(r,0.95)
    cvar = r[r < var95].mean()
    table = {
        "cum_ret": cr,
        "log_ret": lr,
        "max_dd": mdd,
        "cagr": cagr,
        "srtno": srtno,
        "sharpe": shrpe,
        "mean_ret": mu1,
        "median_ret": med,
        "vol": stdev,
        "var": var,
        "skew": skw,
        "exkurt": exkurt,
        "cagr": cagr,
        "rcagr": rcagr,
        "calmar": calmar,
        "var95": var95,
        "cvar": cvar
    }
    if plot:
        fig = plt.figure(constrained_layout=True,figsize=(16,14))
        ax = fig.add_gridspec(4, 4)
        
        ax1 = fig.add_subplot(ax[0:2, 0:3])
        ax2 = fig.add_subplot(ax[2, 0:3],sharex=ax1)
        ax3 = fig.add_subplot(ax[0:2, -1])
        ax4 = fig.add_subplot(ax[2, -1],sharey=ax2)
        ax5 = fig.add_subplot(ax[-1,0:3],sharex=ax1)
        ax6 = fig.add_subplot(ax[-1,-1],sharey=ax5)

        ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax5.xaxis.set_major_locator(plt.MaxNLocator(5))
        
        idxs = [d.strftime('%Y-%m-%d %X') for d in r.index]
        if market is not None:
            if (type(market)==dict) | (type(market)==pd.DataFrame):
                for strat,data in market.items():
                    ax1.plot(
                        idxs,
                        np.log(np.cumprod(1+data.loc[r.index].pct_change().fillna(0))),
                        label=strat,
                        linestyle=":",
                        alpha=0.75
                    )
            else:
                ax1.plot(idxs,np.log(np.cumprod(1+market.loc[r.index]["ret"])))
        ax1.plot(idxs,lr)
        ax1.set_ylabel('log capital returns')
        ax1.legend()

        ax2.plot(idxs,rdd_fn(cr,253))
        ax2.plot(idxs,rmdd_fn(cr,253))
        ax2.set_ylabel('drawdowns')

        pd_series = pd.Series(table)[[
            "cagr",
            "srtno",
            "sharpe",
            "mean_ret",
            "median_ret",
            "vol",
            "var",
            "skew",
            "exkurt",
            "cagr",
            "var95"
        ]].apply(lambda x:np.round(x,3))
        
        pd_frame = pd_series.reset_index().rename(columns={'index':'Metric',0:'Value'})
        ax3.table(
            cellText=pd_frame.values,colLabels=pd_frame.keys(),loc='center',colWidths=[0.3,0.3],colColours=['grey','grey'],cellLoc='left'
        )
        ax3.axis('off')
        
        if len(r) > 253:
            ax4.hist(rdd_fn(cr,253),orientation='horizontal',bins=40)

        ax6.hist(r,orientation='horizontal',bins=60)
        ax5.plot(idxs,r)
        ax5.set_ylabel('Returns')
        
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)

        if not show:
            Path(os.path.abspath(os.getcwd()+path)).mkdir(parents=True,exist_ok=True)
            fig.savefig(f".{path}/stats_board.png")
            plt.close()
        else:
            plt.show()
    return table

def plot_hypothesis(timer_tuple,picker_tuple,trader_tuple,return_samples,strat_name):
    path="/images"
    Path(os.path.abspath(os.getcwd()+path)).mkdir(parents=True,exist_ok=True)
    timer_paths,timer_p,timer_dist = timer_tuple
    picker_paths,picker_p,picker_dist = picker_tuple
    trader_paths,trader_p,trader_dist = trader_tuple

    timer_paths.index = return_samples.index
    picker_paths.index = return_samples.index
    trader_paths.index = return_samples.index

    fig = plt.figure(constrained_layout=True,figsize=(15,11))
    ax = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(ax[:2, :])
    ax2 = fig.add_subplot(ax[2:, :2])
    ax3 = fig.add_subplot(ax[2:, -1])

    ax1.plot((1+timer_paths).cumprod().apply(np.log),color='red',alpha=0.3)
    ax1.plot((1+picker_paths).cumprod().apply(np.log),color='blue',alpha=0.3)
    ax1.plot((1+trader_paths).cumprod().apply(np.log),color='green',alpha=0.3)
    ax1.plot((1+return_samples).cumprod().apply(np.log),color='black',linewidth=4)

    stats1,stats2,stats3 = timer_dist,picker_dist,trader_dist
    pd.Series(stats1).plot(color='red',ax=ax2,kind='kde')
    pd.Series(stats2).plot(color='blue',ax=ax2,kind='kde')
    pd.Series(stats3).plot(color='green',ax=ax2,kind='kde')
    shrpe = np.mean(return_samples.values)/np.std(return_samples.values)*np.sqrt(253)
    ax2.axvline(shrpe,color='black',linewidth=4)
    ps = pd.Series({
        "Permuted MC (Timing)": timer_p,
        "Permuted MC (Picking)": picker_p,
        "Permuted MC (Skill)": trader_p
    }).apply(lambda x: np.round(x,4)).reset_index().rename(columns={'index':'Tests',0:'p values'})
    ax3.table(
        cellText=ps.values,colLabels=ps.keys(),loc='center',colWidths=[0.4,0.4],colColours=['grey','grey'],cellLoc='left'
    )
    ax3.axis('off')

    strat_name = strat_name if strat_name is not None else ""
    fig.savefig(f".{path}/permuted_returns_{strat_name}.png")
    plt.close()

def plot_analysis(x,y):
    path = "/images"
    Path(os.path.abspath(os.getcwd()+path)).mkdir(parents=True,exist_ok=True)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    plt.scatter(x,y)
    plt.plot(x,c + m*A[:,0])
    plt.show()

    # plt.savefig(f".{path}/alpha_analysis.png")
    # plt.close()
