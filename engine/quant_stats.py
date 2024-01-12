import random
import asyncio
import numpy as np
import pandas as pd
import multiprocess as mp

from tqdm import tqdm
from scipy import stats
from copy import deepcopy
from collections import defaultdict
from pathos.multiprocessing import ProcessingPool


def shuffle_weights_on_eligs(weights_df, eligs_df, method="time", ord=2):
    assert method == "time" or method == "xs"
    if method == "time" and ord == 1:
        cs = []
        aligner = pd.DataFrame(weights_df.index)
        for wr, er in zip(weights_df.T.values, eligs_df.T.values):
            msk = np.where(er)[0]
            prm = permutation_member(wr[msk])
            nwr = np.zeros(len(wr))
            np.put(nwr, msk, prm)
            cs.append(pd.Series(nwr))
        nweight = pd.concat(cs, axis=1)
        nweight.columns = weights_df.columns
        nweight.index = weights_df.index
        nweight = nweight.div(nweight.sum(axis=1), axis=0).fillna(0.0)
        return nweight

    if method == "time" and ord == 2:
        idxs = list(range(len(list(weights_df))))
        tmp = weights_df.copy()
        tmp.columns = eligs_df.columns
        fil = tmp.div(eligs_df.astype(int))
        idxdts = {k: set(list(fil[col].dropna().index)) \
                  for k, col in zip(idxs, list(fil))}
        partitions, partition_idxs = partition_max_overlap(idxdts)
        tmp.columns = idxs
        chunked_ws = defaultdict(list)
        for partition, idx_list in zip(partitions, partition_idxs):
            permdf = perm_idx(
                tmp[idx_list].loc[partition]
            )
            for idx in idx_list:
                chunked_ws[idx].append(permdf[idx])

        aligner = pd.DataFrame(index=weights_df.index)
        idxws = []
        for idx in idx_list:
            iw = aligner.join(pd.concat(chunked_ws[idx], axis=0))
            idxws.append(iw)

        nweight = pd.concat(idxws, axis=1).fillna(0)
        nweight.columns = weights_df.columns
        nweight = nweight.div(nweight.sum(axis=1), axis=0).fillna(0.0)
        return nweight

    if method == "xs":
        rs = []
        for wr, er in zip(weights_df.values, eligs_df.values):
            msk = np.where(er)[0]
            prm = permutation_member(wr[msk])
            nwr = np.zeros(len(wr))
            np.put(nwr, msk, prm)
            rs.append(pd.Series(nwr))
        nweight = pd.concat(rs, axis=1).T
        nweight.columns = weights_df.columns
        nweight.index = weights_df.index
        return nweight


def generate_permutations(array):
    if len(array) <= 1:
        return [array]
    permutations = []
    for i in range(len(array)):
        element = array[i]
        sub_sequence = array[:i] + array[i + 1:]
        sub_permutations = generate_permutations(sub_sequence)
        for permutation in sub_permutations:
            permutations.append([element] + permutation)
    assert len(permutations) == np.math.factorial(len(array))
    return permutations

def permutation_member(array):
    i = len(array)
    np.random.seed()
    while i > 1:
        j = int(np.random.uniform(0, 1) * i)
        if j >= i: j = i - 1
        i -= 1
        temp = array[i]
        array[i] = array[j]
        array[j] = temp
    return array

def perm_idx(df):
    sidx = permutation_member(list(df.index))
    ndf = df.loc[sidx]
    ndf.index = df.index
    return ndf

# def permutation_shuffler_test(criterion_function, generator_function, m=1000, **kwargs):
#     unpermuted,_=criterion_function(**kwargs)
#     permuted_res=[]
#     permuted_caps=[]
#     for i in tqdm(range(m)):
#         permuted_args=generator_function(**kwargs)
#         permuted,permuted_cap=criterion_function(**permuted_args)
#         permuted_caps.append(permuted_cap)
#         permuted_res.append(permuted)
    
#     p = (1 + np.sum(permuted_res >= unpermuted)) / (len(permuted_res) + 1)
#     get_samples = np.random.choice(np.array(list(range(m))), size=np.min(np.array([m,17])), replace=False)
#     return pd.DataFrame(permuted_caps).iloc[get_samples].T, p, np.array(permuted_res)

def permute_in_new_event_loop(criterion_function, generator_function, size, kwargs):
    async def batch_shuffler(size):
        kwargss = await asyncio.gather(
            *[generator_function(**kwargs) for _ in range(size)]
        )
        criterions = [criterion_function(**kwargs)[0] for kwargs in kwargss]
        paths = [criterion_function(**kwargs)[1] for kwargs in kwargss]
        return criterions,paths

    return asyncio.run(batch_shuffler(size))

def split_rounds(m, max_splits=mp.cpu_count() * 4):
    batch_size = np.ceil(m / max_splits)
    batch_sizes = []
    while m > 0:
        batch_sizes.append(int(min(batch_size, m)))
        m -= batch_size
    return batch_sizes


def split_array(arr, max_splits=mp.cpu_count() * 4):
    split_sizes = split_rounds(m=len(arr), max_splits=max_splits)
    splits = []
    for size in split_sizes:
        splits.append(arr[:size])
        arr = arr[size:]
    return splits


def init_pools():
    random.seed()

async def permutation_shuffler_test(criterion_function, generator_function, m=1000, **kwargs):
    print(f"permutation test in progress for {generator_function} {m} times...")
    unpermuted,_ = criterion_function(**kwargs)
    batch_sizes = split_rounds(m=m)
    f = lambda args: permute_in_new_event_loop(*args)
    with ProcessingPool(initializer=init_pools) as pool:
        criterions = pool.map(f, [
            (criterion_function, generator_function, size, kwargs) for size in batch_sizes
        ])
    criterions_p,criterions_paths = [],[]
    for batched_criterion in criterions:
        criterions_p.extend(batched_criterion[0])
        criterions_paths.extend(batched_criterion[1])
    get_samples = np.random.choice(np.array(list(range(m))), size=17, replace=False)
    p = (1 + np.sum(criterions_p >= unpermuted)) / (len(criterions_p) + 1)
    return pd.DataFrame(criterions_paths).iloc[get_samples].T, p, np.array(criterions_p)

# =====================================================================
#
#
# =====================================================================

def one_sample_signed_rank_test(sample, m0, side="greater"):
    assert side == "greater" or side == "lesser"
    n=len(sample)
    ranks = stats.rankdata(np.abs(sample-m0))
    signs = np.sign(sample-m0)
    signed_ranks = ranks*signs
    W = np.sum(signed_ranks[signed_ranks>0])
    EW = n*(n+1)/4
    VARW = n*(n+1)*(2*n+1)/24
    Z = (W - EW)/np.sqrt(VARW)
    p = 1 - stats.norm.cdf(Z) if side == "greater" else stats.norm.cdf(Z)
    return p

def one_sample_sign_test(sample, m0, side="greater",norm_approx=True):
    assert side == "greater" or side == "lesser"
    n=len(sample)
    S = np.sum((sample - m0)>0) if side == "greater" else np.sum((sample-m0)<0)
    if norm_approx:
        if side == "greater":
            Z = (S - n * 0.5 - 0.5)/np.sqrt(n/4)
            p=1-stats.norm.cdf(Z)
            return p
        elif side == "lesser":
            Z = (S - n * 0.5 + 0.5)/np.sqrt(n/4)
            p=stats.norm.cdf(Z)
            return p
    return stats.binom_test(S,n=n,p=0.5, alternative=side)

def one_sample_t_test(sample, mu0, side="greater"):
    res = stats.ttest_1samp(sample,mu0,alternative=side)
    return res.pvalue, res.statistic

# =====================================================================
#
#
# =====================================================================

def marginal_family_test(unpermuted_criterions,criterion_function,
                        alpha_family,member_stats_generator=None,m=1000,alpha=0.05):
    from utils import Alpha
    if not member_stats_generator:
        assert all(isinstance(member,Alpha) for member in alpha_family)
        def member_stats(member):
            zfs = member.get_zero_filtered_stats()
            nweights = shuffle_weights_on_eligs(weights_df=zfs["weights"], eligs_df=zfs["eligs"], shuffle_type="time")
            nweights = shuffle_weights_on_eligs(weights_df=nweights, eligs_df=zfs["eligs"], shuffle_type="xs")
            return {
                "retdf": zfs["retdf"], "leverages": zfs["leverages"], 
                "weights": nweights, "eligs": zfs["eligs"]
            }
        member_stats_generator = member_stats
    unpermuted_criterions = np.array(unpermuted_criterions)
    round_criterions = []
    for round in range(m):
        round_criterions.append(
            [criterion_function(**member_stats_generator(member)) for member in alpha_family]
        )
    return stepdown_algorithm(
        unpermuted_criterions=unpermuted_criterions,
        round_criterions=np.array(round_criterions)
    )

def stepdown_algorithm(unpermuted_criterions,round_criterions,alpha=0.05):
    pvalues = np.array([None] * len(unpermuted_criterions))
    exact = np.array([False] * len(unpermuted_criterions))
    indices = np.array(list(range(len(unpermuted_criterions))))
    while not all(exact):
        stepwise_indices = indices[~exact]
        stepwise_criterions = np.array(unpermuted_criterions)[stepwise_indices]
        member_count = np.zeros(len(stepwise_criterions))
        for round in range(len(round_criterions)):
            round_max = np.max(np.array(round_criterions[round][stepwise_indices]))
            member_count += (0 + round_max >= stepwise_criterions)
        bounded_pvals = (1 + member_count) / (len(round_criterions) + 1)
        best_member = np.argmin(bounded_pvals)
        exact[stepwise_indices[best_member]] = True
        pvalues[stepwise_indices[best_member]] = np.min(bounded_pvals)
        if np.min(bounded_pvals) >= alpha:
            for bounded_p, index in zip(bounded_pvals, stepwise_indices):
                pvalues[index] = bounded_p
            break
    return pvalues, exact


def permute_price(price, permute_index=None):
    if not permute_index:
        permute_index = permutation_member(list(range(len(price) - 1)))
    log_prices = np.log(price)
    diff_logs = log_prices[1:] - log_prices[:-1]
    diff_perm = diff_logs[permute_index]
    cum_change = np.cumsum(diff_perm)
    new_log_prices = np.concatenate(([log_prices[0]], log_prices[0] + cum_change))
    new_prices = np.exp(new_log_prices)
    return new_prices


def permute_multi_prices(prices):
    assert all([len(price) == len(prices[0]) for price in prices])
    permute_index = permutation_member(list(range(len(prices[0]) - 1)))
    new_prices = [permute_price(price, permute_index=permute_index) for price in prices]
    return new_prices


def permute_bars(ohlcv, index_inter_bar=None, index_intra_bar=None):
    if len(ohlcv) <= 2:
        dropc = [col for col in ohlcv.columns if col not in ["open", "high", "low", "close", "volume"]]
        return deepcopy(ohlcv[["open", "high", "low", "close", "volume"]])
    if not index_inter_bar:
        index_inter_bar = permutation_member(list(range(len(ohlcv) - 1)))
    if not index_intra_bar:
        index_intra_bar = permutation_member(list(range(len(ohlcv) - 2)))
    log_data = np.log(ohlcv.astype("float32"))
    delta_h = log_data["high"].values - log_data["open"].values
    delta_l = log_data["low"].values - log_data["open"].values
    delta_c = log_data["close"].values - log_data["open"].values
    diff_deltas_h = np.concatenate((delta_h[1:-1][index_intra_bar], [delta_h[-1]]))
    diff_deltas_l = np.concatenate((delta_l[1:-1][index_intra_bar], [delta_l[-1]]))
    diff_deltas_c = np.concatenate((delta_c[1:-1][index_intra_bar], [delta_c[-1]]))

    new_volumes = np.concatenate(
        (
            [log_data["volume"].values[0]],
            log_data["volume"].values[1:-1][index_intra_bar],
            [log_data["volume"].values[-1]]
        )
    )

    inter_open_to_close = log_data["open"].values[1:] - log_data["close"].values[:-1]
    diff_inter_open_to_close = inter_open_to_close[index_inter_bar]

    new_opens, new_highs, new_lows, new_closes = \
        [log_data["open"].values[0]], \
            [log_data["high"].values[0]], \
            [log_data["low"].values[0]], \
            [log_data["close"].values[0]]

    last_close = new_closes[0]
    for i_delta_h, i_delta_l, i_delta_c, inter_otc in zip(
            diff_deltas_h, diff_deltas_l, diff_deltas_c, diff_inter_open_to_close
    ):
        new_open = last_close + inter_otc
        new_high = new_open + i_delta_h
        new_low = new_open + i_delta_l
        new_close = new_open + i_delta_c
        new_opens.append(new_open)
        new_highs.append(new_high)
        new_lows.append(new_low)
        new_closes.append(new_close)
        last_close = new_close

    new_df = pd.DataFrame(
        {
            "open": new_opens,
            "high": new_highs,
            "low": new_lows,
            "close": new_closes,
            "volume": new_volumes
        }
    )
    new_df = np.exp(new_df)
    new_df.index = ohlcv.index
    return new_df

def partition_max_overlap(idxidcs):
    '''takes dictionary of idx:set(indices) and returns partitions<>idxs'''
    idxpool=sorted(set().union(*idxidcs.values()))
   
    partitions, partition_idxs = [], []
    temp_partition = []
    temp_set = set([idx for idx,indcs_sets in idxidcs.items() \
        if idxpool[0] in indcs_sets])
    
    for i in idxpool:
        running_i = set()
        for idx, indcs_sets in idxidcs.items():
            if i in indcs_sets:
                running_i.add(idx)
        if running_i == temp_set:
            temp_partition.append(i)
        else:
            partitions.append(temp_partition)
            partition_idxs.append(list(temp_set))
            temp_partition = [i]
            temp_set = running_i
    partitions.append(temp_partition)
    partition_idxs.append(list(temp_set))
    return partitions, partition_idxs

def permute_multi_bars(bars):
    if all([len(bar) == len(bars[0]) for bar in bars]):
        index_inter_bar = permutation_member(list(range(len(bars[0]) - 1)))
        index_intra_bar = permutation_member(list(range(len(bars[0]) - 2)))
        new_bars = [
            permute_bars(
                bar,
                index_inter_bar=index_inter_bar,
                index_intra_bar=index_intra_bar
            )
            for bar in bars
        ]
    else:
        bar_indices = list(range(len(bars)))
        idxdts = {k: set(list(bar.index)) for k, bar in zip(bar_indices, bars)}
        partitions, partition_idxs = partition_max_overlap(idxdts)
        chunked_bars = defaultdict(list)
        for partition, idx_list in zip(partitions, partition_idxs):
            multibar = [bars[idx].loc[partition] for idx in idx_list]
            permuted_bars = permute_multi_bars(multibar)
            for idx, bar in zip(idx_list, permuted_bars):
                chunked_bars[idx].append(bar)

        new_bars = [None] * len(bars)
        for idx in bar_indices:
            new_bars[idx] = pd.concat(chunked_bars[idx], axis=0)
    return new_bars