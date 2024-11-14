import random
import numpy as np
import pandas as pd
import multiprocess as mp

from typing import Callable, Union
from collections import defaultdict
from pathos.multiprocessing import ProcessingPool


def partition_max_overlap(idxidcs: dict[int, set[pd.Index]]) -> tuple[list, list]:
    """_summary_
        takes dictionary of idx:set(indices) and returns partitions<>idxs'
    Args:
        idxidcs (dict[int, set[pd.Index]]): _description_

    Returns:
        (tuple[list, list]): _description_
    """
    idxpool = sorted(set().union(*idxidcs.values()))
   
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

def shuffle_weights_on_eligs(
        weights_df: pd.DataFrame, 
        eligs_df: pd.DataFrame, 
        method: str = "time", 
        ord: int = 2
    ) -> pd.DataFrame:
    """_summary_

    Args:
        weights_df (pd.DataFrame): _description_
        eligs_df (pd.DataFrame): _description_
        method (str, optional): _description_. Defaults to "time".
        ord (int, optional): _description_. Defaults to 2.

    Returns:
        pd.DataFrame: _description_
    """
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
        idxdts = {
            k: set(list(fil[col].dropna().index)) for k, col in zip(idxs, list(fil))
        }
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

def permutation_member(array: np.ndarray | list[pd.DatetimeIndex]) -> np.ndarray:
    """_summary_

    Args:
        array (np.ndarray | list[pd.DatetimeIndex]): _description_

    Returns:
        np.ndarray: _description_
    """
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

def perm_idx(df: pd.DataFrame) -> pd.DataFrame:
    sidx = permutation_member(list(df.index))
    ndf = df.loc[sidx]
    ndf.index = df.index
    return ndf

def permute_in_new_event_loop(
        criterion_function: Callable[[pd.Series, pd.Series, pd.Series], tuple[float, pd.Series]], 
        generator_function: Callable[[pd.Series, pd.Series, pd.Series], dict[str, pd.DataFrame]], 
        size: int, 
        kwargs: pd.DataFrame
    ) -> tuple[list[float], list[pd.Series]]:
    criterions = []
    paths = []
    for _ in range(size):
        gen_kwargs = generator_function(**kwargs)
        criterion, path = criterion_function(**gen_kwargs)
        criterions.append(criterion)
        paths.append(path)
    return criterions, paths

def split_rounds(m: int, max_splits: float = mp.cpu_count() * 4) -> list[int]:
    batch_size = np.ceil(m / max_splits)
    batch_sizes = []
    while m > 0:
        batch_sizes.append(int(min(batch_size, m)))
        m -= batch_size
    return batch_sizes


def init_pools():
    random.seed()

def permutation_shuffler_test(
        criterion_function: Callable[[pd.Series, pd.Series, pd.Series], tuple[float, pd.Series]], 
        generator_function: Callable[[pd.Series, pd.Series, pd.Series], dict[str, pd.DataFrame]], 
        m: int = 1000, 
        **kwargs: pd.DataFrame
    ) -> tuple[pd.DataFrame, float, np.ndarray]:
    """_summary_

    Args:
        criterion_function (Callable[[pd.Series, pd.Series, pd.Series], tuple[float, pd.Series]]): _description_
        generator_function (Callable[[pd.Series, pd.Series, pd.Series], dict[str, pd.DataFrame]]): _description_
        m (int, optional): _description_. Defaults to 1000.

    Returns:
        tuple[pd.DataFrame, float, np.ndarray]: _description_
    """
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
    paths = pd.DataFrame(criterions_paths).iloc[get_samples].T
    p_val = (1 + np.sum(criterions_p >= unpermuted)) / (len(criterions_p) + 1)
    criterions_p = np.array(criterions_p)
    return paths, p_val, criterions_p

def hypothesis_tests( 
        num_decision_shuffles: int = 1000,
        zfs: dict[str, Union[pd.Series, pd.DataFrame]] | None = None
    ) -> dict[str, tuple[list[float], float, list[float]]]:
    """
    Perform hypothesis tests on trading strategies using shuffled weights.

    Parameters
    ----------
    num_decision_shuffles : int, optional
        Number of shuffles for hypothesis testing, by default 1000.
    zfs : dict[str, Union[pd.Series, pd.DataFrame]] | None, optional
        Zero-filtered statistics containing returns, leverages, weights, and eligibilities,
        by default None.

    Returns
    -------
    dict[str, tuple[list[float], float, list[float]]]
        A dictionary containing test results for different shuffling methods.
        Each key (e.g., "timer", "picker", "trader") maps to a tuple of:
        - Paths (list of performance values),
        - P-value from permutation test,
        - Distribution of shuffled performance values.
    """
    assert zfs is not None, "Zero-filtered statistics are required for hypothesis testing."
    retdf, leverages, weights, eligs = zfs["retdf"], zfs["leverages"], zfs["weights"], zfs["eligs"]

    def performance_criterion(rets, leverages, weights, **kwargs) -> tuple[float, list[float]]:
        """Calculate the Sharpe ratio as the performance criterion."""
        capital_ret = [
            lev_scalar * np.dot(weight, ret)
            for lev_scalar, weight, ret in zip(leverages.values, rets.values, weights.values)
        ]
        sharpe = np.mean(capital_ret) / np.std(capital_ret) * np.sqrt(253)
        return round(sharpe, 5), capital_ret

    def time_shuffler(rets, leverages, weights, eligs) -> dict[str, pd.DataFrame]:
        """Shuffle weights based on time-based eligibility."""
        nweights = shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, method="time", ord=1)
        return {"rets": rets, "leverages": leverages, "weights": nweights}

    def mapping_shuffler(rets, leverages, weights, eligs) -> dict[str, pd.DataFrame]:
        """Shuffle weights based on cross-sectional eligibility."""
        nweights = shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, method="xs")
        return {"rets": rets, "leverages": leverages, "weights": nweights}

    def decision_shuffler(rets, leverages, weights, eligs) -> dict[str, pd.DataFrame]:
        """Apply both time-based and cross-sectional weight shuffling."""
        nweights = shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, method="time", ord=1)
        nweights = shuffle_weights_on_eligs(weights_df=nweights, eligs_df=eligs, method="xs")
        return {"rets": rets, "leverages": leverages, "weights": nweights}

    # Perform hypothesis tests with different shuffling methods
    timer_paths, timer_p, timer_dist = permutation_shuffler_test(
        criterion_function=performance_criterion,
        generator_function=time_shuffler,
        m=num_decision_shuffles, rets=retdf, leverages=leverages, weights=weights, eligs=eligs
    )

    picker_paths, picker_p, picker_dist = permutation_shuffler_test(
        criterion_function=performance_criterion,
        generator_function=mapping_shuffler,
        m=num_decision_shuffles, rets=retdf, leverages=leverages, weights=weights, eligs=eligs
    )

    trader_paths, trader_p, trader_dist = permutation_shuffler_test(
        criterion_function=performance_criterion,
        generator_function=decision_shuffler,
        m=num_decision_shuffles, rets=retdf, leverages=leverages, weights=weights, eligs=eligs
    )

    return {
        "timer": (timer_paths, timer_p, timer_dist),
        "picker": (picker_paths, picker_p, picker_dist),
        "trader": (trader_paths, trader_p, trader_dist),
    }