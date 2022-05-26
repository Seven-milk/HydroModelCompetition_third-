# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
import os
from tqdm import *
import matplotlib.pyplot as plt


home = 'H:/research/The third numerical Simulation of water Science'
runoff_src_path = os.path.join(home, 'Preliminary/Preteat_data', '综合3站-20220524.xlsx')
runoff_src_df = pd.read_excel(runoff_src_path, index_col=False, sheet_name=0)
runoff_src_df.index = pd.to_datetime(runoff_src_df.TM, format='%Y-%mm-%d %X')
print('max runoff:', runoff_src_df.Q.max())
save_on = True


# run theory
def run(index: np.ndarray, threshold: float) -> (np.ndarray, np.ndarray):
    """ Implements the RuntheoryBase.run function
    run_threshold to identify develop period (index > threshold, different with run_threshold)
    point explain(discrete): start > threshold, end > threshold --> it is shrinkable and strict

    input:
        index: 1D np.ndarray, fundamental index
        threshold: float, the threshold to identify dry bell(index >= threshold)

    output:
        dry_flag_start/end: 1D np.ndarray, the array contain start/end indexes of Exceed events
    """
    # define develop period based on index and threshold
    dry_flag = np.argwhere(index > threshold).flatten()
    dry_flag_start = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()
    dry_flag_end = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()

    return dry_flag_start, dry_flag_end


# Hyperquantitative sample mean method
def Determinate_S():
    # general set
    Q = runoff_src_df.Q.values
    S_list = list(range(10, int(Q.max()), 10))
    year = runoff_src_df.index.year
    F_number = []
    F_exceed = []
    I = []
    F_year_number_mean = []

    # all peaks
    flag_peaks_all = np.array([i for i in range(1, len(Q) - 1) if Q[i] > Q[i-1] and Q[i] > Q[i+1]])
    peaks_all = np.array([Q[i] for i in flag_peaks_all])

    for S in S_list:
        # select peaks by S
        flag_peaks_S = flag_peaks_all[peaks_all > S]
        peaks_S = np.array([Q[i] for i in flag_peaks_S])

        # cal indicator
        F_number.append(len(flag_peaks_S))
        F_exceed.append((peaks_S - S).mean())

        F_year_number_dict = dict.fromkeys(set(year), 0)
        for j in range(len(flag_peaks_S)):
            Flood_year = year[flag_peaks_S[j]]
            F_year_number_dict[Flood_year] += 1

        F_year_number_list = list(F_year_number_dict.values())
        F_year_number_list_mean = sum(F_year_number_list) / len(F_year_number_list)
        F_year_number_mean.append(F_year_number_list_mean)
        I.append(np.var(F_year_number_list) / F_year_number_list_mean)

    # plot 1
    fig, ax = plt.subplots()
    ax_twinx = ax.twinx()
    font = {'family': 'Microsoft YaHei'}
    ax.plot(S_list, F_year_number_mean, 'r', label='年均超定量个数')
    ax_twinx.plot(S_list, F_exceed, 'b', label='样本超越门限值部分均值')
    ax.set_xlabel('门限值S ($m^3/s$)', fontdict=font)
    ax.set_ylabel('年均超定量个数', fontdict=font)
    ax_twinx.set_ylabel('样本超越门限值部分均值', fontdict=font)
    plt.xlim(S_list[0], S_list[-1])
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_twinx.get_legend_handles_labels()
    plt.legend(handles1+handles2, labels1+labels2, prop=font, loc='upper right')

    if save_on:
        plt.savefig('门限值确定1.jpg')

    # plot 2
    upper = 12.59 / 5  # n is 6
    lower = 1.64 / 5
    fig_I, ax_I = plt.subplots()
    ax_I.plot(S_list, I, 'k', label='分散指数 I')
    plt.fill([S_list[0], S_list[0], S_list[-1], S_list[-1]], [lower, upper, upper, lower], color='y', alpha=0.3, label='[5%, 95%]置信区间')
    plt.xlim(S_list[0], S_list[-1])
    ax_I.set_xlabel('门限值S ($m^3/s$)', fontdict=font)
    ax_I.set_ylabel('分散指数 I', fontdict=font)
    plt.legend(prop=font)

    if save_on:
        plt.savefig('门限值确定2.jpg')

    # df
    df_out = pd.DataFrame(np.vstack((np.array(S_list), np.array(F_year_number_mean), np.array(F_exceed), np.array(I))),
                 index=['S', '年均超定量个数', '样本超越门限值部分均值', 'I'])

    if save_on:
        df_out.to_excel('门限值确定.xlsx')


# Determinate_S()

# S
S = 1300


def extract_flood_peak():
    # define flood based on selected S = 1300
    Q = runoff_src_df.Q.values
    flag_peaks_all = np.array([i for i in range(1, len(Q) - 1) if Q[i] > Q[i-1] and Q[i] > Q[i+1]])
    peaks_all = np.array([Q[i] for i in flag_peaks_all])
    flag_peaks_S = flag_peaks_all[peaks_all > S]

    # pooling and excluding
    A = 2000
    c1_threshold = float(5 + np.log(A))
    i = 0

    while i < len(flag_peaks_S) - 1:
        flag = flag_peaks_S[i]
        flag_next = flag_peaks_S[i + 1]
        interval_duration = runoff_src_df.index[flag_next] - runoff_src_df.index[flag]
        interval_duration_float = interval_duration.total_seconds() / 3600
        interval_min = min(runoff_src_df.Q[flag: flag_next + 1])

        c_1 = c1_threshold < interval_duration_float
        c_2 = interval_min < 0.75 * min(Q[flag], Q[flag_next])

        if c_1 and c_2:
            pass
        else:
            flag_peaks_S = flag_peaks_S.astype(list)
            flag_peaks_S[i] = [flag_peaks_S[i], flag_peaks_S[i + 1]]
            flag_peaks_S = np.delete(flag_peaks_S, i + 1)

        i += 1

    if save_on:
        f = open('洪峰提取.txt', 'a')
    else:
        f = None

    print('All flood:', len(flag_peaks_S), file=f)
    print('----------------------------', file=f)
    print('index as follow: ', file=f)
    for i in flag_peaks_S:
        print(i, file=f)
    print('----------------------------', file=f)
    print('date as follow: ', file=f)
    for i in flag_peaks_S:
        print(runoff_src_df.index[i], file=f)

    if save_on:
        f.close()

    return flag_peaks_S


flag_peaks_S = extract_flood_peak()


def cal_base_flow():
    if save_on:
        f = open('基流计算.txt', 'a')
    else:
        f = None

    month = runoff_src_df.index.month

    # define dry season
    group_month = runoff_src_df.groupby(runoff_src_df.index.month)["Q"].mean()
    print(group_month)
    plt.bar(group_month.index, group_month.values)
    plt.xlabel('month')
    plt.ylabel('Mean Q')

    if save_on:
        plt.savefig('枯水季确定.jpg')

    dry_season = [11, 12, 1, 2]
    print('dry_season', dry_season, file=f)

    # extract dry season Q
    dry_season_bool = [m in dry_season for m in month]
    dry_season_Q = runoff_src_df.Q.values[dry_season_bool]

    # cal base_flow
    base_flow = dry_season_Q.mean()
    print('base_flow: ', base_flow, file=f)

    if save_on:
        f.close()

    return base_flow


base_flow = cal_base_flow()


def extract_flood(flag_peaks_S):
    if save_on:
        f = open('洪水提取.txt', 'a')
    else:
        f = None

    Q = runoff_src_df.Q.values
    Q_index = np.arange(len(Q))
    flag_start_base, flag_end_base = run(Q, base_flow)

    def left(index, search_array):
        return search_array[np.where(index > search_array)[0][-1]]

    def right(index, search_array):
        return search_array[np.where(index < search_array)[0][0]]

    def lowest_index(search_array):
        return np.argmin(search_array)

    flag_peak_base_left_all = []
    flag_peak_base_right_all = []

    # loop for searching start and end index in flag_start_base/flag_end_base for each peak
    for i in range(len(flag_peaks_S)):
        flag_peak = flag_peaks_S[i]

        # start and end
        if isinstance(flag_peak, int):
            flag_peak_start = flag_peak
            flag_peak_end = flag_peak

        elif isinstance(flag_peak, list):
            flag_peak_start = flag_peak[0]
            flag_peak_end = flag_peak[-1]

        # search start left and end right for peaks in start/end of baseflow
        flag_peak_base_left_all.append(left(flag_peak_start, flag_start_base))
        flag_peak_base_right_all.append(right(flag_peak_end, flag_end_base))

    # loop for modifying start and end index by lowest point for each peak
    for i in range(1, len(flag_peak_base_left_all)):
        flag_peak = flag_peaks_S[i]
        flag_peak_before = flag_peaks_S[i - 1]

        # start and end
        if isinstance(flag_peak, int):
            flag_peak_start = flag_peak
            flag_peak_end = flag_peak

        elif isinstance(flag_peak, list):
            flag_peak_start = flag_peak[0]
            flag_peak_end = flag_peak[-1]

        # start and end
        if isinstance(flag_peak_before, int):
            flag_peak_before_start = flag_peak_before
            flag_peak_before_end = flag_peak_before

        elif isinstance(flag_peak_before, list):
            flag_peak_before_start = flag_peak_before[0]
            flag_peak_before_end = flag_peak_before[-1]

        flag_peak_base_left = flag_peak_base_left_all[i]
        flag_peak_base_right = flag_peak_base_right_all[i]

        flag_peak_before_base_left = flag_peak_base_left_all[i - 1]
        flag_peak_before_base_right = flag_peak_base_right_all[i - 1]

        flag_peak_before_start_all = flag_peak_base_left_all[:i]
        flag_peak_before_end_all = flag_peak_base_right_all[:i]

        if flag_peak_base_left in flag_peak_before_start_all:
            # search lowest point between i - 1 and i
            Q_between = Q[flag_peak_before_start: flag_peak_start + 1]
            Q_index_between = Q_index[flag_peak_before_start: flag_peak_start + 1]
            flag_lowest = Q_index_between[lowest_index(Q_between)]

            # modify as lowest point
            flag_peak_base_left_all[i] = flag_lowest  # modify as lowest point
            flag_peak_base_right_all[i - 1] = flag_lowest  # modify as lowest point


    flag_peak_base_left_all = np.array(flag_peak_base_left_all)
    flag_peak_base_right_all = np.array(flag_peak_base_right_all)
    flag_peaks_S = np.array(flag_peaks_S)

    print('All flood:', len(flag_peaks_S), file=f)
    print('----------------------------', file=f)
    print('index as follow: ', file=f)
    for i in range(len(flag_peaks_S)):
        print(flag_peak_base_left_all[i], '-', flag_peak_base_right_all[i], file=f)
    print('----------------------------', file=f)
    print('date as follow: ', file=f)
    for i in range(len(flag_peaks_S)):
        print(runoff_src_df.index[flag_peak_base_left_all[i]], '-', runoff_src_df.index[flag_peak_base_right_all[i]], file=f)

    # plot
    font = {'family': 'Microsoft YaHei'}
    plt.plot(runoff_src_df.index, runoff_src_df.Q, 'k', label='径流 $m^3/s$')
    for i in range(len(flag_peak_base_left_all)):
        label_left = '起涨点' if i == 0 else None
        label_right = '终止点' if i == 0 else None
        plt.vlines(x=runoff_src_df.index[flag_peak_base_left_all[i]], ymin=0, ymax=max(runoff_src_df.Q) * 1.1,
                   colors='r', linestyles='--', linewidth=1, label=label_left, alpha=0.5)
        plt.vlines(x=runoff_src_df.index[flag_peak_base_right_all[i]] + pd.Timedelta('1D'), ymin=0,
                   ymax=max(runoff_src_df.Q) * 1.1,
                   colors='b', linestyles='--', linewidth=1, label=label_right, alpha=0.5)
    plt.xlim(runoff_src_df.index[0], runoff_src_df.index[-1])
    plt.ylim(0, max(runoff_src_df.Q) * 1.1)
    plt.xlabel('date')
    plt.ylabel('Q $m^3/s$')
    plt.legend(prop=font)

    if save_on:
        plt.savefig('洪水提取.jpg')
        f.close()


extract_flood(flag_peaks_S)