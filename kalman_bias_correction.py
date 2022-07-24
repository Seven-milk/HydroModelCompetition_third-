# code: utf-8
# author: "Xudong Zheng"
# email: z786909151@163.com
import os
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg


# read data
def readdata(home):
    data_folds = [fold for fold in os.listdir(home) if os.path.isdir(os.path.join(home, fold))]
    df_list = []
    for fold in data_folds:
        path = os.path.join(home, fold)
        runoff_path = os.path.join(path, f"{fold}.txt")
        pcp_path = os.path.join(path, "pcp.txt")

        runoff = pd.read_csv(runoff_path)
        pcp = pd.read_csv(pcp_path, sep="\t", header=None)
        runoff_index = pd.to_datetime(runoff.iloc[:, 0], format="%Y-%m-%d %X")
        pcp_index = pd.to_datetime(pcp.iloc[:, 1], format="%Y-%m-%d %X")
        runoff.index = runoff_index
        pcp.index = pcp_index

        df = pd.DataFrame(index=runoff_index)
        df = df.join(runoff.iloc[:, 1:])
        df = df.join(pcp.iloc[:, -1])
        df.rename(columns={"实测流量": "runoff_o", "模拟流量": "runoff_p", 2: "pcp"}, inplace=True)
        df["bias"] = df.runoff_o - df.runoff_p
        df_list.append(df)

    return df_list


# create sample and verify dataset
def create_sample_verify_dataset(df_list):
    df_sample_list = df_list[1:4]
    df_sample_list.extend(df_list[5:])
    df_verify_list = [df_list[0], df_list[4]]

    sample_runoff_p_t = []
    sample_runoff_p_t_next = []

    sample_runoff_o_t = []
    sample_runoff_o_t_next = []

    sample_bias_t = []
    sample_bias_t_next = []

    sample_df = pd.concat(df_sample_list)

    for df in df_sample_list:
        sample_runoff_p_t.extend(df.runoff_p.values[:-1].tolist())
        sample_runoff_p_t_next.extend(df.runoff_p.values[1:].tolist())
        sample_runoff_o_t.extend(df.runoff_o.values[:-1].tolist())
        sample_runoff_o_t_next.extend(df.runoff_o.values[1:].tolist())
        sample_bias_t.extend(df.bias.values[1:].tolist())
        sample_bias_t_next.extend(df.bias.values[:-1].tolist())

    verify_runoff_p_t = []
    verify_runoff_p_t_next = []

    verify_runoff_o_t = []
    verify_runoff_o_t_next = []

    verify_bias_t = []
    verify_bias_t_next = []

    verify_df = pd.concat(df_verify_list)

    for df in df_verify_list:
        verify_runoff_p_t.extend(df.runoff_p.values[:-1].tolist())
        verify_runoff_p_t_next.extend(df.runoff_p.values[1:].tolist())
        verify_runoff_o_t.extend(df.runoff_o.values[:-1].tolist())
        verify_runoff_o_t_next.extend(df.runoff_o.values[1:].tolist())
        verify_bias_t.extend(df.bias.values[1:].tolist())
        verify_bias_t_next.extend(df.bias.values[:-1].tolist())

    # return sample_runoff_p_t, sample_runoff_p_t_next,\
    #        sample_runoff_o_t, sample_runoff_o_t_next,\
    #        sample_bias_t, sample_bias_t_next,\
    #        verify_runoff_p_t, verify_runoff_p_t_next,\
    #        verify_runoff_o_t, verify_runoff_o_t_next,\
    #        verify_bias_t, verify_bias_t_next,\
    #        sample_df, verify_df
    return sample_df, verify_df


# AR model
def ARmodel(df, lags=2):
    obj = df.runoff_o
    obj = obj.asfreq("1H", fill_value=0)
    mod = AutoReg(obj, lags)
    res = mod.fit()

    predict = res.predict()
    real = obj
    Q = np.nanvar(predict - real)

    print(res.summary())
    print(res.params)
    return mod, res, Q


# correction function
def Kalman_correction(df, F, H, I, Q, R, lags, plot=True, save_on=False):
    # state runoff_o
    # observe runoff_p
    # optional input pt_1(pre)
    # runoff_o_t_ = np.dot(F, runoff_o_t_1)  # + B * pt_1

    # df
    runoff_o = df.runoff_o.values
    runoff_p = df.runoff_p.values
    bias = df.bias.values

    # init, t = 0
    Pt_0 = np.eye(lags + 1)
    x0 = np.array([[runoff_o[1]], [runoff_o[0]], [1]])  # lags = 2
    # B = 0.1

    def kalman(xt_1, Pt_1, zt):
        xt_ = np.dot(F, xt_1)
        Pt_ = np.dot(np.dot(F, Pt_1), F.T) + Q
        Kt = np.dot(Pt_, H.T) / (np.dot(np.dot(H, Pt_), H.T) + R)
        xt = xt_ + Kt * (zt - np.dot(H, xt_))
        Pt = np.dot((I - np.dot(Kt, H)), Pt_)

        return xt, Pt

    # loop for filter
    corrected_runoff_p = []
    corrected_runoff_p.append(runoff_o[0])
    Pt_1 = Pt_0
    xt_1 = x0
    for i in range(1, len(runoff_p)):
        xt, Pt = kalman(xt_1, Pt_1, runoff_p[i])
        Pt_1 = Pt
        xt_1 = xt
        runoff_o_t = float(np.dot(H, xt))
        corrected_runoff_p.append(runoff_o_t)

    corrected_runoff_p = np.array(corrected_runoff_p)

    # plot
    h = plt.figure()
    plt.plot(runoff_o, "k-", label="observation")
    plt.plot(runoff_p, "b--", label="prediction")
    plt.plot(corrected_runoff_p, "r--", label="corrected")
    plt.xlabel = "Date"
    plt.ylabel = "Q m/s"
    plt.legend()

    if plot:
        plt.show()  # block=True

    df["runoff_p_corrected"] = corrected_runoff_p

    if save_on:
        df.to_csv(f"{save_on}.csv")
        h.savefig(f"{save_on}.tiff")

    return df


def verify_flood(df):
    flood_volume_diff_abs_before = df.runoff_p.sum() - df.runoff_o.sum()
    flood_volume_diff_abs_after = df.runoff_p_corrected.sum() - df.runoff_o.sum()
    flood_volume_diff_relative_before = (df.runoff_p.sum() - df.runoff_o.sum()) / df.runoff_o.sum()
    flood_volume_diff_relative_after = (df.runoff_p_corrected.sum() - df.runoff_o.sum()) / df.runoff_o.sum()

    NSE_before = (sum((df.runoff_p - df.runoff_o.mean()) ** 2) - sum((df.runoff_p - df.runoff_o) ** 2)) / sum(
        (df.runoff_o - df.runoff_o.mean()) ** 2)
    NSE_after = (sum((df.runoff_p_corrected - df.runoff_o.mean()) ** 2) - sum((df.runoff_p_corrected - df.runoff_o) ** 2)) / sum(
        (df.runoff_o - df.runoff_o.mean()) ** 2)

    peak_time_diff_before = np.argmax(df.runoff_p.values) - np.argmax(df.runoff_o.values)
    peak_time_diff_after = np.argmax(df.runoff_p_corrected.values) - np.argmax(df.runoff_o.values)

    peak_diff_abs_before = df.runoff_p.values.max() - df.runoff_o.values.max()
    peak_diff_abs_after = df.runoff_p_corrected.values.max() - df.runoff_o.values.max()
    peak_diff_relative_before = (df.runoff_p.values.max() - df.runoff_o.values.max()) / df.runoff_o.values.max()
    peak_diff_relative_after = (df.runoff_p_corrected.values.max() - df.runoff_o.values.max()) / df.runoff_o.values.max()

    df_out = pd.DataFrame({"flood_volume_diff_abs": [flood_volume_diff_abs_before, flood_volume_diff_abs_after],
                           "flood_volume_diff_relative": [flood_volume_diff_relative_before, flood_volume_diff_relative_after],
                           "peak_diff_abs": [peak_diff_abs_before, peak_diff_abs_after],
                           "peak_diff_relative": [peak_diff_relative_before, peak_diff_relative_after],
                           "NSE": [NSE_before, NSE_after],
                           "peak_time_diff": [peak_time_diff_before, peak_time_diff_after]},
                          index=["before", "after"])
    return df_out


if __name__ == "__main__":
    save = True
    home = "F:/research/The third numerical Simulation of water Science/Intermediary_heat/data"
    df_list = readdata(home)
    sample_df, verify_df = create_sample_verify_dataset(df_list)
    lags = 2
    mod, res, Q = ARmodel(sample_df, lags=lags)
    Q = Q * np.eye(lags + 1)
    print("Q\t", Q)
    R = np.var(sample_df.bias.values)
    print("R\t", R)

    F = np.array([[res.params[1], res.params[2], res.params[0]],
                  [1, 0, 0],
                  [0, 0, 1]])  # lags=2
    H = np.array([[1, 0, 0]])
    I = np.eye(lags + 1)
    for i in range(len(df_list)):
        df_ = df_list[i]
        if save:
            save_on = os.path.join('F:/research/The third numerical Simulation of water Science/Intermediary_heat', 'kalman_corrected_ret', str(i))
        else:
            save_on = False
        Kalman_correction(df_, F, H, I, Q, R, lags, plot=False, save_on=f"{save_on}")

    # verify
    df_0 = Kalman_correction(df_list[0], F, H, I, Q, R, lags, plot=False)
    df_4 = Kalman_correction(df_list[4], F, H, I, Q, R, lags, plot=False)
    verify_df_out0 = verify_flood(df_0)
    verify_df_out4 = verify_flood(df_4)
    if save:
        verify_df_out0.to_csv(os.path.join('F:/research/The third numerical Simulation of water Science/Intermediary_heat',
                                       'kalman_corrected_ret', "verify_0.csv"))
        verify_df_out4.to_csv(os.path.join('F:/research/The third numerical Simulation of water Science/Intermediary_heat',
                                       'kalman_corrected_ret', "verify_4.csv"))

