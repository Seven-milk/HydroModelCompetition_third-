# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import os
from tqdm import *

home = 'H:/research/第三届水科学数值模拟'
pre_src_path = os.path.join(home, '初赛/数据预处理', '原始降水.xlsx')
pre_src = pd.read_excel(pre_src_path, index_col=False)
station_path = os.path.join(home, '初赛', '流域内雨量站.xlsx')
station = pd.read_excel(station_path, index_col=False)

# writer
writer = pd.ExcelWriter('修正降水.xlsx')
writer_one_hour = pd.ExcelWriter('修正降水1小时间隔.xlsx')

# loop for each station to pre-treat data
# (1) remove nan, negative value, make sum of DRP be equal to DYP
# (2) make 1 hour interval
for i in tqdm(range(len(station)), desc="loop for each station to pre-treat data", colour="green"):
    # create pre_station_df
    station_id = station.iloc[i, 1]
    pre_station_df = pre_src.loc[pre_src.STCD == station_id]
    pre_station_df.index = pd.to_datetime(pre_station_df.TM, format='%Y-%mm-%d %X')
    date = pre_station_df.index

    # search DYP that is not nan, not 0 -> get index
    DYP_array = pre_station_df.DYP.values
    DYP_not_nan = ~np.isnan(DYP_array)
    DYP_not_zero = ~(DYP_array == 0)
    DYP_not_nan_zero = [DYP_not_nan[j] and DYP_not_zero[j] for j in range(len(DYP_not_nan))]
    DYP_not_nan_zero_index = np.where(DYP_not_nan_zero)[0]

    # loop for each DYP (not nan, not 0) to compare with sum of DRP
    for k in DYP_not_nan_zero_index:
        # DYP
        DYP = pre_station_df.DYP.iloc[k]

        # DYP -> search slice
        date_now = date[k]
        date_before_one_day = date_now - DateOffset(days=1)

        start_index = np.where(date >= date_before_one_day)[0][0]
        end_index = k

        # slice: pass address
        DRP_slice = pre_station_df.DRP.iloc[start_index: end_index + 1]

        # DRP: nan -> 0, -x -> 0
        nan_ = np.isnan(DRP_slice)
        negative_ = DRP_slice < 0
        DRP_slice[np.isnan(DRP_slice)] = 0
        DRP_slice[DRP_slice < 0] = 0

        # compare sum(DRP) and DYP
        diff = DYP - np.nansum(DRP_slice)

        if diff > 0:
            DRP_slice += diff / len(DRP_slice)
        elif diff < 0:
            # ignore values on negative -> zero
            DRP_slice[~negative_] += diff / (len(DRP_slice) - sum(negative_))
        else:
            pass

    # All DRP (not only for corresponding slice of DYP): nan -> 0, -x -> 0
    DRP = pre_station_df.DRP[:]
    DRP.loc[np.isnan(DRP)] = 0
    DRP[DRP < 0] = 0

    # make 1 hour interval
    date_one_hour_interval = pd.date_range(start=date[0], end=date[-1], freq="H")
    pre_station_df_one_hour_interval = pre_station_df.reindex(date_one_hour_interval, columns=['STCD'],
                                                              fill_value=station_id)
    pre_station_df_one_hour_interval["TM"] = date_one_hour_interval
    pre_station_df_one_hour_interval["DRP"] = pre_station_df.reindex(date_one_hour_interval, columns=['DRP'],
                                                                     fill_value=0)
    pre_station_df_one_hour_interval["INTV"] = pre_station_df.reindex(date_one_hour_interval, columns=['INTV'],
                                                                      fill_value=np.nan)
    pre_station_df_one_hour_interval["PDR"] = pre_station_df.reindex(date_one_hour_interval, columns=['PDR'],
                                                                     fill_value=np.nan)
    pre_station_df_one_hour_interval["DYP"] = pre_station_df.reindex(date_one_hour_interval, columns=['DYP'],
                                                                     fill_value=np.nan)

    # save
    pre_station_df.to_excel(writer, sheet_name=str(station_id), index=False)
    pre_station_df_one_hour_interval.to_excel(writer_one_hour, sheet_name=str(station_id), index=False)

writer.save()
writer.close()
writer_one_hour.save()
writer_one_hour.close()
