import numpy as np
import pandas as pd

from adns.simulators.wireless_contest_simulator import WirelessContestSimulator
import json
import sys
import copy


def neaten_data(cell_state):
    """整理出各类关键数据"""
    n_cell = cell_state.shape[0]
    cell_MRcount = np.zeros((n_cell, 4, 8))
    cell_SSB_RSRP = np.zeros((n_cell, 4, 8))
    cell_dl_thrp = np.zeros((n_cell, 4, 8))
    cell_prb = np.zeros((n_cell, 4, 8))
    cell_CSI_RSRP = np.zeros((n_cell, 4, 8))
    for i in range(0, n_cell):
        for j in range(0, 4):
            for k in range(0, 8):
                cell_MRcount[i, j, k] = cell_state.iloc[i, 10 + j * 8 + k]
                cell_SSB_RSRP[i, j, k] = cell_state.iloc[i, 42 + j * 8 + k]
                cell_dl_thrp[i, j, k] = cell_state.iloc[i, 74 + j * 8 + k]
                cell_prb[i, j, k] = cell_state.iloc[i, 106 + j * 8 + k]
                cell_CSI_RSRP[i, j, k] = cell_state.iloc[i, 138 + j * 8 + k]
                if cell_MRcount[i, j, k] == 0:
                    cell_prb[i, j, k] = 1
    return cell_MRcount, cell_SSB_RSRP, cell_dl_thrp, cell_prb, cell_CSI_RSRP


def cal_score(cur_cv, cur_sp, cur_eg):
    base_cv = 0.03079317
    base_sp = 130.3422004
    base_eg = 158.1873666
    score = (cur_sp / base_sp + 2 - cur_eg / base_eg) * 1000
    if (base_sp - cur_sp) / base_sp > 0.05:
        score = score * 0.9
    if (cur_cv - base_cv) / base_cv > 0.05:
        score = score * 0.9
    return score


def entrance(data_enter):
    data_scenario = data_enter["data_scenario"]
    data_enter["data_id"] = data_enter["data_scenario"]
    wireless_contest_simulator = WirelessContestSimulator(**data_enter)
    max_iter_num = 200

    with open("./input_with_constr_13.json", "r") as f:
        information_dict = json.load(f)
    ori_cell_state = pd.read_csv("./cell_report.csv", encoding='gbk')
    ori_param = information_dict["opti_param"]
    param = copy.deepcopy(ori_param)

    allocate(ori_cell_state, param, 0, CSI_data_option=0, cell_id_list=range(100), repeat_ratio=0.05)

    cell_state, kpi = wireless_contest_simulator.run(param)
    best_score = cal_score(kpi.res_cluster_weak_coverage[0], kpi.res_cluster_speed[0], kpi.res_cluster_energy[0])
    # best_score, cell_state = my_optimizer_pwr(best_score, cell_state, param, wireless_contest_simulator,
    #                                           cell_id_list=range(100), ori_pwr=-3, try_pwr=0)
    # cell_state, kpi = wireless_contest_simulator.run(param)
    # best_score = cal_score(kpi.res_cluster_weak_coverage[0], kpi.res_cluster_speed[0], kpi.res_cluster_energy[0])

    predict_cell_speed = copy.deepcopy(cell_state.predict_cell_speed)
    cell_id_list = np.argsort(predict_cell_speed)

    # idx = copy.deepcopy(cell_id_list[60: 100])
    # for repeat_ratio in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     ori_param1 = copy.deepcopy(param)
    #     ori_state1 = copy.deepcopy(cell_state)
    #     allocate(ori_cell_state, param, 0, CSI_data_option=0, cell_id_list=idx, repeat_ratio=repeat_ratio)
    #     cell_state, kpi = wireless_contest_simulator.run(param)
    #     cur_score = cal_score(kpi.res_cluster_weak_coverage[0], kpi.res_cluster_speed[0], kpi.res_cluster_energy[0])
    #     if cur_score < best_score:
    #         param = copy.deepcopy(ori_param1)
    #         cell_state = copy.deepcopy(ori_state1)
    #     else:
    #         best_score = cur_score

    idx = copy.deepcopy(cell_id_list[0: 25])
    best_score = my_optimizer_strategy(best_score, cell_state, param, CSI_data_option=7, repeat_ratio=1,
                                       wireless_contest_simulator=wireless_contest_simulator, cell_id_list=idx)
    best_score = my_optimizer_strategy(best_score, cell_state, param, CSI_data_option=1, repeat_ratio=1,
                                       wireless_contest_simulator=wireless_contest_simulator, cell_id_list=idx)
    best_score = my_optimizer_strategy(best_score, cell_state, param, CSI_data_option=2, repeat_ratio=1,
                                       wireless_contest_simulator=wireless_contest_simulator, cell_id_list=idx)
    best_score = my_optimizer_strategy(best_score, cell_state, param, CSI_data_option=3, repeat_ratio=1,
                                       wireless_contest_simulator=wireless_contest_simulator, cell_id_list=idx)
    best_score = my_optimizer_strategy(best_score, cell_state, param, CSI_data_option=0, repeat_ratio=1,
                                       wireless_contest_simulator=wireless_contest_simulator, cell_id_list=idx)
    best_score = my_optimizer_strategy(best_score, cell_state, param, CSI_data_option=4, repeat_ratio=1,
                                       wireless_contest_simulator=wireless_contest_simulator, cell_id_list=idx)
    best_score = my_optimizer_strategy(best_score, cell_state, param, CSI_data_option=5, repeat_ratio=1,
                                       wireless_contest_simulator=wireless_contest_simulator, cell_id_list=idx)
    best_score = my_optimizer_strategy(best_score, cell_state, param, CSI_data_option=6, repeat_ratio=1,
                                       wireless_contest_simulator=wireless_contest_simulator, cell_id_list=idx)
    pass


def set_pwr(cell_id_list, param, pwr):
    for cell_id in cell_id_list:
        param[str(cell_id)]['pwr_offset'] = pwr


def my_optimizer_pwr(best_score, cell_state, param, wireless_contest_simulator, cell_id_list, ori_pwr, try_pwr):
    for cell_id in cell_id_list:
        ori_pwr_offset = param[str(cell_id)]['pwr_offset']
        if ori_pwr_offset == ori_pwr:
            param[str(cell_id)]['pwr_offset'] = try_pwr
        cell_state, kpi = wireless_contest_simulator.run(param)
        cur_score = cal_score(kpi.res_cluster_weak_coverage[0], kpi.res_cluster_speed[0], kpi.res_cluster_energy[0])
        if cur_score < best_score:
            param[str(cell_id)]['pwr_offset'] = ori_pwr_offset
        else:
            best_score = cur_score
            # print(param)
    return best_score, cell_state


def my_optimizer_strategy(best_score, cell_state, param, CSI_data_option, repeat_ratio, wireless_contest_simulator, cell_id_list):
    for cell_id in cell_id_list:
        ori_azi = np.zeros(8)
        ori_til = np.zeros(8)
        ori_pwr = param[str(cell_id)]['pwr_offset']
        for i in range(0, 8):
            ori_azi[i] = param[str(cell_id)]['beam_azimuth'][i]
            ori_til[i] = param[str(cell_id)]['beam_tilt'][i]
        allocate(cell_state, param, 0, CSI_data_option=CSI_data_option, cell_id_list=np.array([cell_id]),
                 repeat_ratio=repeat_ratio)
        # param[str(cell_id)]['pwr_offset'] = 0
        cell_state, kpi = wireless_contest_simulator.run(param)
        cur_score = cal_score(kpi.res_cluster_weak_coverage[0], kpi.res_cluster_speed[0], kpi.res_cluster_energy[0])
        if cur_score < best_score:
            for i in range(0, 8):
                param[str(cell_id)]['beam_azimuth'][i] = ori_azi[i]
                param[str(cell_id)]['beam_tilt'][i] = ori_til[i]
                param[str(cell_id)]['pwr_offset'] = ori_pwr
        else:
            best_score = cur_score
    return best_score


def allocate(cell_state, param, iter_round, CSI_data_option, cell_id_list, repeat_ratio):
    cell_MRcount, cell_SSB_RSRP, cell_dl_thrp, cell_prb, cell_CSI_RSRP = neaten_data(cell_state)

    azi_angle2portrait = np.array([-45, -30, -15, 15, 30, 45])
    til_angle2portrait = np.array([-2, 4, 11])
    azi_offset = np.array([0, 0, -3, 3, -2, 2, -1, 1])
    til_offset = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    if repeat_ratio < 10:
        azi_offset = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        til_offset = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    if repeat_ratio > 0.5:
        til_angle2portrait = np.array([-2, 4, 13])

    predict_cell_speed = copy.deepcopy(cell_state.predict_cell_speed)
    predict_cell_speed = predict_cell_speed.sort_values(ascending=False)

    # 根据CSI信道的统计信息，调整方位角和下倾角
    for cell_id in cell_id_list:
        if cell_state.predict_cell_speed[cell_id] > predict_cell_speed.iloc[iter_round]:
            continue
        if CSI_data_option == 0:
            CSI_data = copy.deepcopy(cell_dl_thrp[cell_id])
        elif CSI_data_option == 1:
            CSI_data = copy.deepcopy(-cell_prb[cell_id])
        elif CSI_data_option == 2:
            CSI_data = copy.deepcopy(cell_MRcount[cell_id])
        elif CSI_data_option == 3:
            CSI_data = copy.deepcopy(cell_dl_thrp[cell_id] / (cell_prb[cell_id]))
        elif CSI_data_option == 4:
            CSI_data = copy.deepcopy((cell_prb[cell_id]))
        elif CSI_data_option == 5:
            CSI_data = copy.deepcopy(cell_MRcount[cell_id])
        elif CSI_data_option == 6:
            CSI_data = copy.deepcopy(cell_SSB_RSRP[cell_id])
        elif CSI_data_option == 7:
            CSI_data = copy.deepcopy(cell_dl_thrp[cell_id] / cell_MRcount[cell_id])
        elif CSI_data_option == 8:
            CSI_data = copy.deepcopy(cell_dl_thrp[cell_id] / cell_MRcount[cell_id])
        dict = {}
        for k in range(0, 8):
            best_val = CSI_data.min() - 1
            if CSI_data_option == 8:
                best_val = 0
            best_row_idx = 0
            best_col_idx = 1
            dx = np.array([-1, 0, 1])
            for row_idx in range(0, 3):
                for col_idx in range(1, 7):
                    cur_val = 0
                    cnt_valid = 0
                    # 计算以当前格子为中心的3个格子的下行流量总和
                    for m in range(0, 3):
                        cur_row = row_idx
                        cur_col = col_idx + dx[m]
                        # 异常值处理
                        if cell_MRcount[cell_id][cur_row, cur_col] == 0:
                            if CSI_data_option == 0:
                                continue
                            elif CSI_data_option == 1 or CSI_data_option == 2 or CSI_data_option == 3:
                                CSI_data[cur_row, cur_col] = CSI_data.min() - 1
                        if m == 1:
                            cur_val = cur_val + CSI_data[cur_row, cur_col]
                        else:
                            if CSI_data_option == 0:
                                cur_val = cur_val + 0.5 * CSI_data[cur_row, cur_col]
                            elif CSI_data_option == 1:
                                cur_val = cur_val + 2 * CSI_data[cur_row, cur_col]
                            elif CSI_data_option == 2:
                                cur_val = cur_val + 0.5 * CSI_data[cur_row, cur_col]
                            elif CSI_data_option == 3:
                                cur_val = cur_val + 0.5 * CSI_data[cur_row, cur_col]
                        cnt_valid = cnt_valid + 1
                    if cur_val > best_val:
                        best_val = cur_val
                        best_row_idx = row_idx
                        best_col_idx = col_idx
            CSI_data[best_row_idx, best_col_idx] = repeat_ratio * CSI_data[best_row_idx, best_col_idx]
            if str(best_row_idx * 8 + best_col_idx) not in dict:
                dict[str(best_row_idx * 8 + best_col_idx)] = 1
            else:
                dict[str(best_row_idx * 8 + best_col_idx)] = dict[str(best_row_idx * 8 + best_col_idx)] + 1
            offset_idx = dict[str(best_row_idx * 8 + best_col_idx)] - 1
            azi = azi_angle2portrait[best_col_idx - 1] + azi_offset[offset_idx]
            til = til_angle2portrait[best_row_idx] + til_offset[offset_idx]
            param[str(cell_id)]['beam_azimuth'][k] = azi
            param[str(cell_id)]['beam_tilt'][k] = til


if __name__ == '__main__':
    print("Game start!")
    data_enter = {'data_scenario': 13}
    entrance(data_enter)
    print("Game Finish!")
    pass
