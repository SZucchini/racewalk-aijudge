import glob
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate

def norm_bodysize(data):
    nose_y = data[:, 0, 1]
    hip = (data[:, 11, 1] + data[:, 12, 1]) / 2
    std = abs(nose_y - hip)
    K = 1 / std
    A = np.array([[[k, 0], [0, k]] for k in K])
    data = data[:, :, :] @ A
    return data

def linalg(x):
    return np.linalg.norm(x)

def culc_angle(hip, knee, ankle):
    knee_to_hip = np.array(hip - knee)
    knee_to_ankle = np.array(ankle - knee)
    len_knee_to_hip = np.apply_along_axis(linalg, 2, knee_to_hip)
    len_knee_to_ankle = np.apply_along_axis(linalg, 2, knee_to_ankle)

    tmp = np.zeros_like(knee_to_ankle)
    tmp[:, :, 0], tmp[:, :, 1] = knee_to_ankle[:, :, 1], -1*knee_to_ankle[:, :, 0]
    inner = np.sum(knee_to_hip*knee_to_ankle, axis=2)
    outer = np.sum(knee_to_hip*tmp, axis=2)

    cos = inner / (len_knee_to_hip * len_knee_to_ankle)
    cos = np.where(cos<-1, -1, cos)
    deg = np.rad2deg(np.arccos(cos))
    idx = (outer > 0)
    deg = np.where(idx, 360-deg, deg)
    return deg

def culc_sd(kpt_path):
    files = glob.glob(kpt_path)
    all_diff = []
    for f in files:
        data = np.load(f, allow_pickle=True)[10:, :, :]
        data = np.squeeze(data)
        angle = culc_angle(data[:, 11:13, :], data[:, 13:15, :], data[:, 15:17, :])
        l_diff = np.diff(angle[:, 0])
        r_diff = np.diff(angle[:, 1])
        all_diff.append(l_diff)
        all_diff.append(r_diff)

    diffs = np.array(list(itertools.chain.from_iterable(all_diff)))
    sd = np.std(abs(diffs))
    return sd

def del_outlier(angle, data, sd):
    dup_idx = []
    r_angle = angle[:, 1]
    r_diff = abs(np.diff(r_angle))
    outlier_idx = np.where(r_diff > 3*sd)[0] + 1
    for i in range(1, len(outlier_idx)):
        if outlier_idx[i] - outlier_idx[i-1] == 1:
            if len(dup_idx) == 0: dup_idx.append(i)
            elif outlier_idx[i] - outlier_idx[dup_idx[-1]] != 1: dup_idx.append(i)
    large_val = np.where(r_angle > 215)[0]
    outlier_idx = np.delete(outlier_idx, dup_idx, 0)
    outlier_idx = np.unique(np.append(outlier_idx, large_val))
    revised = np.delete(data, outlier_idx, 0)
    return revised

def adjust_pos(data):
    nose = data[:, 0, :]
    nose = nose[:, np.newaxis, :]
    data = data - nose
    return data

def findpeaks(x, y, n, w, direction):
    if direction == '-':
        index_all = list(signal.argrelmin(y, order=w))
    elif direction == '+':
        index_all = list(signal.argrelmax(y, order=w))
    index = []
    peaks = []

    for i in range(n):
        if i >= len(index_all[0]):
            break
        index.append(index_all[0][i])
        peaks.append(y[index_all[0][i]])

    if len(index) != n:
        index = index + ([0] * (n - len(index)))
        peaks = peaks + ([0] * (n - len(peaks)))
    index = np.array(index) * x[1]
    return index, peaks

def extract_frames(idx, max_data_len):
    frames = []
    max_length = 0
    for i in range(len(idx)-2):
        length = idx[i+2] - idx[i]
        if length > max_length:
            frames = [idx[i], idx[i+1], idx[i+2], length]
        max_length = max(max_length, length)
    return frames, max(max_data_len, max_length)

def culc_interp(data, start, end, max_data_len):
    fx = interpolate.interp1d([x for x in range(end-start)], data[:, 0], kind='linear')
    fy = interpolate.interp1d([x for x in range(end-start)], data[:, 1], kind='linear')
    X = np.linspace(start-start, end-start-1, max_data_len)
    coor = np.array([fx(X), fy(X)]).T
    return coor

def adjust_frame_length(data, i, width, start, end, max_data_len, res):
    if width == 0:
        coor = data[start:end, i]
    else:
        coor = culc_interp(data[start:end, i, :], start, end, max_data_len)
    res.append(coor)

    if i == 13 or i == 14:
        shank = (data[:, i, :] + data[:, i+2, :]) / 2
        coor = culc_interp(shank[start:end, :], start, end, max_data_len)
        res.append(coor)
    return res

def load_kpt(kpt_path, sd, judge_data):
    data_dict = {}
    fault = dict(zip(judge_data['name'], judge_data['judge']))
    walker = dict(zip(judge_data['name'], judge_data['person_type']))
    kpt_files = glob.glob(kpt_path)
    for f in kpt_files:
        data = np.load(f, allow_pickle=True)[10:, :, :]
        data = np.squeeze(data)
        data = adjust_pos(data)
        data = norm_bodysize(data)
        angle = culc_angle(data[:, 11:13, :], data[:, 13:15, :], data[:, 15:17, :])
        data = del_outlier(angle, data, sd)
        angle = culc_angle(data[:, 11:13, :], data[:, 13:15, :], data[:, 15:17, :])
        output = np.concatenate((data, angle[:, np.newaxis, :]), axis=1)
        key = f.split('/')[5].split('kpt_')[1].split('.npy')[0]
        if key in fault.keys():
            key = key.split('.')[0] + '_' + walker[key] + '_' + fault[key]
            data_dict[key] = output

    peak_idx = {}
    failed_data = []
    max_data_len = 0
    for k, v in data_dict.items():
        frames = []
        idx, _ = findpeaks([x for x in range(len(v[:, 17, 1]))], v[:, 17, 1], 20, 15, '-')
        if np.count_nonzero(idx) < 3:
            failed_data.append(k)
            continue
        peak_idx[k], max_data_len = extract_frames(idx, max_data_len)

    input_data = {}
    for k, v in data_dict.items():
        if k in failed_data:
            continue
        res = []
        data_len = peak_idx[k][3]
        width = max_data_len - data_len
        start = peak_idx[k][0]
        end = peak_idx[k][2]
        for i in range(18):
            res = adjust_frame_length(v, i, width, start, end, max_data_len, res)
        res = np.array(res).transpose(1, 0, 2)
        input_data[k] = res.reshape(-1)
    return input_data, max_data_len

def dict_to_df(input_data, max_data_len):
    df_col = []
    for i in range(max_data_len):
        df_col += ['waste'] * 22
        df_col += [f'l_hip{i}_x', f'l_hip{i}_y',
                    f'r_hip{i}_x', f'r_hip{i}_y',
                    f'l_knee{i}_x', f'l_knee{i}_y',
                    f'l_shank{i}_x', f'l_shank{i}_y',
                    f'r_knee{i}_x', f'r_knee{i}_y',
                    f'r_shank{i}_x', f'r_shank{i}_y',
                    f'l_ankl{i}_x', f'l_ankl{i}_y',
                    f'r_ankl{i}_x', f'r_ankl{i}_y',
                    f'l_angle{i}', f'r_angle{i}']

    df = pd.DataFrame.from_dict(input_data, orient='index')
    df = df.set_axis(df_col, axis=1)
    df['label'] = 0
    bk_idx = []
    lc_idx = []
    for idx, _ in df.iterrows():
        if 'BK' in idx: bk_idx.append(idx)
        elif 'LC' in idx: lc_idx.append(idx)

    bk_input = df.drop(lc_idx)
    lc_input = df.drop(bk_idx)
    for idx, _ in bk_input.iterrows():
        if 'Correct' in idx: bk_input.at[idx, 'label'] = 1
        else: bk_input.at[idx, 'label'] = 0
    for idx, _ in lc_input.iterrows():
        if 'Correct' in idx: lc_input.at[idx, 'label'] = 1
        else: lc_input.at[idx, 'label'] = 0

    bk_input = bk_input.drop('waste', axis=1)
    lc_input = lc_input.drop('waste', axis=1)
    return bk_input, lc_input

def main():
    judge_data = pd.read_csv('../data/interim/judge_result.csv')
    kpt_path = '../data/interim/keypoints/*.npy'
    sd = culc_sd(kpt_path)
    input_data, max_data_len = load_kpt(kpt_path, sd, judge_data)

    bk_input, lc_input = dict_to_df(input_data, max_data_len)
    bk_input.to_csv('../data/processed/bk_input_data.csv')
    lc_input.to_csv('../data/processed/lc_input_data.csv')

if __name__ == "__main__":
    main()
