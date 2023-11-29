# 将CamCAN数据集中的visual-audio passive任务数据进行预处理，每个subject的epoch单独存储为一个.npz文件
# 预处理流程：带通滤波[1, 45]Hz，时间区间为[-0.296, 0.503]s，取梯度计数据，1/8降采样
# Reference：
# [1] J. R. Taylor et al., “The Cambridge Centre for Ageing and Neuroscience (Cam-CAN) data repository: Structural and functional MRI, MEG, and cognitive data from a cross-sectional adult lifespan sample,” NeuroImage, vol. 144, pp. 262–269, Jan. 2017, doi: 10.1016/j.neuroimage.2015.09.018.
# [2] I. Zubarev, R. Zetter, H.-L. Halme, and L. Parkkonen, “Adaptive neural network classifier for decoding MEG signals,” Neuroimage, vol. 197, pp. 425–434, Aug. 2019, doi: 10.1016/j.neuroimage.2019.04.068.
# [3] https://www.cam-can.com/
import shelve

import joblib
import mne
import os
from glob import glob
from multiprocessing import Pool
import numpy as np
import traceback

from distilllib.engine.utils import save_figure

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# MEG RAW 预处理参数
freq_min = 1.
freq_max = 45.
t_min = -0.296
t_max = 0.503
ch_type = 'grad'
ch_reject = {'grad': 4000e-13}
decimate = 8

# MEG源文件路径、epochs存储路径
data_path = 'D:/Cam-CAN/meg_passive_raw/'
raw_format = '/ses-passive/meg/sub-CC*_ses-passive_task-passive_proc-sss.fif'
# Epochs存储路径
save_path = 'D:/Cam-CAN/freq{}-{}_{}_decimate{}/'.format(freq_min, freq_max, ch_type, decimate)
if not os.path.exists(save_path):
    os.mkdir(save_path)


# npz文件存储两个键值
# 'data'为epochs数据List
# 'labels'为对应事件标签List
def raw2epochs(subject, cover=False):
    sub_id_name = subject[-12:]
    # 当cover=False时，判断是否已经处理过，存在该subject的epochs文件则跳过
    if not cover and len(glob(save_path + sub_id_name + '*')) > 0:
        print(subject, 'has been preprocessed!')
        return
    # 获取源文件绝对路径
    meg_raw_filename = glob(subject + raw_format)[0]
    try:
        # 读取源文件并加载到内存中，设置日志级别为‘CRITICAL’（最低）
        raw = mne.io.read_raw_fif(meg_raw_filename, preload=True, verbose='CRITICAL')
        # 滤波
        raw = raw.filter(l_freq=freq_min, h_freq=freq_max)
        # 读取’STI101‘刺激信道上的事件及其开始时间，事件持续时间最小为0.003s
        events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003, output='onset')
        # 选择所有梯度计
        picks = mne.pick_types(raw.info, meg=ch_type)
        # 提取epochs，设置降采样，拒绝阈值
        epochs = mne.epochs.Epochs(raw, events, tmin=t_min, tmax=t_max, decim=decimate,
                                   detrend=1, preload=True, picks=picks, reject=ch_reject)
        # 平衡样本标签数量
        epochs.equalize_event_counts(['6', '7', '8'])
        epochs.equalize_event_counts([['6', '7', '8'], '9'])

        aud_evoked = epochs['6', '7', '8'].average()
        vis_evoked = epochs['9'].average()

        return aud_evoked, vis_evoked

    except IOError:
        print(subject, IOError)
        traceback.print_exc()
    except ValueError:
        print(subject, ValueError)
        traceback.print_exc()
    except KeyError:
        print(subject, KeyError)
        traceback.print_exc()


if __name__ == '__main__':
    sub_dirs = glob(data_path + 'sub-CC*')
    # 按照受试者编号从小到大排序
    sub_dirs.sort()
    num_id_start = 200
    num_ids = 50
    joblib_file = '../dataset/CamCAN_{}-{}_evoked_list'.format(num_id_start, num_id_start+num_ids)

    if os.path.exists(joblib_file):
        aud_evoked_list, vis_evoked_list = joblib.load(joblib_file)
    else:
        aud_evoked_list, vis_evoked_list = [], []
        for sub_id in range(num_id_start, num_id_start+num_ids):
            aud_evoked, vis_evoked = raw2epochs(sub_dirs[sub_id], True)
            aud_evoked_list.append(aud_evoked)
            vis_evoked_list.append(vis_evoked)

        joblib.dump([aud_evoked_list, vis_evoked_list], joblib_file)

    aud_evoked_average = aud_evoked_list[0].copy()
    vis_evoked_average = vis_evoked_list[0].copy()
    aud_evoked_average.data = aud_evoked_list[0].data / num_ids
    vis_evoked_average.data = vis_evoked_list[0].data / num_ids
    for sub_id in range(1, num_ids):
        aud_evoked_average.data += aud_evoked_list[sub_id].data/num_ids
        vis_evoked_average.data += vis_evoked_list[sub_id].data/num_ids

    evoked_average = aud_evoked_average.copy()
    evoked_average.data = aud_evoked_average.data/2 + vis_evoked_average.data/2

    fig_aud_evoked_average_joint = aud_evoked_average.plot_joint()
    fig_vis_evoked_average_joint = vis_evoked_average.plot_joint()
    save_figure(fig_aud_evoked_average_joint, '../plot/evoked/', 'CamCAN_fig_aud_evoked_average_joint')
    save_figure(fig_vis_evoked_average_joint, '../plot/evoked/', 'CamCAN_fig_vis_evoked_average_joint')

    # times = np.arange(0.05, 0.25, 0.01)
    # times = np.arange(0.1, 0.16, 0.01)
    # times = 0.15
    aud_times = 0.128
    vis_times = 0.136
    cmap = 'Oranges'

    aud_evoked_peak_feature = aud_evoked_average.data[:, 53]
    vis_evoked_peak_feature = vis_evoked_average.data[:, 54]
    evoked_feature_db = shelve.open('../dataset/CamCAN_evoked_feature')
    evoked_feature_db["aud"] = aud_evoked_peak_feature
    evoked_feature_db["vis"] = vis_evoked_peak_feature
    evoked_feature_db.close()

    # fig_aud_evoked_average = aud_evoked_average.plot_topomap(times, ch_type=ch_type, average=0.02, cmap=cmap, sensors=False, ncols=4, nrows="auto")
    # fig_vis_evoked_average = vis_evoked_average.plot_topomap(times, ch_type=ch_type, average=0.02, cmap=cmap, sensors=False, ncols=4, nrows="auto")
    # fig_aud_evoked_average = aud_evoked_average.plot_topomap(times, ch_type=ch_type, cmap=cmap, sensors=False, ncols=4, nrows="auto")
    # fig_vis_evoked_average = vis_evoked_average.plot_topomap(times, ch_type=ch_type, cmap=cmap, sensors=False, ncols=4, nrows="auto")
    fig_aud_evoked_average = aud_evoked_average.plot_topomap(aud_times, ch_type=ch_type, average=None, cmap=cmap, sensors=False, ncols=4, nrows="auto")
    fig_vis_evoked_average = vis_evoked_average.plot_topomap(vis_times, ch_type=ch_type, average=None, cmap=cmap, sensors=False, ncols=4, nrows="auto")
    # fig_evoked_average = evoked_average.plot_topomap(times, ch_type=ch_type, average=0.1, cmap=cmap, sensors=False, ncols=4, nrows="auto")
    save_figure(fig_aud_evoked_average, '../plot/evoked/', 'CamCAN_fig_aud_evoked_average')
    save_figure(fig_vis_evoked_average, '../plot/evoked/', 'CamCAN_fig_vis_evoked_average')
    # save_figure(fig_evoked_average, '../plot/evoked/', 'CamCAN_fig_evoked_average')
