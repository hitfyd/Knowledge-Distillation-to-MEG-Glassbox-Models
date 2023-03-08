# 将CamCAN数据集中的visual-audio passive任务数据进行预处理，每个subject的epoch单独存储为一个.npz文件
# 预处理流程：带通滤波[1, 45]Hz，时间区间为[-0.296, 0.503]s，取梯度计数据，1/8降采样
# Reference：
# [1] I. Zubarev, R. Zetter, H.-L. Halme, and L. Parkkonen, “Adaptive neural network classifier for decoding MEG signals,” Neuroimage, vol. 197, pp. 425–434, Aug. 2019, doi: 10.1016/j.neuroimage.2019.04.068.

import mne
import os
from glob import glob
from multiprocessing import Pool
import numpy as np
import traceback

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

        data = epochs.get_data()
        labels = epochs.events[:, 2]  # epochs.events的shape是(events_num, 3)，第三列就是event_id
        # 转换数据类型，适应pytorch
        data = data.astype(np.float32)
        labels = labels.astype(np.longlong)

        npz_name = save_path + sub_id_name + '_' + str(len(labels)) + '_epochs.npz'
        np.savez(npz_name, data=data, labels=labels)

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
    # 多线程处理
    pool = Pool()
    pool.map(raw2epochs, sub_dirs)
    pool.close()
    pool.join()
