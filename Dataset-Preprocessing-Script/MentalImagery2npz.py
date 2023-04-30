# 将CamCAN数据集中的visual-audio passive任务数据进行预处理，每个subject的每一个session的epoch单独存储为一个.npz文件
# 预处理流程：带通滤波[1, 45]Hz，时间区间为[0.5, 3.5]s，取梯度计数据，1/8降采样
# Reference：
# [1] D. Rathee, H. Raza, S. Roy, and G. Prasad, “A magnetoencephalography dataset for motor and cognitive imagery-based brain-computer interface,” Sci Data, vol. 8, no. 1, Art. no. 1, Apr. 2021, doi: 10.1038/s41597-021-00899-7.

import mne
import os
from glob import glob
from multiprocessing import Pool
import numpy as np
import traceback

# MEG RAW 预处理参数
freq_min = 8.
freq_max = 30.
t_min = -0.5
t_max = 3.5
ch_type = 'grad'
decimate = 8

# MEG源文件路径、epochs存储路径
data_path = 'C:/MEG_BIDS/'
meg_format = 'sub-*/ses-*/meg/sub-*_ses-*_task-bcimici_meg.fif'
# Epochs存储路径
save_path = 'C:/MentalImagery/'
if not os.path.exists(save_path):
    os.mkdir(save_path)


# npz文件存储两个键值
# 'data'为epochs数据List
# 'labels'为对应事件标签List
def raw2epochs(meg_raw_filename, cover=False):
    sub_ses_id_name = os.path.basename(meg_raw_filename)[:-21]
    # 当cover=False时，判断是否已经处理过，存在该subject的epochs文件则跳过
    if not cover and len(glob(save_path + sub_ses_id_name + '*')) > 0:
        print(sub_ses_id_name, 'has been preprocessed!')
        return
    else:
        print(sub_ses_id_name, 'is being preprocessed!')
    try:
        # 读取源文件并加载到内存中，设置日志级别为‘CRITICAL’（最低）
        raw = mne.io.read_raw_fif(meg_raw_filename, preload=True, verbose='CRITICAL')
        # 滤波
        raw = raw.filter(l_freq=freq_min, h_freq=freq_max)
        # 读取’STI101‘刺激信道上的事件及其开始时间，事件持续时间最小为0.003s
        events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003, output='onset')
        # 排除编号255的事件
        events = mne.pick_events(events, exclude=255)
        # 选择所有梯度计
        picks = mne.pick_types(raw.info, meg=ch_type)
        # 提取epochs，设置降采样，拒绝阈值
        epochs = mne.epochs.Epochs(raw, events, tmin=t_min, tmax=t_max, decim=decimate, baseline=(None, 0.),
                                   detrend=1, preload=True, picks=picks, reject=None)
        # 平衡样本标签数量
        epochs.equalize_event_counts()

        data = epochs.get_data()
        labels = epochs.events[:, 2]  # epochs.events的shape是(events_num, 3)，第三列就是event_id
        # 转换数据类型，适应pytorch
        data = data.astype(np.float32)
        labels = labels.astype(np.longlong)
        # 部分编号存在偏移，需要统一为4, 8, 16, and 32
        if 5 in labels:
            # labels = labels-1
            return
        if 6 in labels:
            # labels = labels - 2
            return

        npz_name = save_path + sub_ses_id_name + '_epochs-' + str(len(labels)) + '.npz'
        np.savez(npz_name, data=data, labels=labels)

    except IOError:
        print(sub_ses_id_name, IOError)
        traceback.print_exc()
    except ValueError:
        print(sub_ses_id_name, ValueError)
        traceback.print_exc()
    except KeyError:
        print(sub_ses_id_name, KeyError)
        traceback.print_exc()


if __name__ == '__main__':
    meg_raw_filenames = glob(data_path + meg_format)
    # 按照受试者编号从小到大排序
    meg_raw_filenames.sort()
    # 单线程测试
    for meg_raw_filename in meg_raw_filenames:
        raw2epochs(meg_raw_filename)
    # # 多线程处理
    # pool = Pool(2)  # 每个会话的MEG源文件均为2GB以上，并行线程不能过多，避免内存跑满
    # pool.map(raw2epochs, meg_raw_filenames)
    # pool.close()
    # pool.join()
