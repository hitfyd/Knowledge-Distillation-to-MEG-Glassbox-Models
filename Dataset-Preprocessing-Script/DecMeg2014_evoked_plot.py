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

from scipy import signal
from scipy.io import loadmat

from distilllib.engine.utils import save_figure
from DataUtil import butterBandPassFilter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# MEG RAW 预处理参数
freq_min = 1.
freq_max = 45.
t_min = -0.499
t_max = 1.0
ch_type = 'grad'
ch_reject = {'grad': 4000e-13}
decimate = 4

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
    sub_id = 0

    # 构建一个evoked
    scramble_evoked, face_evoked = raw2epochs(sub_dirs[sub_id], True)

    # 使用DecMeg2014的数据替换evoked的数据
    # 数据集名称
    dataset = 'DecMeg2014'
    # CamCAN经过预处理后的Epochs数据
    epochs_path = 'D:/decoding-the-human-brain/data/'
    test_subject = 4

    filename = '%strain_subject%02d.mat' % (epochs_path, test_subject)
    data = loadmat(filename, squeeze_me=True)
    sfreq = data['sfreq']
    X = data['X']
    y = data['y']
    print("Dataset summary:", X.shape, y.shape)

    # 筛选Grad类型的通道
    grad_channels_num = int(X.shape[1] / 3 * 2)
    X_grad = np.zeros((X.shape[0], grad_channels_num, X.shape[2]))
    for ch in range(int(grad_channels_num / 2)):
        X_grad[:, 2 * ch, :] = X[:, 3 * ch, :]
        X_grad[:, 2 * ch + 1, :] = X[:, 3 * ch + 1, :]
    print("select_grad_channels:", X_grad.shape)
    X = X_grad

    # 4阶巴特沃斯滤波器
    b, a = butterBandPassFilter(0.1, 20, sfreq, order=4)  # b，a: IIR滤波器的分子（b）和分母（a）多项式系数向量
    X = signal.lfilter(b, a, X)

    scramble_data = X[y == 0]
    face_data = X[y == 1]

    scramble_data = np.mean(scramble_data, axis=0)
    face_data = np.mean(face_data, axis=0)

    scramble_evoked.data = scramble_data
    face_evoked.data = face_data

    fig_scramble_evoked_average_joint = scramble_evoked.plot_joint()
    fig_face_evoked_average_joint = face_evoked.plot_joint()
    save_figure(fig_scramble_evoked_average_joint, '../plot/evoked/', 'DecMeg2014_fig_scramble_evoked_average_joint')
    save_figure(fig_face_evoked_average_joint, '../plot/evoked/', 'DecMeg2014_fig_face_evoked_average_joint')

    scramble_times = np.arange(0.16, 0.28, 0.004)
    face_times = np.arange(0.16, 0.28, 0.004)
    # scramble_times = 0.184
    # face_times = 0.224
    cmap = 'Oranges'

    scramble_evoked_peak_feature = scramble_evoked.data[:, 171]
    face_evoked_peak_feature = face_evoked.data[:, 171]
    evoked_feature_db = shelve.open('../dataset/DecMeg2014_evoked_feature')
    evoked_feature_db["scramble"] = scramble_evoked_peak_feature
    evoked_feature_db["face"] = face_evoked_peak_feature
    evoked_feature_db.close()

    # mne.viz.plot_topomap(scramble_evoked_peak_feature, scramble_evoked.info, ch_type='grad', cmap=cmap, outlines='head')
    # fig_scramble_evoked_average = mne.viz.plot_evoked_topomap(scramble_evoked, times='peaks', ch_type=ch_type, average=None, cmap=cmap, outlines='head')
    fig_scramble_evoked_average = scramble_evoked.plot_topomap(scramble_times, ch_type=ch_type, average=None, cmap=cmap,
                                                               sensors=False, outlines='head', ncols=4, nrows="auto")
    # fig_face_evoked_average = mne.viz.plot_evoked_topomap(face_evoked, times='peaks', ch_type=ch_type, average=None, cmap=cmap, outlines='head')
    fig_face_evoked_average = face_evoked.plot_topomap(face_times, ch_type=ch_type, average=None, cmap=cmap,
                                                       sensors=False, outlines='head', ncols=4, nrows="auto")
    save_figure(fig_scramble_evoked_average, '../plot/evoked/', 'DecMeg2014_fig_scramble_evoked_average')
    save_figure(fig_face_evoked_average, '../plot/evoked/', 'DecMeg2014_fig_face_evoked_average')
