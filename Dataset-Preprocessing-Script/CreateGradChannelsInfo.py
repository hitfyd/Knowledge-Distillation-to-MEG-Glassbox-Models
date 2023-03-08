# 目前2个数据集的采集设备均为306-channel VectorView MEG system (Elekta Neuromag, Helsinki)，获取其中204通道的Grad所对应的大脑位置信息，用于绘制大脑显著性图
import shelve

import mne

sub_id = 'sub-CC110045'
meg_raw_filename = 'D:/Cam-CAN/meg_passive_raw/{}/ses-passive/meg/{}_ses-passive_task-passive_proc-sss.fif'\
    .format(sub_id, sub_id)
ch_type = 'grad'
save_path = '../dataset/{}_info'.format(ch_type)

raw = mne.io.read_raw_fif(meg_raw_filename)
raw.pick_types(meg=ch_type)     # 筛选通道，可以是梯度计、磁强计或全部

info = raw.info
channel_db = shelve.open(save_path)
channel_db['info'] = info
channel_db.close()

# 验证持久化的有效性
channel_db = shelve.open(save_path)
read_info = channel_db['info']
channel_db.close()
