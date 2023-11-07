import numpy as np
from braindecode.datasets import MOABBDataset

subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=subject_ids)


from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor)
from numpy import multiply

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)


from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)


splitted = windows_dataset.split('session')
train_set = splitted['0train']
valid_set = splitted['1test']


train_data, train_labels = [], []
for i in train_set:
    train_data.append(i[0])
    train_labels.append(i[1])
train_data = np.array(train_data)
train_labels = np.array(train_labels)

test_data, test_labels = [], []
for i in valid_set:
    test_data.append(i[0])
    test_labels.append(i[1])
test_data = np.array(test_data)
test_labels = np.array(test_labels)

dataset_save_path = '../dataset/'
dataset = 'BCIIV2a'
# 保存训练集
np.savez('{}{}_train'.format(dataset_save_path, dataset), data=train_data, labels=train_labels)
# 保存测试集
np.savez('{}{}_test'.format(dataset_save_path, dataset), data=test_data, labels=test_labels)
