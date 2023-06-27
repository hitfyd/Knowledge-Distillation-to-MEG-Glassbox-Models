import os

import mne
import numpy as np
import ray
from matplotlib import pyplot as plt, colors, colorbar, gridspec

from distilllib.engine.utils import predict


class AttributionResult(object):
    def __init__(self, dataset: str, label_names: list,  # dataset information
                 sample_id: int, origin_input: np.ndarray, truth_label: int,  # sample information
                 model_name: str, pred: list, pred_label: int,  # model information and model predictions for the sample
                 channel_attribution_maps: np.ndarray,  # feature attribution maps
                 run_time: float = 0.0):  # the run time of feature attribution method
        # input check
        assert len(channel_attribution_maps.shape) == 2
        assert len(origin_input.shape) == 2
        assert origin_input.shape[0] == channel_attribution_maps.shape[0]
        assert len(label_names) == len(pred) == channel_attribution_maps.shape[1]
        # dataset information
        self.dataset = dataset
        self.label_names = label_names
        # sample information
        self.sample_id = sample_id
        self.origin_input = origin_input  # which shape is [channels, points]
        self.truth_label = truth_label
        self.channels = origin_input.shape[0]
        self.points = origin_input.shape[1]
        # model information and model predictions for the sample
        self.model_name = model_name
        self.pred = pred
        self.pred_label = pred_label
        # feature attribution maps, which shape is [channels, points, len(pred)]
        self.channel_attribution_maps = channel_attribution_maps
        # the run time of feature attribution method
        self.run_time = run_time
        # Automatically generated result ID for retrieval after persistence
        self.result_id = "{}_{}_{}".format(dataset, sample_id, model_name)


def feature_segment(channels, points, window_length):
    channel_windows_num = int(points / window_length)  # 需要计算的通道特征数和时间特征数，总特征数为c_features x p_features
    features_num = channels * channel_windows_num
    channel_list, point_start_list = [], []
    for feature_id in range(features_num):
        channel_list.append(int(feature_id / channel_windows_num))
        point_start_list.append(int(feature_id % channel_windows_num * window_length))
    return features_num, channel_list, point_start_list


def shapley_fakd_parallel(data, model_list, M=1):
    if not ray.is_initialized():
        ray.init(num_gpus=0, num_cpus=32,  # 计算资源
                 local_mode=False,  # 是否启动串行模型，用于调试
                 ignore_reinit_error=True,  # 重复启动不视为错误
                 include_dashboard=False,  # 是否启动仪表盘
                 configure_logging=False,  # 不配置日志
                 log_to_driver=False,  # 日志记录不配置到driver
                 )

    batch_size, channels, points = data.shape
    window_length = points
    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)

    S1 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float16)
    S2 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float16)

    @ray.remote
    def run(feature, data_r):
        S1_r = np.zeros((batch_size, M, channels, points), dtype=np.float16)
        S2_r = np.zeros((batch_size, M, channels, points), dtype=np.float16)
        for m in range(M):
            # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
            feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)  # bool_类型不能改为int8类型
            feature_mark[feature] = 0
            feature_mark = np.repeat(feature_mark, window_length)
            feature_mark = np.reshape(feature_mark, (channels, points))  # reshape是view，resize是copy
            for index in range(batch_size):
                # 随机选择一个参考样本，用于替换不考虑的特征核
                reference_index = (index + np.random.randint(1, batch_size)) % batch_size
                assert index != reference_index  # 参考样本不能是样本本身
                reference_input = data_r[reference_index]
                S1_r[index, m] = S2_r[index, m] = feature_mark * data_r[index] + ~feature_mark * reference_input
                S2_r[index, m][channel_list[feature],
                point_start_list[feature]:point_start_list[feature] + window_length] = \
                    data_r[index][channel_list[feature],
                    point_start_list[feature]:point_start_list[feature] + window_length]
        return feature, S1_r, S2_r

    data_ = ray.put(data)
    rs = [run.remote(feature, data_) for feature in range(features_num)]
    rs_list = ray.get(rs)
    for feature, S1_r, S2_r in rs_list:
        S1[:, feature] = S1_r
        S2[:, feature] = S2_r

    # 计算S1和S2的预测差值
    S1 = S1.reshape(-1, channels, points)
    S2 = S2.reshape(-1, channels, points)
    features_lists = []
    if not isinstance(model_list, list):
        model_list = [model_list]
    for model in model_list:
        S1_preds = predict(model, S1, batch_size=2048, eval=True)
        S2_preds = predict(model, S2, batch_size=2048, eval=True)
        features = (S1_preds.view(batch_size, features_num, M, -1) -
                    S2_preds.view(batch_size, features_num, M, -1)).sum(axis=2) / M
        features_lists.append(features)
    return features_lists


def class_mean_plot(attribution_results, channels_info, top_channel_num=10):
    assert isinstance(attribution_results, list)
    first_attribution_result = attribution_results[0]
    assert isinstance(first_attribution_result, AttributionResult)
    channels = first_attribution_result.channels
    points = first_attribution_result.points
    title = 'Dataset: {}    Model: {}'.format(
        first_attribution_result.dataset, first_attribution_result.model_name)

    heatmap_list = []
    for attribution_result in attribution_results:
        heatmap_list.append(attribution_result.channel_attribution_maps[:, attribution_result.pred_label])
    heatmap_list = np.array(heatmap_list)
    heatmap_channel = heatmap_list.mean(axis=0)
    heatmap_channel = np.abs(heatmap_channel)
    heatmap_channel = (heatmap_channel - np.mean(heatmap_channel)) / (np.std(heatmap_channel))

    # 计算地形图中需要突出显示的通道及名称，注意：由于在绘制地形图时两两合并为一个位置，需要保证TOP通道的名称一定显示，其余通道对显示第一个通道的名称
    mask_list = np.zeros(channels//2, dtype=bool)   # 由于通道类型为Grad，在绘制地形图时两两合并为一个位置
    top_channel_index = np.argsort(-heatmap_channel)[:top_channel_num]
    names_list = []     # 两两合并后对应的通道名称
    for channel_index in range(channels//2):
        if 2*channel_index in top_channel_index:
            mask_list[channel_index] = True
            names_list.append(channels_info.ch_names[2 * channel_index] + '\n')     # 避免显示标记遮挡通道名称
            if 2 * channel_index + 1 in top_channel_index:
                names_list[channel_index] += channels_info.ch_names[2 * channel_index+1] + '\n\n'
        elif 2*channel_index+1 in top_channel_index:
            mask_list[channel_index] = True
            names_list.append(channels_info.ch_names[2 * channel_index+1] + '\n')
        else:
            names_list.append(channels_info.ch_names[2*channel_index])

    # 打印TOP通道及其名称、贡献值
    print(title)
    print("index\tchannel name\tcontribution value")
    top_channels = {}
    for index in top_channel_index:
        print(index, channels_info.ch_names[index], heatmap_channel[index])
        top_channels[index] = (channels_info.ch_names[index], heatmap_channel[index])

    fig = plt.figure(figsize=(6, 6))
    gridlayout = gridspec.GridSpec(ncols=24, nrows=12, figure=fig, top=0.92, wspace=None, hspace=0.2)
    axs1 = fig.add_subplot(gridlayout[:, :23])
    axs1_colorbar = fig.add_subplot(gridlayout[:, 23])

    fontsize = 10
    # 配色方案
    # 贡献由大到小颜色由深变浅：'plasma' 'viridis'
    # 有浅变深：'summer' 'YlGn' 'YlOrRd'
    # 'Oranges'
    cmap = 'Oranges'
    plt.rcParams['font.size'] = fontsize

    fig.suptitle(title, y=0.99, fontsize=fontsize)

    # 绘制地形图
    # 地形图中TOP通道的显示参数
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=4)
    mne.viz.plot_topomap(heatmap_channel, channels_info, ch_type='grad', cmap=cmap, axes=axs1, outlines='head',
                         show=False, names=names_list, mask=mask_list, mask_params=mask_params)
    axs1.set_title("Channel Contribution\n(Topomap)", y=0.9, fontsize=fontsize)
    # 设置颜色条带
    norm = colors.Normalize(vmin=heatmap_channel.min(), vmax=heatmap_channel.max())
    colorbar.ColorbarBase(axs1_colorbar, cmap=cmap, norm=norm)

    plt.show()
    return fig, heatmap_channel, top_channels


def save_figure(fig, save_dir, figure_name, save_dpi=400, format_list=None):
    # EPS format for LaTeX
    # PDF format for LaTeX/Display
    # SVG format for Web
    # JPG format for display
    if format_list is None:
        format_list = ["eps", "pdf", "svg"]
    plt.rcParams['savefig.dpi'] = save_dpi  # 图片保存像素
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)  # 确保路径存在
    for save_format in format_list:
        fig.savefig('{}{}.{}'.format(save_dir, figure_name, save_format), format=save_format,
                    bbox_inches='tight', transparent=False)
