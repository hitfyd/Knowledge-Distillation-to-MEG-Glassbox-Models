import mne
import numpy as np
from math import floor, ceil
from matplotlib import pyplot as plt, colors, colorbar, gridspec
from matplotlib.collections import LineCollection


class AttributionResult(object):
    def __init__(self, dataset: str, label_names: list,  # dataset information
                 sample_id: int, origin_input: np.ndarray, truth_label: int,  # sample information
                 model_name: str, pred: list, pred_label: int,  # model information and model predictions for the sample
                 attribution_method: str,  # feature attribution method
                 attribution_maps: np.ndarray,  # feature attribution maps
                 run_time: float = 0.0):  # the run time of feature attribution method
        # input check
        assert len(attribution_maps.shape) == 3
        assert len(origin_input.shape) == 2
        assert origin_input.shape == attribution_maps.shape[:2]
        assert len(label_names) == len(pred) == attribution_maps.shape[2]
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
        # feature attribution method
        self.attribution_method = attribution_method
        # feature attribution maps, which shape is [channels, points, len(pred)]
        self.attribution_maps = attribution_maps
        # the run time of feature attribution method
        self.run_time = run_time
        # Automatically generated result ID for retrieval after persistence
        self.result_id = "{}_{}_{}_{}".format(dataset, sample_id, model_name, attribution_method)


def class_mean_plot(sample_explanations, channels_info, top_channels=10, attribution_method=None):
    assert isinstance(sample_explanations, list)
    sample_explanation = sample_explanations[0]
    channels = sample_explanation.channels
    points = sample_explanation.points
    title = 'Dataset: {}   Label: {}    Model: {}'.format(
        sample_explanation.dataset, sample_explanation.label_names[sample_explanation.truth_label],
        sample_explanation.model_name)
    if attribution_method is None:
        attribution_method = sample_explanation.attribution_method
    title += '\nInterpretation: {}'.format(attribution_method)

    heatmap_list = []
    origin_input_list = []  # 绘制归因贡献图的原始数据需要平均
    for sample_explanation in sample_explanations:
        heatmap_list.append(sample_explanation.attribution_maps[:, :, sample_explanation.pred_label])
        origin_input_list.append(sample_explanation.origin_input)
    heatmap_list = np.array(heatmap_list)
    origin_input_list = np.array(origin_input_list)

    origin_input = origin_input_list.mean(axis=0)
    heatmap = heatmap_list.mean(axis=0)
    heatmap_channel = heatmap.mean(axis=1)
    heatmap_time = heatmap.mean(axis=0)
    heatmap = (heatmap - np.mean(heatmap)) / (np.std(heatmap))
    heatmap_channel = (heatmap_channel - np.mean(heatmap_channel)) / (np.std(heatmap_channel))
    if attribution_method is not "SingleChannel":
        heatmap_time = (heatmap_time - np.mean(heatmap_time)) / (np.std(heatmap_time))

    # 计算地形图中需要突出显示的通道及名称，注意：由于在绘制地形图时两两合并为一个位置，需要保证TOP通道的名称一定显示，其余通道对显示第一个通道的名称
    mask_list = np.zeros(channels//2, dtype=bool)   # 由于通道类型为Grad，在绘制地形图时两两合并为一个位置
    top_channel_index = np.argsort(-heatmap_channel)[:top_channels]
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
    print("index\tchannel name\tcontribution value")
    id = 0
    for index in top_channel_index:
        print(id, index, channels_info.ch_names[index], heatmap_channel[index])
        id += 1

    fig = plt.figure(figsize=(12, 12))
    gridlayout = gridspec.GridSpec(ncols=48, nrows=12, figure=fig, top=0.92, wspace=None, hspace=0.2)
    axs0 = fig.add_subplot(gridlayout[:, :20])
    axs1 = fig.add_subplot(gridlayout[:9, 20:47])
    axs1_colorbar = fig.add_subplot(gridlayout[2:8, 47])
    axs2 = fig.add_subplot(gridlayout[9:, 24:47])

    fontsize = 16
    linewidth = 2
    # 配色方案
    # 贡献由大到小颜色由深变浅：'plasma' 'viridis'
    # 有浅变深：'summer' 'YlGn' 'YlOrRd'
    # 'Oranges'
    cmap = 'Oranges'
    plt.rcParams['font.size'] = fontsize
    time_xticks = [0, 25, 50, 75, 100]
    time_xticklabels = ['-0.2', '0', '0.2', '0.4', '0.6(s)']

    fig.suptitle(title, y=0.99, fontsize=fontsize)

    # 绘制时间曲线图
    thespan = np.percentile(origin_input, 98)
    xx = np.arange(1, points + 1)

    for channel in range(channels):
        y = origin_input[channel, :] + thespan * (channels - 1 - channel)
        dydx = heatmap[channel, :]

        img_points = np.array([xx, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([img_points[:-1], img_points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(-1, 1), linewidths=(1,))
        lc.set_array(dydx)
        axs0.add_collection(lc)

    axs0.set_xlim([0, points + 1])
    axs0.set_xticks(time_xticks)
    axs0.set_xticklabels(time_xticklabels, fontsize=fontsize)
    axs0.set_xlabel('Time', fontsize=fontsize)
    axs0.set_title("(a)Contribution Map", fontsize=fontsize)

    inversechannelnames = []
    for channel in range(channels):
        inversechannelnames.append(channels_info.ch_names[channels - 1 - channel])

    yttics = np.zeros(channels)
    for gi in range(channels):
        yttics[gi] = gi * thespan

    axs0.set_ylim([-thespan, thespan * channels])
    plt.sca(axs0)
    plt.yticks(yttics, inversechannelnames, fontsize=fontsize//3)

    # 绘制地形图
    # 地形图中TOP通道的显示参数
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=4)
    mne.viz.plot_topomap(heatmap_channel, channels_info, ch_type='grad', cmap=cmap, axes=axs1, outlines='head',
                         show=False, names=names_list, mask=mask_list, mask_params=mask_params)
    axs1.set_title("(b)Channel Contribution\n(Topomap)", y=0.9, fontsize=fontsize)
    # 设置颜色条带
    norm = colors.Normalize(vmin=heatmap_channel.min(), vmax=heatmap_channel.max())
    colorbar.ColorbarBase(axs1_colorbar, cmap=cmap, norm=norm)

    # 绘制时间贡献曲线
    xx = np.arange(1, points + 1)
    img_points = np.array([xx, heatmap_time]).T.reshape(-1, 1, 2)
    segments = np.concatenate([img_points[:-1], img_points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidths=(linewidth+1,))
    lc.set_array(heatmap_time)
    axs2.set_title("(c)Time Contribution", fontsize=fontsize)
    axs2.add_collection(lc)
    axs2.set_ylim(floor(heatmap_time.min()), ceil(heatmap_time.max()))
    axs2.set_xticks(time_xticks)
    axs2.set_xticklabels(time_xticklabels, fontsize=fontsize)
    axs2.set_ylabel('Contribution', fontsize=fontsize)
    axs2.set_xlabel('Time', fontsize=fontsize)
    axs2.patch.set_facecolor('lightgreen')

    plt.show()
    return fig, heatmap, heatmap_channel, heatmap_time
