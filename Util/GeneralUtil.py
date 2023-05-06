import os

from matplotlib import pyplot as plt

def get_project_path():
    """得到项目路径"""
    package_path = os.path.dirname(__file__)
    project_path = os.path.dirname(package_path)
    return project_path


def create_dir(dir_path):
    os.makedirs(os.path.dirname(dir_path), exist_ok=True)  # 确保路径存在


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
