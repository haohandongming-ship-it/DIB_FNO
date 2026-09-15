"""
可视化样式配置
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_plot_style():
    """设置全局绘图风格"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False,
    })


# 论文配色方案
MODEL_COLORS = {
    'DIB-FNO': '#2196F3',
    'FourCastNet': '#FF9800',
    'AdaptFNO': '#4CAF50',
}

MODEL_MARKERS = {
    'DIB-FNO': 'o',
    'FourCastNet': 's',
    'AdaptFNO': '^',
}

BLOCK_COLORS = {
    'input': '#2196F3',
    'mask': '#4CAF50',
    'filter': '#FF9800',
    'projection': '#9C27B0',
    'afno': '#E91E63',
    'output': '#9C27B0',
}