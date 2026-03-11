"""配置文件"""

import os

# 数据路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'tamper_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# 数据集类别
CATEGORIES = ['easy', 'difficult', 'good']

# 图像大小
IMG_SIZE = (256, 256)

# 特征列表
FEATURES = [
    'ela',           # Error Level Analysis
    'dct',           # DCT域分析
    'cfa',           # CFA插值检测
    'noise',         # 噪声一致性
    'edge',          # 边缘一致性
    'lbp',           # LBP纹理特征
    'histogram',     # 直方图分析
    'sift',          # SIFT特征匹配
    'fft',           # FFT频域分析
    'metadata'       # 元数据分析
]

# 默认阈值
DEFAULT_THRESHOLDS = {
    'ela': 0.5,
    'dct': 0.5,
    'cfa': 0.5,
    'noise': 0.5,
    'edge': 0.5,
    'lbp': 0.5,
    'histogram': 0.5,
    'sift': 0.5,
    'fft': 0.5,
    'metadata': 0.5
}
