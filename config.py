# config.py
# -----------------------------
# 本文件用于统一管理情感分析模型的配置参数
# -----------------------------

import torch

# 预训练模型名称
MODEL_NAME = "bert-base-chinese"
# 如果你有古文 BERT，可替换为对应名称

# -----------------------------
# 自动选择训练参数（CPU/GPU最佳值）
# -----------------------------
if torch.cuda.is_available():
    # GPU 训练推荐参数
    MAX_LEN =64   #64~128
    BATCH_SIZE = 16      # GPU显存足够可以提高，我的电脑GPU还是合适16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    NUM_WORKERS = 4       # DataLoader加速
    USE_AMP = True        # 混合精度训练
else:
    # CPU 训练推荐参数
    MAX_LEN = 48
    BATCH_SIZE = 8     # CPU内存限制  16时CPU温度过高
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    NUM_WORKERS = 1      # DataLoader线程
    USE_AMP = False       # CPU上禁用AMP

# -----------------------------
# 随机种子
# -----------------------------
RANDOM_SEED = 42

# -----------------------------
# 数据路径
# -----------------------------
SENTIMENT_DATA_PATH = "古文情感分析数据_5000.csv"
FULLTEXT_PATH = "24_structured_events_v3_test.csv"

# -----------------------------
# 模型保存路径
# -----------------------------
MODEL_SAVE_PATH = "result_model/sentiment_bert_model_v1"

"""针对情感全文扫描predict_fulltext_v2.py的改动"""
# -----------------------------
# 新增：二十四史全文扫描专用配置（20260104补充）
# -----------------------------


# # 输出CSV路径
OUTPUT_CSV_PATH = "24_SentimentScan_test.csv"

# 模型版本标识
MODEL_VERSION = "v1"

# 数据源标识
DATA_SOURCE = "二十四史"
