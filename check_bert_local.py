
"""
from transformers import BertTokenizer, BertForSequenceClassification


model_name = "bert-base-chinese"
save_dir = r"D:\毕设\代码\models\bert-base-chinese"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("bert-base-chinese 已完整保存到本地")
"""
"""
from transformers import BertTokenizer, BertForSequenceClassification
MODEL_PATH = r"D:\毕设\代码\models\bert-base-chinese"
print("开始加载 tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
print("tokenizer 加载成功")

print("开始加载 model...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=3
)
print("model 加载成功")

print("🎉 本地 BERT 自检通过")

"""



"""from transformers import BertTokenizer, BertForSequenceClassification

BASE_MODEL_PATH = r"D:\毕设\代码\models\bert-base-chinese"

# 下载 tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer.save_pretrained(BASE_MODEL_PATH)

# 下载模型主体（不带分类头）
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=3,                  # 三分类
    ignore_mismatched_sizes=True   # 自动初始化分类头
)
model.save_pretrained(BASE_MODEL_PATH)"""



"""
import torch

print(torch.cuda.is_available())  # True 表示可以使用 GPU
print(torch.cuda.device_count())  # GPU 数量
print(torch.cuda.get_device_name(0))  # GPU 名称
"""


"""
2️⃣ 如何选择 MAX_LEN
原则：尽量覆盖文本大部分长度，同时不要浪费计算
方法：
统计文本长度分布（token 数），例如取 95% 的文本长度作为 MAX_LEN
CPU 训练时，可以适当降低，牺牲一些 padding 精度，减轻负担
举例 Python 统计 token 长度：
统计后，比如 95% 文本长度 ≤ 64，就可以安全设置：
MAX_LEN = 64  # CPU模式下安全
如果 GPU 足够，可以用：
MAX_LEN = 128  # 覆盖几乎所有文本
"""
from transformers import BertTokenizer
import pandas as pd
df = pd.read_csv("古文情感分析数据_5000.csv")
texts = df["Original Sentence"].dropna().tolist()
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

lengths = [len(tokenizer.tokenize(t)) for t in texts]
lengths.sort()
max_len_95 = lengths[int(len(lengths)*0.95)]
print("95%文本长度分位数:", max_len_95)
#95%文本长度分位数: 41
"""
CPU 模式
MAX_LEN = 48（略大于 95% 分位数，保证大部分文本覆盖，同时避免过多 padding）
BATCH_SIZE = 8（CPU 安全）
NUM_WORKERS = 1
USE_AMP = False

GPU 模式
MAX_LEN = 64~128（可以覆盖几乎所有文本）
BATCH_SIZE = 16
NUM_WORKERS = 4
USE_AMP = True
这里 CPU 模式选择 48，比 41 稍大一些，留出一定余量，训练计算量低，CPU 温度更安全。
"""
