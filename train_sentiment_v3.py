# train_sentiment.py
# -------------------------------------------------
# 本文件用于训练古文情感分析模型（BERT 三分类）
# 功能：
# 1. 使用 PyTorch AdamW 训练 BERT 模型
# 2. 保存模型和 tokenizer
# 3. 可选：记录训练日志到 CSV 文件
# 4. 自动处理 CSV 中缺失值和标签类型
# 5. 优先从本地缓存加载模型，屏蔽 HuggingFace 警告
# -------------------------------------------------


"""该版本
根据版本2中，类别2权重偏大，进一步对类别权重进行调节实验，权重取 2.5 。
但模型在少数类上不能达到更优平衡
但也不再继续改权重为：2 / 1.5，来跑代码
因为确定：2 / 1.5 不会“更好”，只会“看起来可能有点不一样”。
下面是为什么可以确定，以及在什么极端情况下才值得再试。
一、为什么我可以“确定”——不是凭经验，是凭结果
只用已经跑出来的真实数据，不做任何假设。
1️⃣ 把三次实验按“权重连续变化”排成一条线
权重	         类别2_F1	类别1_Recall	  Accuracy
1.0 (v1.1)	0.4604	      0.6877	  0.6496
2.5 (v3)	0.4630	      0.5469 ↓	  0.6416
3.0 (v2)	0.4380 ↓	  0.7202	  0.6537

关键事实：
F1 在 0.46 左右震荡
没有随权重单调上升
每一次提升，都伴随别的类别明显下降
这已经说明：
模型的瓶颈不在 loss 权重，而在“语义可分性”。

注：
类别权重本质上做了一件事：拉动决策边界向少数类方向平移
但你这个数据有两个特征：
类别 2 文本短、模糊、语境依赖强
与类别 0 / 1 在 embedding 空间高度重叠

结果就是：
权重 ↑ → 边界推过去 → 误伤大量 0 / 1
权重 ↓ → 回到原边界 → 行为 ≈ v1
📌 没有新的信息注入，边界只能平移，不能变形。
"""
import os
import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import config

# -------------------------------------------------
# 训练日志开关
# -------------------------------------------------
SAVE_TRAIN_LOG = False

# -------------------------------------------------
# 覆盖保存路径
# -------------------------------------------------
config.MODEL_SAVE_PATH = os.path.join("result_model", "sentiment_bert_model_v3")


# -------------------------------------------------
# 屏蔽 HuggingFace 警告信息
# -------------------------------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
os.environ["HF_HUB_DISABLE_XET_WARNING"] = "true"


# -------------------------------------------------
# 1. 自定义数据集类
# -------------------------------------------------
class SentimentDataset(Dataset):
    """自定义情感分析数据集"""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# -------------------------------------------------
# 2. 读取数据并划分训练 / 验证集
# -------------------------------------------------
def load_data():
    df = pd.read_csv(config.SENTIMENT_DATA_PATH)
    df = df[["Original Sentence", "Sentiment"]]

    print("缺失值统计：")
    print(df.isna().sum())

    df = df.dropna(subset=["Original Sentence", "Sentiment"])

    df["Sentiment"] = pd.to_numeric(df["Sentiment"], errors="coerce")
    df = df.dropna(subset=["Sentiment"])
    df["Sentiment"] = df["Sentiment"].astype(int)

    df.columns = ["text", "label"]

    return train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=df["label"]
    )


# -------------------------------------------------
# 3. 单轮训练函数（★加入类别权重）
# -------------------------------------------------
def train_model(model, train_loader, optimizer, device, use_amp, scaler, epoch, loss_fn):
    model.train()
    total_loss = 0.0

    if use_amp and scaler is None:
        scaler = torch.amp.GradScaler()

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # ★ 使用带类别权重的 loss
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # ★ 使用带类别权重的 loss
            loss = loss_fn(outputs.logits, labels)

            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    torch.cuda.empty_cache()
    return total_loss / len(train_loader)


# -------------------------------------------------
# 4. 模型评估函数
# -------------------------------------------------
def evaluate_model(model, val_loader, device):
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = torch.argmax(outputs.logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(batch["labels"].numpy())

    print(classification_report(true_labels, preds, digits=4))


# -------------------------------------------------
# 5. 保存训练日志
# -------------------------------------------------
def save_training_log(log_path, epoch_losses):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss"])
        for epoch, loss in enumerate(epoch_losses, start=1):
            writer.writerow([epoch, loss])

    print(f"训练日志已保存到 {log_path}")


# -------------------------------------------------
# 6. 主程序入口
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    train_texts, val_texts, train_labels, val_labels = load_data()

    LOCAL_MODEL_PATH = r"D:\毕设\代码\models\bert-base-chinese"

    required_files = ["config.json", "vocab.txt", "tokenizer_config.json"]
    model_file = "model.safetensors"

    if os.path.exists(LOCAL_MODEL_PATH) and \
            all(os.path.isfile(os.path.join(LOCAL_MODEL_PATH, f)) for f in required_files) and \
            os.path.isfile(os.path.join(LOCAL_MODEL_PATH, model_file)):
        print(f"从本地缓存加载模型：{LOCAL_MODEL_PATH}")
        tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(
            LOCAL_MODEL_PATH, num_labels=3, ignore_mismatched_sizes=True
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(
            config.MODEL_NAME, num_labels=3
        )

    model.to(device)

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.MAX_LEN)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # -------------------------------------------------
    # ★ 新增：类别权重（只改这里）
    # 类别 2 权重取中“2.5”，缓解不平衡
    # -------------------------------------------------
    class_weights = torch.tensor([1.0, 1.0, 2.5], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    epoch_losses = []

    for epoch in range(config.EPOCHS):
        avg_loss = train_model(
            model,
            train_loader,
            optimizer,
            device,
            use_amp,
            scaler,
            epoch,
            loss_fn
        )
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)

    evaluate_model(model, val_loader, device)

    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(config.MODEL_SAVE_PATH)

    print(f"模型已保存到 {config.MODEL_SAVE_PATH}")

    if SAVE_TRAIN_LOG:
        log_path = os.path.join(config.MODEL_SAVE_PATH, "training_log.csv")
        save_training_log(log_path, epoch_losses)


# -------------------------------------------------
# 入口
# -------------------------------------------------
if __name__ == "__main__":
    main()
"""
D:\AnacondaLocation\envs\nlp\python.exe D:\毕设\代码\sentiment_analysis\train_sentiment_v3.py 
缺失值统计：
Original Sentence    2
Sentiment            3
dtype: int64
从本地缓存加载模型：D:\毕设\代码\models\bert-base-chinese
Epoch 2:   0%|          | 0/500 [00:00<?, ?it/s]Epoch 1 - Loss: 0.9239
Epoch 3:   0%|          | 0/500 [00:00<?, ?it/s]Epoch 2 - Loss: 0.7212
Epoch 3 - Loss: 0.5283
              precision    recall  f1-score   support

           0     0.5818    0.8324    0.6849       346
           1     0.7829    0.5469    0.6440       554
           2     0.4274    0.5051    0.4630        99

    accuracy                         0.6416       999
   macro avg     0.5974    0.6281    0.5973       999
weighted avg     0.6780    0.6416    0.6402       999

模型已保存到 sentiment_analysis\sentiment_bert_model_v3

进程已结束，退出代码为 0

"""

"""选择哪一个版本？"""
"""
v1 / v2 / v3 情感模型实验结果对比
表 1 不同模型版本在验证集上的分类性能对比
版本	              类别2权重	    Accuracy	Macro_F1	Weighted_F1	  类别2_Precision	 类别2_Recall	   类别2_F1
v1.2（最新一次）	      无	     0.6927	    0.6154	     0.6878	          0.5574	          0.3434	    0.4250
v2	                 3.0	     0.6537	    0.5935	     0.6575	          0.3706	          0.5354	    0.4380
v3	                 2.5	     0.6416	    0.5973	     0.6402	          0.4274	          0.5051	    0.4630

说明：
类别 0 / 1 / 2 分别对应：负向 / 中性 / 正向（强烈评价）
类别 2 为样本数量最少的少数类
v1 最新结果来自你最后一次重新训练的输出

二、实验现象的客观解读（不下结论）
1️⃣ v1（无类别权重）
整体 Accuracy、Weighted F1 最高
主流类别（0、1）预测稳定
类别 2：
Precision 较高
Recall 明显偏低（漏判较多）
说明：模型更倾向于“谨慎”预测类别 2，只在把握较高时才给出该标签。
2️⃣ v2（类别 2 权重 = 3）
类别 2 Recall 显著提升
但 Precision 明显下降
整体 Accuracy 与 Weighted F1 均下降
说明：权重过大导致模型对类别 2 过度敏感，出现一定程度的“泛化扩张”。
3️⃣ v3（类别 2 权重 = 2.5）
类别 2 的 Precision / Recall 达到相对均衡
类别 2 F1 为三者中最高
但整体性能进一步下降
说明：v3 是一个**“少数类友好型”模型**，但以牺牲整体稳定性为代价。
"""



"""
1,自动检测 GPU/CPU：
GPU 可用 → AMP 自动开启 + num_workers 增加 + pin_memory=True
GPU 不可用 → AMP 自动关闭，CPU 训练正常

2,AMP 使用：
只有 GPU 有效，不会再出现 CPU 警告
在 GPU 上训练速度提升 1.5~2 倍，显存占用降低

3,DataLoader 优化：
GPU：更多线程 + pinned memory，提高加载速度
CPU：线程适中，防止系统卡顿

4,原注释、功能保持不变：
训练日志、评估、模型保存等都不受影响
tqdm 显示仍然清晰、每 epoch 内显示进度
"""