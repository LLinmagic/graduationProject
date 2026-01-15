# predict_fulltext_v2.py
# -------------------------------------------------
# 使用训练好的情感模型扫描《二十四史》结构化 CSV
# 输入 : 24_structured_events_v3.csv（包含删除 para_id 列后如何修改）
# 输出 : 情感预测结果 CSV（不包含 para_id）
# -------------------------------------------------
#代码并不使用 para_id，在输入文件中删除该列 不会影响模型预测。只要 不在代码中引用 row["para_id"]，程序即可正常运行
# 自动识别分隔符 / 自动修复表头 / CPU 友好
# -------------------------------------------------

import torch
import csv
from transformers import BertTokenizer, BertForSequenceClassification
import config


def batch_predict(texts, model, tokenizer, device):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=config.MAX_LEN
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    return preds.cpu().tolist(), probs.max(dim=1).values.cpu().tolist()


def normalize_fieldnames(fieldnames):
    """
    清洗字段名：去 BOM / 空格 / 引号 / 回车
    """
    cleaned = []
    for f in fieldnames:
        f = f.replace("\ufeff", "")
        f = f.strip().strip('"').strip("'")
        cleaned.append(f)
    return cleaned


def main():
    # -------------------------------------------------
    # 1. 设备
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===== 运行设备：{device} =====")

    # -------------------------------------------------
    # 2. 加载模型
    # -------------------------------------------------
    print("===== 加载情感分析模型 =====")
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_SAVE_PATH)
    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_SAVE_PATH
    ).to(device)
    model.eval()

    batch_size = config.BATCH_SIZE
    results = []
    text_buffer = []
    row_buffer = []

    # -------------------------------------------------
    # 3. 打开 CSV（自动检测分隔符）
    # -------------------------------------------------
    print("===== 开始扫描 24_structured_events_v3.csv =====")
    with open(config.FULLTEXT_PATH, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        dialect = csv.Sniffer().sniff(sample)
        reader = csv.DictReader(f, dialect=dialect)

        # 清洗字段名
        reader.fieldnames = normalize_fieldnames(reader.fieldnames)

        # --- 安全检查 ---
        if "original_text" not in reader.fieldnames:
            raise RuntimeError(
                f"CSV 表头中未找到 original_text 字段，实际字段为：{reader.fieldnames}"
            )

        for idx, row in enumerate(reader, 1):
            # 同步清洗 row 的 key
            row = {k.replace("\ufeff", "").strip(): v for k, v in row.items()}

            text = row["original_text"].strip()
            if not text:
                continue

            text_buffer.append(text)
            row_buffer.append(row)

            if len(text_buffer) >= batch_size:
                preds, confs = batch_predict(
                    text_buffer, model, tokenizer, device
                )

                for r, label, conf in zip(row_buffer, preds, confs):
                    results.append([
                        r.get("event_id", ""),
                        r.get("book", ""),
                        r.get("chapter_title", ""),
                        r.get("biography_id", ""),
                        r.get("bio_name", ""),
                        r.get("person_all", ""),
                        r.get("original_text", ""),
                        label,
                        round(conf, 4),
                        config.MODEL_VERSION,
                        config.DATA_SOURCE
                    ])

                text_buffer.clear()
                row_buffer.clear()

            if idx % 2000 == 0:
                print(f"已处理 {idx} 条记录")

        # 剩余 batch
        if text_buffer:
            preds, confs = batch_predict(
                text_buffer, model, tokenizer, device
            )
            for r, label, conf in zip(row_buffer, preds, confs):
                results.append([
                    r.get("event_id", ""),
                    r.get("book", ""),
                    r.get("chapter_title", ""),
                    r.get("biography_id", ""),
                    r.get("bio_name", ""),
                    r.get("person_all", ""),
                    r.get("original_text", ""),
                    label,
                    round(conf, 4),
                    config.MODEL_VERSION,
                    config.DATA_SOURCE
                ])

    # -------------------------------------------------
    # 4. 写出 CSV
    # -------------------------------------------------
    print("===== 写入结果 CSV =====")
    with open(config.OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id",
            "book",
            "chapter_title",
            "biography_id",
            "bio_name",
            "person_all",
            "original_text",
            "sentiment_label",
            "confidence",
            "model_version",
            "source"
        ])
        writer.writerows(results)

    print(f"===== 完成：共输出 {len(results)} 条记录 =====")
    print(f"===== 输出文件：{config.OUTPUT_CSV_PATH} =====")


if __name__ == "__main__":
    main()