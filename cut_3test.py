#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将TXT文件平均拆分为3个部分（保留空行/分段结构）:二十四史_test_part1/2/3.txt
适配二十四史文本的分段特点，用于测试predict_fulltext_v2.py
"""

import os

def split_txt_into_three_parts(input_file, output_prefix="test_part_", encoding="utf-8"):
    """
    将TXT文件按总行数（含空行）平均拆分为3个部分，保留原始分段空行
    :param input_file: 待拆分的TXT文件路径（必填）
    :param output_prefix: 输出文件前缀（默认：test_part_）
    :param encoding: 文件编码（默认UTF-8，适配中文）
    :return: None
    """
    # 1. 校验输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        return

    # 2. 读取文件所有行（保留空行、换行符，完全还原原始结构）
    try:
        with open(input_file, "r", encoding=encoding) as f:
            all_lines = [line for line in f]  # 保留所有行（包括空行、仅换行的行）
    except Exception as e:
        print(f"错误：读取文件失败 → {e}")
        return

    # 3. 基础统计
    total_all_lines = len(all_lines)  # 总行数（含空行）
    total_valid_lines = len([line for line in all_lines if line.strip()])  # 有效行数（非空行）

    if total_all_lines == 0:
        print("错误：待拆分的TXT文件无任何内容！")
        return

    # 4. 计算拆分节点（按总行数平均，剩余行分配到第三部分）
    part_line_num = total_all_lines // 3
    part1 = all_lines[:part_line_num]
    part2 = all_lines[part_line_num: part_line_num * 2]
    part3 = all_lines[part_line_num * 2:]

    # 5. 写入拆分后的文件（保留原始换行/空行）
    for idx, part_content in enumerate([part1, part2, part3], 1):
        output_file = f"{output_prefix}{idx}.txt"
        try:
            with open(output_file, "w", encoding=encoding) as f:
                f.writelines(part_content)  # 直接写入所有行，保留原始格式
        except Exception as e:
            print(f"错误：写入第{idx}部分失败 → {e}")
            continue

        # 统计当前部分的行数
        part_all_lines = len(part_content)
        part_valid_lines = len([line for line in part_content if line.strip()])
        print(f"✅ 第{idx}部分生成完成：{output_file}")
        print(f"   - 总行数（含空行）：{part_all_lines} | 有效行数：{part_valid_lines}")

    # 6. 打印整体拆分统计
    print("\n📊 拆分完成汇总：")
    print(f"原文件：{input_file}")
    print(f"原文件总行数（含空行）：{total_all_lines} | 有效行数：{total_valid_lines}")
    print(f"拆分后：")
    print(f"  第1部分：{len(part1)}行 | 第2部分：{len(part2)}行 | 第3部分：{len(part3)}行")

# ------------------- 核心调用入口 -------------------
if __name__ == "__main__":
    # ================== 请根据实际情况修改以下参数 ==================
    INPUT_TXT_PATH = "二十四史文白对照版_二十四史语料.txt"  # 你的测试TXT文件路径
    OUTPUT_PREFIX = "二十四史_test_part_"  # 输出文件前缀（可选自定义）
    FILE_ENCODING = "utf-8"  # 文本编码（通常为utf-8，若乱码可尝试gbk/gb2312）
    # ==============================================================

    # 执行拆分
    split_txt_into_three_parts(
        input_file=INPUT_TXT_PATH,
        output_prefix=OUTPUT_PREFIX,
        encoding=FILE_ENCODING
    )

    """
    D:\AnacondaLocation\envs\nlp\python.exe D:\毕设\代码\sentiment_analysis\cut_3test.py 
✅ 第1部分生成完成：二十四史_test_part_1.txt
   - 总行数（含空行）：22341 | 有效行数：10303
✅ 第2部分生成完成：二十四史_test_part_2.txt
   - 总行数（含空行）：22341 | 有效行数：9810
✅ 第3部分生成完成：二十四史_test_part_3.txt
   - 总行数（含空行）：22341 | 有效行数：9872

📊 拆分完成汇总：
原文件：二十四史文白对照版_二十四史语料.txt
原文件总行数（含空行）：67023 | 有效行数：29985
拆分后：
  第1部分：22341行 | 第2部分：22341行 | 第3部分：22341行

进程已结束，退出代码为 0

    """