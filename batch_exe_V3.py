import os
import csv
import json
import traceback
from multimodal_lie_detection_v11_MCPfallback import analyze_single_video

# 路徑設定
VIDEO_DIR = "videos"
OUTPUT_DIR = "outputs"
CSV_PATH = "results.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 處理 numpy.float32 無法序列化的問題
import numpy as np
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    else:
        return obj

# 準備 CSV（若檔案不存在，建立並加上表頭）
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "text",
            "audio_text_similarity",
            "audio_facial_similarity",
            "text_facial_similarity",
            "overall_consistent"
        ])

# 開始逐一分析影片
for filename in os.listdir(VIDEO_DIR):
    if not filename.endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEO_DIR, filename)
    print(f"🔍 分析中：{filename}")

    try:
        result = analyze_single_video(video_path)

        # 寫入 JSON 結果
        json_path = os.path.join(OUTPUT_DIR, filename.replace(".mp4", ".json"))
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(convert_numpy(result), jf, indent=2, ensure_ascii=False)

        # 寫入 CSV 結果
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,
                result["text"],
                result["consistency"]["audio_text_similarity"],
                result["consistency"]["audio_facial_similarity"],
                result["consistency"]["text_facial_similarity"],
                result["consistency"]["overall_consistent"]
            ])

    except Exception as e:
        print(f"❌ 錯誤：{filename} 無法分析，錯誤原因：{e}")
        traceback.print_exc()
