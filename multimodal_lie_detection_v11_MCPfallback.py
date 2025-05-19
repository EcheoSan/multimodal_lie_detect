import os
import sys
import json
import numpy as np
import whisper
import torch
from transformers import pipeline
from moviepy.editor import VideoFileClip
from PIL import Image
from deepface import DeepFace

# === ENV CONFIG ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === GLOBAL MODEL INSTANCES (reuse for memory efficiency) ===
whisper_model = whisper.load_model("base")
def load_audio_classifier_with_fallback():
    try:
        if torch.backends.mps.is_available():
            print("🚀 嘗試使用 MPS")
            return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=torch.device("mps"))
        else:
            print("⚠️ MPS 不可用，改用 CPU")
            return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=-1)
    except Exception as e:
        print(f"⚠️ MPS 載入失敗：{e}")
        print("🔁 自動 fallback 到 CPU")
        return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=-1)
audio_classifier = load_audio_classifier_with_fallback()
text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=7)


# === CONFIG ===
FRAME_RATE = 16000
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
AUDIO_LABEL_MAP = {
    "ang": "angry", "hap": "happy", "sad": "sad",
    "neu": "neutral", "fea": "fear", "dis": "disgust", "sur": "surprise"
}
TEXT_LABEL_MAP = {
    "anger": "angry", "joy": "happy", "sadness": "sad", "fear": "fear",
    "disgust": "disgust", "surprise": "surprise", "neutral": "neutral", "love": "happy"
}
CONSISTENCY_THRESHOLD = 0.75

# === STEP 1: Extract Audio ===
def extract_audio(video_path, output_wav):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_wav, fps=FRAME_RATE)

# === STEP 2: Transcribe Speech ===
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# === STEP 3: Analyze Audio Emotion ===
def analyze_audio_emotion(audio_path):
    global audio_classifier
    try:
        results = audio_classifier(audio_path, top_k=7)
    except NotImplementedError as e:
        print(f"⚠️ MPS 推論失敗：{e}，改用 CPU 重新載入模型")
        # fallback 時強制使用 CPU 重建模型
        
        audio_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=-1)
        results = audio_classifier(audio_path, top_k=7)
    
    scores = {
        AUDIO_LABEL_MAP.get(r["label"].lower(), None): float(r["score"])
        for r in results if AUDIO_LABEL_MAP.get(r["label"].lower(), None)
    }
    vector = [scores.get(e, 0.0) for e in EMOTION_CLASSES]
    return {"vector": vector, "raw": scores}

# === STEP 4: Analyze Text Emotion ===
def analyze_text_emotion(text):
    results = text_classifier(text[:256])[0]  # returns a list inside a list
    scores = {TEXT_LABEL_MAP.get(r["label"].lower(), None): float(r["score"]) for r in results if TEXT_LABEL_MAP.get(r["label"].lower(), None)}
    vector = [scores.get(e, 0.0) for e in EMOTION_CLASSES]
    return {"vector": vector, "raw": scores}

# === STEP 5: Extract Frames ===
def extract_frames(video_path, output_dir="frames", fps=1):
    os.makedirs(output_dir, exist_ok=True)
    clip = VideoFileClip(video_path)
    frame_paths = []
    for t in range(0, int(clip.duration), max(1, int(1/fps))):
        frame = clip.get_frame(t)
        path = os.path.join(output_dir, f"frame_{t}.jpg")
        Image.fromarray(frame).save(path)
        frame_paths.append(path)
    return frame_paths

# === STEP 6: Analyze Facial Emotion ===
def analyze_facial_emotion(paths):
    vectors = []
    all_scores = []
    for path in paths:
        try:
            result = DeepFace.analyze(img_path=path, actions=["emotion"], enforce_detection=False)[0]
            scores = {k.lower(): v/100.0 for k, v in result["emotion"].items()}
            vector = [scores.get(e, 0.0) for e in EMOTION_CLASSES]
            vectors.append(vector)
            all_scores.append(scores)
        except Exception as e:
            print(f"⚠️ Failed to analyze {path}: {e}")
    if not vectors:
        raise RuntimeError("No valid frames for facial emotion analysis")
    average = np.mean(vectors, axis=0).tolist()
    return {"vector": average, "raw": all_scores}

# === COSINE SIMILARITY ===
def cosine_similarity(v1, v2):
    a, b = np.array(v1), np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# === MAIN ANALYSIS FUNCTION ===
def analyze_single_video(video_path):
    audio_path = "extracted_audio.wav"
    extract_audio(video_path, audio_path)
    text = transcribe_audio(audio_path)
    audio_result = analyze_audio_emotion(audio_path)
    text_result = analyze_text_emotion(text)
    frames = extract_frames(video_path)
    face_result = analyze_facial_emotion(frames)

    sim_audio_text = cosine_similarity(audio_result["vector"], text_result["vector"])
    sim_audio_face = cosine_similarity(audio_result["vector"], face_result["vector"])
    sim_text_face = cosine_similarity(text_result["vector"], face_result["vector"])

    consistency = {
        "audio_text_similarity": sim_audio_text,
        "audio_facial_similarity": sim_audio_face,
        "text_facial_similarity": sim_text_face,
        "threshold": CONSISTENCY_THRESHOLD,
        "overall_consistent": all([
            sim_audio_text > CONSISTENCY_THRESHOLD,
            sim_audio_face > CONSISTENCY_THRESHOLD,
            sim_text_face > CONSISTENCY_THRESHOLD
        ])
    }

    return {
        "video": os.path.basename(video_path),
        "text": text,
        "audio_emotion": audio_result,
        "semantic_emotion": text_result,
        "facial_emotion": face_result,
        "consistency": consistency
    }

# === CLI USAGE ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 請提供影片路徑，例如 python script.py /path/to/video.mp4")
        sys.exit(1)
    path = sys.argv[1]
    result = analyze_single_video(path)
    with open("analysis_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("✅ 分析完成，已儲存為 analysis_output.json")
