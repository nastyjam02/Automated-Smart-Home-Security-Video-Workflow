import os
import shutil
import json
import av  # PyAV 库，用于视频解码
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度计算
from tqdm import tqdm  # 进度条显示
import pandas as pd


def extract_frames(video_path, num_frames):
    """
    从视频中均匀抽取指定数量的帧
    """
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = set(np.linspace(0, total_frames, num=num_frames, endpoint=False, dtype=int))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))  # # 将帧转换为 RGB24 格式的 numpy 数组并添加
    return frames  # # 返回抽取的帧列表


def compute_video_embedding(frames, processor, model, device):
    """
    计算视频的特征向量，通过对所有抽取帧的编码结果做平均
    """
    pixel_values = processor(frames, return_tensors="pt").pixel_values.to(device)
    encoder = model.get_encoder()
    with torch.no_grad():
        outputs = encoder(pixel_values)
        # # outputs.last_hidden_state 形状: (batch_size, seq_len, hidden_dim),seq_len 是每帧被切成 patch 后的 patch 数
        feats = outputs.last_hidden_state.mean(dim=1)   # [num_frames, hidden_dim]
    return feats.cpu().numpy()[0]  # 所有帧的特征再做平均,视频的全局特征向量


def generate_caption(frames, processor, tokenizer, model, device, min_length=10, max_length=20, num_beams=8):
    """
    为视频生成文本描述
    """
    pixel_values = processor(frames, return_tensors="pt").pixel_values.to(device)
    gen_kwargs = {"min_length": min_length, "max_length": max_length, "num_beams": num_beams}
    with torch.no_grad():
        # # 调用模型 generate 方法进行文本生成
        tokens = model.generate(pixel_values, **gen_kwargs)
        # # 解码 token 序列，跳过特殊 token
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return caption


def batch_infer_and_dedup(
    folder_path,
    processor,
    tokenizer,
    model,
    device,
    num_frames,
    similarity_threshold=0.8,
    out_csv="captions.csv",
    dup_folder="duplicates",
    state_file="state.json"
):
    """
    批量处理视频：抽帧、提特征、生成描述、去重，并支持断点续跑
    """
    os.makedirs(dup_folder, exist_ok=True)

    # 已有状态文件时加载
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as sf:
            state = json.load(sf)
    else:
        state = {"processed": []}  # # 否则初始化

    processed = set(state.get("processed", []))  # # 已处理文件集合
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.mov', '.avi'))]  # # 列出所有视频文件

    # 如果 CSV 存在，则加载已有记录
    if os.path.exists(out_csv):
        df_records = pd.read_csv(out_csv)  # # 已保留记录列表
        # # 保留名列表，用于避免重复添加
        keep_names = set(df_records['video'].tolist())
    else:
        df_records = pd.DataFrame(columns=["video", "caption", "max_similarity"])
        keep_names = set()

    embeddings, names = [], []  # 存储已保留视频的嵌入与名称
    records = df_records.values.tolist()

    for video in tqdm(video_files, desc="Processing videos"):
        if video in processed:
            # # 跳过已处理
            continue
        path = os.path.join(folder_path, video)
        try:
            frames = extract_frames(path, num_frames)
        except Exception:
            # # 解码失败时记录并跳过
            print(f"跳过无法解码视频: {video}")
            processed.add(video)
            continue
        # # 计算嵌入和生成描述
        emb = compute_video_embedding(frames, processor, model, device)
        cap = generate_caption(frames, processor, tokenizer, model, device)

        # 与已有保留视频计算相似度
        if embeddings:
            sims = cosine_similarity([emb], embeddings)[0]
            best_j = int(np.argmax(sims))  # # 找出最高相似度的索引
            max_sim = float(sims[best_j])
        else:
            best_j, max_sim = None, 0.0

        if max_sim >= similarity_threshold:
            # 重复：移动文件并写 JSON 信息
            dst = os.path.join(dup_folder, video)
            shutil.move(path, dst)
            info = {"duplicate": video, "matched_to": names[best_j], "similarity": max_sim}
            json_path = os.path.join(dup_folder, f"{os.path.splitext(video)[0]}.json")
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(info, jf, ensure_ascii=False, indent=2)
        else:
            # 保留：更新列表和记录
            embeddings.append(emb)
            names.append(video)
            records.append([video, cap, max_sim])
        # 标记为已处理
        processed.add(video)
        # 保存中间状态：CSV 与 状态 JSON
        pd.DataFrame(records, columns=["video", "caption", "max_similarity"]).to_csv(out_csv, index=False)
        with open(state_file, 'w', encoding='utf-8') as sf:
            json.dump({"processed": list(processed)}, sf, ensure_ascii=False, indent=2)

    print(f"处理完成，总共保留 {len(records)} 个视频。重复视频已移至 '{dup_folder}'。结果保存在 {out_csv}")


if __name__ == "__main__":
    folder = "/remote-home/cr/video_Chain/videos/extracted_videos"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", use_fast=True)  # videomae提取视频帧的视觉特征，是编码器部分的视觉主干网络
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained(
        "Neleac/timesformer-gpt2-video-captioning"
    ).to(device)  # 用于视频字幕生成
    num_frames = model.config.encoder.num_frames

    batch_infer_and_dedup(
        folder,
        processor,
        tokenizer,
        model,
        device,
        num_frames,
        similarity_threshold=0.8,
        out_csv="/remote-home/cr/video_Chain/videos/extracted_videos/video_captions.csv",
        dup_folder="/remote-home/cr/video_Chain/videos/extracted_videos/duplicates",
        state_file="/remote-home/cr/video_Chain/videos/extracted_videos/state.json"
    )
