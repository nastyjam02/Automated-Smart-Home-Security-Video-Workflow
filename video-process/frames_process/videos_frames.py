import os
import shutil
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms  # 提供预训练模型、图像变换工具
import torchvision.models as models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 固定参数
VIDEO_ROOT = "/remote-home/cr/video_Chain/videos/test_pipeline"        # # 待处理视频的根目录
TEMP_DIR = "/remote-home/cr/video_Chain/temp_dir"                  # 临时帧存放目录
OUT_DIR = "/remote-home/cr/video_Chain/out1_dir"                  # 去重后帧存放目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#  抽取视频帧
def extract_frames(video_path, output_dir, interval_sec):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)  # 计算间隔帧数
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(Path(output_dir) / frame_name), frame)
            saved += 1
        count += 1

    cap.release()
    return saved


def load_feature_extractor(device):
    model = models.resnet50(pretrained=True)  # 下载预训练模型
    modules = list(model.children())[:-1]  # 去掉分类头
    feature_extractor = nn.Sequential(*modules).to(device)
    feature_extractor.eval()  # 切换到评估模式 eval() 以关闭 Dropout/BatchNorm 的训练行为。
    return feature_extractor

#  单张图像特征抽取
def extract_feature(image_path, model, device, transform):
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 PIL Image
    img_pil = transforms.ToPILImage()(img)  #  transform（后面定义）完成缩放、归一化等
    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(tensor)  # （ResNet50 去头）无梯度地提取特征向量[1,2048,1,1]
    feat = feat.cpu().numpy().reshape(-1)  # 展平为一维长度 2048
    feat = feat / np.linalg.norm(feat)  #  L2 归一化（使特征模长为 1）,归一化后，余弦相似度就是两向量的点积
    return feat


def deduplicate_frames(frame_dir, output_dir, model, device, threshold):
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  # 每帧缩放到 224×224，并做 ImageNet 标准归一化,预训练的 ResNet50 在 ImageNet 上训练，输入时用了特定的均值和方差做归一化

    saved_features = []
    for img_name in sorted(os.listdir(frame_dir)):
        img_path = Path(frame_dir) / img_name
        feat = extract_feature(img_path, model, device, transform)  # 抽取当前帧的特征 feat
        if not saved_features:
            saved_features.append(feat)  # 第一个帧，直接保存
            shutil.copy(str(img_path), str(Path(output_dir) / img_name))
        else:
            sims = cosine_similarity([feat], saved_features)[0]  # 否则计算它与已保存帧特征的余弦相似度列表 sims
            if np.max(sims) < threshold:
                saved_features.append(feat)  # 只要所有相似度都低于 threshold（默认 0.95），就视为“新”帧，保存并把特征加入 saved_features
                shutil.copy(str(img_path), str(Path(output_dir) / img_name))
    return len(saved_features)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Frame sampling and deduplication")
    parser.add_argument("--interval", type=float, default=2.0, help="采样间隔（秒）")
    parser.add_argument("--threshold", type=float, default=0.95, help="相似度阈值（0-1）")
    args = parser.parse_args()

    print(f"开始批量处理目录 {VIDEO_ROOT} 下的所有视频，每 {args.interval}s 采样一次，阈值 {args.threshold}")
    print("正在加载 ResNet50，请耐心等待哦🎈...")
    model = load_feature_extractor(DEVICE)

    for video_path in Path(VIDEO_ROOT).rglob("*.mp4"):
        rel = video_path.relative_to(VIDEO_ROOT).with_suffix("")
        temp_dir = Path(TEMP_DIR)
        out_dir = Path(OUT_DIR) / rel

        # 清空临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n处理视频: {video_path}")
        n = extract_frames(video_path, temp_dir, args.interval)
        print(f" 提取 {n} 帧至 {temp_dir}")

        m = deduplicate_frames(temp_dir, out_dir, model, DEVICE, args.threshold)
        print(f" 去重后保存 {m} 帧至 {out_dir}")

        # 清理临时目录
        shutil.rmtree(temp_dir)

    print("所有视频处理完成！")

if __name__ == "__main__":
        main()

