import os
import json
from pathlib import Path

# 设置输入输出路径
video_dir = Path("/remote-home/cr/video_Chain/videos/videos_valid")  # 视频目录（支持递归）
output_file = "/remote-home/cr/video_Chain/videos/videos.jsonl"       # 输出文件路径

# 支持的视频扩展名
video_extensions = ['.mp4', '.mov', '.avi', '.mkv']

# 打开输出文件准备写入
with open(output_file, 'w', encoding='utf-8') as fout:
    # 遍历所有子目录和文件
    for video_path in sorted(video_dir.rglob("*")):
        # 检查文件扩展名
        if video_path.suffix.lower() in video_extensions:
            # 构造新的 JSON 数据格式
            new_data = {
                "videos": [str(video_path)],
                "text": "<__dj__video> "
            }
            # 写入到 JSONL 文件
            fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")

print(f"✅ 已生成格式化 JSONL 文件: {output_file}")
