
import argparse
import json
import os
import shutil
from pathlib import Path

def extract_and_copy(jsonl_path: Path, dst_dir: Path):
    """
    从 jsonl 文件中提取所有 video 路径，并复制到目标文件夹。
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open('r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[跳过] 第 {line_number} 行 JSON 解析失败：{e}")
                continue

            videos = obj.get("videos") or []
            if not isinstance(videos, list):
                print(f"[警告] 第 {line_number} 行的 'videos' 不是列表，跳过")
                continue

            for vid_path in videos:
                src = Path(vid_path)
                if not src.exists():
                    print(f"[未找到] {src}，跳过")
                    continue

                dst = dst_dir / src.name
                try:
                    shutil.copy2(src, dst)
                    print(f"[复制] {src} → {dst}")
                except Exception as e:
                    print(f"[错误] 复制 {src} 时出错：{e}")

def main():
    parser = argparse.ArgumentParser(
        description="从 JSONL 文件中提取所有视频路径，并复制到新的文件夹。"
    )
    parser.add_argument(
        "--jsonl", "-j",
        required=True,
        type=Path,
        help="输入的 JSONL 文件路径。"
    )
    parser.add_argument(
        "--outdir", "-o",
        required=True,
        type=Path,
        help="复制视频的目标文件夹。"
    )
    args = parser.parse_args()

    extract_and_copy(args.jsonl, args.outdir)

if __name__ == "__main__":
    main()
