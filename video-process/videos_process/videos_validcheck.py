import av
import os
import shutil
# 判断一个视频是否包含有效的视频流，如果没有（例如损坏或不是视频文件），就返回 False
def has_valid_video_stream(video_path):
    try:
        container = av.open(video_path)
        return len(container.streams.video) > 0  # # 判断是否包含至少一个视频流
    except Exception:
        return False

def move_invalid_videos_recursive(root_dir, invalid_dir):
    os.makedirs(invalid_dir, exist_ok=True)  # # 如果目标“无效视频”目录不存在，就创建它（包括所有父目录）

    for dirpath, _, filenames in os.walk(root_dir):  # # 递归遍历 root_dir 下所有子目录和文件
        for filename in filenames:
            if filename.lower().endswith(".mp4"):
                full_path = os.path.join(dirpath, filename)  # # 获取完整路径
                if not has_valid_video_stream(full_path):  # # 判断视频是否有效
                    print(f"[移动] 无效视频流文件: {full_path}")
                    try:
                        rel_path = os.path.relpath(full_path, root_dir)
                        dest_path = os.path.join(invalid_dir, rel_path)  #  # 构造目标路径（保持原始子目录结构）
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.move(full_path, dest_path)  # # 移动无效视频到目标目录
                    except Exception as e:
                        print(f"[错误] 移动失败: {full_path} -> {e}")

if __name__ == "__main__":
    root_video_dir = "extracted_videos"  # # 要检查的主视频目录（含子目录）
    invalid_dir = os.path.join(root_video_dir, "invalid")  # # 所有无效视频要移动到的目录
    move_invalid_videos_recursive(root_video_dir, invalid_dir)
