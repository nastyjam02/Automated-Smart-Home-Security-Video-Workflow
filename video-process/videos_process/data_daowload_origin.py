import os
import json
import requests
from datetime import datetime, timedelta
import argparse
import subprocess
import tempfile

# 设备ID列表
device_ids = [
    "SH801D2501000005", "SH801D2501000006", "SH801D2501000008", "SH801D2501000010",
    "SH801D2501000011", "SH801D2501000012", "SH801D2501000013", "SH801D2501000015",
    "SH801D2501000021", "SH801D2501000023", "SH801D2501000026", "SH801D2501000030",
    "SH801D2501000031", "SH801D2501000032", "SH801D2501000033", "SH801D2501000035",
    "SH801D2501000036", "SH801D2501000038", "SH801D2501000039", "SH801D2501000041",
    "SH801D2501000042", "SH801D2501000045", "SH801D2501000048", "SH801D2501000049",
    "SH801D2501000067", "SH801D2501000069", "SH801D2501000072", "SH801D2501000073",
    "SH801D2501000074", "SH801D2501000076"
]

API_HOST = "http://8.129.72.6:7180"
API_PATH = "/api"
HEADERS = {
    'action': 'getCloudStorageEvents',
    'Authorization': 'basic eG1HdWVzdDo3N1pNWmJPanBCZG43MHJk',
    'Content-Type': 'application/json'
}

# 忽略SSL警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_timestamp(dt):
    # 转为毫秒级时间戳
    return int(dt.timestamp() * 1000)

def fetch_events(device_id, start_ts, end_ts, size=100, page=1):
    payload = {
        "start_time": start_ts,
        "end_time": end_ts,
        "devices": [device_id],
        "size": size,
        "page": page
    }
    response = requests.post(
        API_HOST + API_PATH,
        headers=HEADERS,
        data=json.dumps(payload),
        verify=False,
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def download_m3u8_and_convert(url, output_path):
    """下载m3u8文件并转换为mp4"""
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.m3u8', delete=False) as temp_m3u8:
            temp_m3u8_path = temp_m3u8.name
        
        # 下载m3u8文件
        print(f"Downloading m3u8 from: {url}")
        response = requests.get(url, verify=False, timeout=30)
        response.raise_for_status()
        
        # 保存m3u8文件
        with open(temp_m3u8_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # 使用ffmpeg转换为mp4
        print(f"Converting m3u8 to mp4: {output_path}")
        cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,http,https,tcp,tls,crypto,data',  # 允许HTTPS协议
            '-i', temp_m3u8_path,
            '-c', 'copy',  # 直接复制流，不重新编码（更快）
            '-bsf:a', 'aac_adtstoasc',  # 处理音频流
            '-y',  # 覆盖输出文件
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # 清理临时文件
        os.unlink(temp_m3u8_path)
        
        if result.returncode == 0:
            print(f"Successfully converted to: {output_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error converting m3u8 to mp4: {e}")
        # 清理临时文件
        if 'temp_m3u8_path' in locals():
            try:
                os.unlink(temp_m3u8_path)
            except:
                pass
        return False

def download_file(url, file_path):
    """下载文件到指定路径"""
    try:
        response = requests.get(url, verify=False, stream=True, timeout=30)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def save_event_data(device_id, data, day_str):
    folder = os.path.join("videos", device_id)
    os.makedirs(folder, exist_ok=True)
    
    # 保存原始json数据
    json_file_path = os.path.join(folder, f"events_{day_str}.json")
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {json_file_path}")
    
    # 下载视频和缩略图
    if data.get("result") and data.get("payload", {}).get("list"):
        events = data["payload"]["list"]
        for i, event in enumerate(events):
            event_id = event.get("event_id", f"event_{i}")
            
            # 下载视频（m3u8转mp4）
            if event.get("play_url"):
                video_filename = f"{day_str}_{device_id}_{event_id}.mp4"
                video_path = os.path.join(folder, video_filename)
                if os.path.exists(video_path):
                    print(f"Video already exists: {video_path}")
                    continue
                if download_m3u8_and_convert(event["play_url"], video_path):
                    print(f"Downloaded and converted video: {video_path}")


def main(days):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(days):
        day = today - timedelta(days=i)
        start_ts = get_timestamp(day)
        end_ts = get_timestamp(day + timedelta(days=1))
        day_str = day.strftime("%Y%m%d")
        print(f"Fetching {day_str}...")
        for device_id in device_ids:
            try:
                data = fetch_events(device_id, start_ts, end_ts)
                save_event_data(device_id, data, day_str)
            except Exception as e:
                print(f"Error fetching {device_id} {day_str}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download videos by device and date range.")
    parser.add_argument("--days", type=int, default=7, help="Number of days to download (default: 7)")
    args = parser.parse_args()
    main(args.days)
