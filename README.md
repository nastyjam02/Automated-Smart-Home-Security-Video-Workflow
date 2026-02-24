# Automated-Smart-Home-Security-Video-Workflow

#### 介绍
视频数据预处理pipeline，本项目提出一个全自动的视频数据预处理pipeline，从原始视频下载开始，经过多级清洗、去重和内容分析，最终生成高质量、结构化的视频数据集。为后续针对智能家居安防的多模态模型训练，视频理解任务进行视频数据集构造。

#### pipeline

1.  通过videos_process文件夹中的data_daowload_origin.py下载原始视频数据。
2.  使用videos_validcheck.py初步筛掉损坏的,空的视频文件，所有无效视频移动到目录invalid，有效视频保持在原始目录valid。
3.  videos_create_jsonl.py帮助我们创建data-juicer能够读取的videos.jsonl文件用于数据集配置。
4.  利用data-juicer帮助我们进行视频数据的初步处理，包括视频时长检查，视频tags过滤，视频hash去重，使用到的算子有video_duration_filter；video_deduplicator；video_tagging_from_frames_filter，自定义的配置config_video.yaml，经过data-juicer初步处理得到的结果和日志都将记录在outputs中。
5.  利用得到的outputs中的video-processed.jsonl，使用videos_extract_from_jsonl.py得到data-juicer处理完成的视频数据并保持原valid文件夹结构。
6.  最后一步处理videos_content_dedup.py，利用Neleac/timesformer-gpt2-video-captioning预训练的video caption模型，进行视频内容相似度去重，我们把similarity_threshold设置为0.8（一个合适的阈值），得到最终清洗完全的视频数据，我们同时还保留了duplicates在每个子文件夹中供读者查看去重的视频，并为每个视频match了最相似的视频写在我们重复视频相应的json文件中。
    json文件内容如下：
   { "duplicate": "20250716_SH801D2501000005_1945251081520943104.mp4", 
     "matched_to": "20250716_SH801D2501000005_1945263489215893504.mp4", 
      "similarity": 0.8264052867889404
   }
