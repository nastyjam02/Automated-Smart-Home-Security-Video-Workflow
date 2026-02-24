import os
import json
from dashscope import MultiModalConversation

def infer_from_frames(
    frames_dir: str,
    output_path: str,
    api_key: str = None,
    model: str = "qwen-vl-max",
    system_prompt: str = None,
    batch_size: int = 10,
):
    """
    frames_dir: 包含已抽帧图像的目录
    output_path: 保存最终 JSON 结果的路径
    api_key: DASHSCOPE_API_KEY 环境变量覆盖或传入
    model: 多模态模型名称
    system_prompt: 自定义系统提示；若为 None，则使用默认提示
    batch_size: 每次发送给 API 的图像数
    """
    # 加载 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("请提供 DASHSCOPE_API_KEY 或在环境变量中设置它。")

    # 默认 system prompt
    if system_prompt is None:
        system_prompt =  '''你会得到一段监控视频的帧序列，任务分两步：
    1) 先分析视频帧内容，生成一个简洁精炼的 “description” 文本，描述视频帧内容，格式：
       { "description": "…" }
       请从文本中识别出视频中实际出现B类对象，选取列表中的名称，列表中没有出现禁止进行扩展或推理：
       **禁止**使用“一个人”“一辆车”等泛称，主体必须选用具体名称，
         ！如果主体是人一定要判断性别！并且使用矩形框标注人所在位置，给出矩形框四点坐标！
         ！如果主体是猫或者狗一定要判断品种！并且使用矩形框标注宠物所在位置，给出矩形框四点坐标！
         ！如果主体是车一定判断车型！并且使用矩形框标注车所在位置，给出矩形框四点坐标！
         ## B类对象列表
        { 人：女性成人、男性成人、男孩、女孩、男性老年人、女性老年人
         宠物：狗（泰迪、金毛、拉布拉多、柯基、比熊、博美、哈士奇、柴犬、边牧、雪纳瑞）、猫(英短、美短、布偶、缅因、加菲、暹罗、挪威森林猫、俄罗斯蓝猫、苏格兰折耳猫)、兔、鸟
         车：自行车，轿车（小型乘用车），SUV（运动型多用途车），MPV（多用途车型），卡车（货运车），面包车，摩托车，电动车 / 电瓶车（两轮或三轮）
         物品：常见家居物品
        }
    2) 然后，基于上一步的 description 识别视频帧中 B 类对象和发生的 B 类事件，输出为：**B 类对象（发起者）+B 类事件+（行为对象）**的结构，例如：
       --女性成人坐下  
       --狗追逐猫  
       --门打开
    ## 注意事项 ##
    - 禁止输出 B 类事件列表之外的事件！  
    - 如果无法从描述中明确主语和事件，则不标注该事件。  
    - 最终输出严格遵循以下 JSON 格式：
    ```json
    {
      "description": "...",
      "object": ["目标对象1", "目标对象2", ...],
      "目标对象1": ["矩形框左上角坐标", "矩形框左下角坐标", "矩形框右上角坐标", "矩形框右下角坐标"],
      "event": ["事件1（主谓明确）", "事件2", ...]
    }
    ```'''

    # 列出所有图片文件
    imgs = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if not imgs:
        raise FileNotFoundError(f"未在目录 {frames_dir} 找到任何图像文件。")

    results = {}
    # 分批处理
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size]
        image_list = [
            {"type": "image", "image": f"file://{os.path.join(frames_dir, fn)}"}
            for fn in batch
        ]
        # 构建对话消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_list + [
                {"type": "text", "text": "请根据上述两步规则，一次性生成最终的 JSON 输出。"}
            ]}
        ]
        # 调用接口
        resp = MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages
        )
        text = resp["output"]["choices"][0]["message"]["content"][0]["text"]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f"第 {i // batch_size + 1} 批解析失败，原始输出：\n{text}")
            continue
        # 保存本批结果
        for fn in batch:
            results[fn] = data
        print(f"批次 {i // batch_size + 1}: 处理 {len(batch)} 张图片，完成。")

    # 写入输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 全部处理完毕，结果已写入 {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对已抽帧图像进行多模态推理")
    parser.add_argument("frames_dir", help="抽帧图像目录")
    parser.add_argument("output_json", help="保存结果的 JSON 文件路径")
    parser.add_argument("--batch", type=int, default=10, help="每批次图片数量")
    parser.add_argument("--model", default="qwen-vl-max", help="使用的多模态模型")
    args = parser.parse_args()

    infer_from_frames(
        frames_dir=args.frames_dir,
        output_path=args.output_json,
        model=args.model,
        batch_size=args.batch
    )
