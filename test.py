import sys
sys.path.append('third_party/Matcha-TTS')
from modelscope import AutoModelForCausalLM, AutoTokenizer
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import pygame
import os
import time

# Qwen 模型路径
model_name = r"C:\Users\G1581\Desktop\AI\Qwen3_0.6B\Qwen3-0.6B"

# CosyVoice 模型路径
cosyvoice_model_path = r"C:\Users\G1581\Desktop\AI\CosyVoice\pretrained_models\CosyVoice2-0.5B"
prompt_speech_path = r'C:\Users\G1581\Desktop\AI\CosyVoice\voice-wav\gyf.wav'

# 加载 Qwen 模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # 自动选择数据类型
    device_map="auto"    # 自动分配设备
)

# 加载 CosyVoice 模型
cosyvoice = CosyVoice2(cosyvoice_model_path, load_jit=False, load_trt=False, fp16=False, use_flow_cache=True)
prompt_speech_16k = load_wav(prompt_speech_path, 16000)  # 加载参考语音

while True:
    # 获取用户输入
    prompt = input("请输入您的问题（输入 'exit' 退出）：")
    if prompt.lower() == "exit":
        print("程序已退出。")
        break

    # 准备 Qwen 模型输入
    messages = [
        {"role": "user", "content": prompt}  # 定义用户角色和内容
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 不对输入进行分词
        add_generation_prompt=True,  # 添加生成提示
        enable_thinking=False  # 启用“思考模式”，默认为 True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)  # 转换为张量并移动到模型设备

    # 执行文本生成
    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=500  # 设置最大生成的 token 数
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()  # 提取生成的 token

    end_time = time.time()
    # 计算生成时间
    all_time = end_time - start_time
    print(f"文本生成耗时: {all_time:.2f} 秒")

    # 解码生成的内容
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")  # 解码最终生成的内容
    print("生成内容:", content)

    # 使用 CosyVoice 将生成的文本转换为语音
    for i, j in enumerate(cosyvoice.inference_instruct2(content, '用中文正常情绪说这句话', prompt_speech_16k, stream=False)):
        # 保存生成的语音文件
        output_path = rf'C:\Users\G1581\Desktop\AI\CosyVoice\output\output_{i}.wav'
        torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)

        # 播放生成的语音
        if os.path.exists(output_path):
            print("开始播放音频...")
            pygame.mixer.init()  # 初始化混音器
            pygame.mixer.music.load(output_path)  # 加载音频文件
            pygame.mixer.music.play()  # 播放音频

            # 等待音频播放完成
            while pygame.mixer.music.get_busy():
                continue

            print("音频播放完成。")
        else:
            print(f"文件 {output_path} 不存在，无法播放。")