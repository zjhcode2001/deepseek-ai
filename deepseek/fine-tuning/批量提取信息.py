from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from peft import PeftModel
import json

model_path = '/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B'
lora_path = '/root/deepseek/fine-tuning/output/deepseek/checkpoint-1240'

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 创建生成配置
generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    generation_config=generation_config
)
model = PeftModel.from_pretrained(model, lora_path)

# 初始化对话历史 - 移除了系统提示词
messages = []

def chat(prompt):
    try:
        messages.append({"role": "user", "content": prompt})
        
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        model_inputs = {
            'input_ids': inputs.input_ids.to('cuda'),
            'attention_mask': inputs.attention_mask.to('cuda')
        }
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1028,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        ]
        
        response = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        messages.append({"role": "assistant", "content": response})
        return response
        
    finally:
        del model_inputs, generated_ids
        torch.cuda.empty_cache()

prompt_text = "请提取下列文本中的所有可能的实体，多个相同类型的实体并列需要分开，实体类型包括：犯罪嫌疑人，受害人，被盗货币，物品价值，盗窃获利，被盗物品，作案工具，时间，地点，组织机构。输出格式：json格式，要求输出实体及其类型: "

# 打开文件（假设文件名为 data.jsonl）
i = 1
with open('/root/deepseek/fine-tuning/test.json', 'r', encoding='utf-8') as file:
    for line in file:
        if i > 10:
            print("对话结束")
            break
        try:
            # 解析每行的 JSON 数据
            data = json.loads(line.strip())
            # 提取 context 字段
            context = data.get('context', None)
            # 加上提示词
            if i == 1:
                context = prompt_text + context
            if context:
                print("\n原文: " + context)
                response = chat(context)
                print("\nAI: " + response)
            else:
                print("该行缺少 'context' 字段")
        except json.JSONDecodeError as e:
            print(f"JSON 解码错误: {e}")
        i += 1
