from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from typing import List, Dict
import torch
import uvicorn
from pydantic import BaseModel

app = FastAPI()

# 全局变量
model_path = '/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 设置pad token和eos token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

# 用于存储每个会话的历史记录
chat_histories = {}

def generate_response(messages: List[Dict]):
    # 使用对话模板格式化整个对话历史
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 生成回复的输入
    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to('cuda')
    
    # 创建streamer
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    # 生成配置
    generation_kwargs = dict(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )
    
    # 在新线程中运行生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 清理显存
    try:
        for text in streamer:
            yield f"data: {text}\n\n"
    finally:
        del model_inputs
        torch.cuda.empty_cache()

@app.post("/v1/chat/completions")
async def chat_endpoint(request: ChatRequest):
    def generate():
        try:
            for chunk in generate_response([{"role": msg.role, "content": msg.content} for msg in request.messages]):
                yield chunk
        except Exception as e:
            yield f"data: Error occurred: {str(e)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'text/event-stream',
        }
    )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)