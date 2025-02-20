import requests
import json

def chat_with_llm(messages):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    data = {
        "messages": messages
    }
    
    # 发送请求并获取响应
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    # 完整的响应文本
    full_response = ""
    
    # 处理流式响应
    try:
        for line in response.iter_lines():
            if not line:
                continue
                
            # 解码二进制数据
            line = line.decode('utf-8')
            
            # 检查是否是SSE数据行
            if line.startswith('data: '):
                data = line[6:]  # 移除 'data: ' 前缀
                
                if data == '[DONE]':
                    print("\n[会话结束]")
                    break
                    
                # 打印实时token并刷新输出
                print(data, end='', flush=True)
                full_response += data
                
    except KeyboardInterrupt:
        print("\n[用户中断]")
        return full_response
    except Exception as e:
        print(f"\n[错误] {str(e)}")
        return full_response
        
    return full_response

def main():
    # 初始化对话历史
    conversation = []
    
    print("开始对话 (输入 'quit' 结束对话)")
    
    while True:
        # 获取用户输入
        user_input = input("\n用户: ")
        
        if user_input.lower() == 'quit':
            print("对话结束")
            break
        
        # 添加用户消息到对话历史
        conversation.append({
            "role": "user",
            "content": user_input
        })
        
        # 调用API并打印响应
        print("\nAI: ", end='')
        response = chat_with_llm(conversation)
        
        if response:  # 只有在有响应时才添加到对话历史
            conversation.append({
                "role": "assistant",
                "content": response
            })

if __name__ == "__main__":
    main()