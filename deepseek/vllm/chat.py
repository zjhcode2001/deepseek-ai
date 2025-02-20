from openai import OpenAI

messages=[{'role':'system','content':'你是一个傲娇AI'}]

def chat():
    while True:
        user = input('你：')
        messages.append({'role':'user','content':user})
        client = OpenAI(
            base_url="http://localhost:6006/v1",
            api_key="not-needed"
        )
        response = client.chat.completions.create(
            model="/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B",
            messages=messages,
            stream=True
        )
        
        full_assistant = ''
        print('AI：', end='')
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                ai_response = chunk.choices[0].delta.content
                print(ai_response, end='')  # 添加 end='' 参数
                full_assistant += ai_response
        print('\n')  # 在回答结束后才换行
                
        messages.append({'role':'assistant','content':full_assistant})  # 注意这里改为 'assistant'
        
if __name__ == '__main__':
    chat()