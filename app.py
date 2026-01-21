from flask import Flask, render_template, request, jsonify, session,Response
import os
from tools import search_materials_logic, DEEPSEEK_TOOLS_SPEC # 导入工具
import requests
import json
import uuid

app = Flask(__name__)
# 必须设置 secret_key 才能使用 session，可以写一段随机字符串
app.secret_key = 'deepseek_secret_key_12345'
# 配置DeepSeek API密钥（建议通过环境变量设置）
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# --- 关键修改 1：创建一个全局字典来模拟数据库 ---
# 结构: { "用户ID": [消息列表] }
# 使用全局变量存历史（解决 context error）
CHAT_MEMORY = {}

@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    # --- 1. 基础检查与初始化 ---
    # 提前提取所有 request 相关数据，防止进入 generate 后上下文失效
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': '消息不能为空'}), 400

    user_id = session.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id

    if user_id not in CHAT_MEMORY:
        CHAT_MEMORY[user_id] = []
    
    messages = [{
        "role": "system", 
        "content": """你是材料科学专家MatAgent。
                    当需要查询数据时，请直接调用工具，**禁止**在回复正文中展示 `<｜DSML｜`、`function_calls` 或任何 XML 格式的代码块。
                    如果调用了工具，请在获取结果后直接给出人类可读的分析报告。
                    请在每次回复前加入 'MatAgent:' 字样。"""
    }]
    messages.extend(CHAT_MEMORY[user_id])
    messages.append({"role": "user", "content": user_message})

    # --- 2. 第一次请求 (非流式，用于判断工具) ---
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "tools": DEEPSEEK_TOOLS_SPEC,
        "tool_choice": "auto"
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        res_data = response.json()
        msg = res_data['choices'][0]['message']
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # --- 3. 执行工具逻辑 (同步执行) ---
    tool_logs = [] 

    if msg.get("tool_calls"):
        messages.append(msg)
        for tool_call in msg["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]
            
            # 记录日志，准备在 generate 内部 yield 给前端
            tool_logs.append({
                "type": "tool",
                "name": func_name,
                "args": func_args
            })
            
            if func_name == "search_materials":
                try:
                    args_dict = json.loads(func_args)
                    print(f"正在调用工具查询: {args_dict}")
                    tool_result = search_materials_logic(**args_dict)
                except Exception as e:
                    tool_result = {"error": str(e)}
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result)
                })

    # --- 4. 定义流式生成器 ---
    def generate():
        # 【修复点】所有的 yield 必须在 generate 内部！
        
        # A. 先把工具日志发给前端
        for log in tool_logs:
            yield f"data: {json.dumps(log)}\n\n"

        # B. 请求最终的 AI 回复
        final_payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "stream": True
        }
        
        # 此时 request context 已失效，但我们使用的是闭包传入的局部变量 user_id 和 user_message
        print(f"处理来自用户 {user_id} 的消息: {user_message}")
        
        try:
            stream_resp = requests.post(DEEPSEEK_API_URL, json=final_payload, headers=headers, stream=True)
            
            full_reply = ""
            for line in stream_resp.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_str = line_text[6:]
                        if data_str == '[DONE]': break
                        
                        try:
                            data_json = json.loads(data_str)
                            delta = data_json['choices'][0].get('delta', {})
                            
                            if 'content' in delta:
                                content = delta.get('content')
                                if content:
                                    full_reply += content
                                    yield f"data: {json.dumps({'type': 'message', 'content': content})}\n\n"
                        except:
                            continue
            
            # D. 更新内存历史 (在 generate 内部更新全局变量)
            if user_id in CHAT_MEMORY:
                CHAT_MEMORY[user_id].append({"role": "user", "content": user_message})
                CHAT_MEMORY[user_id].append({"role": "assistant", "content": full_reply})
                if len(CHAT_MEMORY[user_id]) > 10:
                    CHAT_MEMORY[user_id] = CHAT_MEMORY[user_id][-10:]

        except Exception as e:
            yield f"data: {json.dumps({'type': 'message', 'content': f'Error: {str(e)}'})}\n\n"

    # --- 5. 返回响应 ---
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')