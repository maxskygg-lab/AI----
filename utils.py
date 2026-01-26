import time

def fix_latex_errors(text):
    if not text: return text
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    return text

def generate_html_report(chat_history):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI 研究笔记</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; }
            h1 { border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
            .message { margin-bottom: 20px; padding: 15px; border-radius: 8px; }
            .user { background-color: #e3f2fd; border-left: 5px solid #2196F3; }
            .assistant { background-color: #f1f8e9; border-left: 5px solid #4CAF50; }
            .system { background-color: #fff3e0; border-left: 5px solid #ff9800; font-style: italic; }
            .role-label { font-weight: bold; margin-bottom: 5px; display: block; }
        </style>
    </head>
    <body>
        <h1>🎓 AI 深度研读笔记</h1>
        <p>导出时间：""" + time.strftime('%Y-%m-%d %H:%M') + """</p>
    """
    for msg in chat_history:
        role_class = msg['role'] if msg['role'] in ['user', 'assistant'] else 'system'
        role_name = "🧑‍💻 我" if msg['role'] == 'user' else "🤖 AI 研究员" if msg['role'] == 'assistant' else "🔔 系统"
        content_html = msg['content'].replace('\n', '<br>')
        html += f"""
        <div class="message {role_class}">
            <span class="role-label">{role_name}</span>
            <div>{content_html}</div>
        </div>
        """
    html += "</body></html>"
    return html
