import os, loadenv
import uuid
import time
import shutil
import threading
from flask import Flask, send_from_directory, send_file, abort
import io
import mimetypes

config = loadenv.Config()
local_host = config.get_ip()

class MemoryFileServer:
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        
        # 初始化两个缓存字典
        self.image_cache = {}  # 专门存图片
        self.file_cache = {}   # 存通用文件
        
        self._setup_routes()

    def _setup_routes(self):
        # 1. 图片查看路由
        @self.app.route('/image/<image_id>')
        def serve_image(image_id):
            if image_id not in self.image_cache:
                abort(404)
            img_data = self.image_cache[image_id]
            return send_file(
                io.BytesIO(img_data),
                mimetype='image/png'
            )

        # 2. 文件下载路由
        @self.app.route('/download/<file_id>')
        def download_file(file_id):
            if file_id not in self.file_cache:
                abort(404)
            file_info = self.file_cache[file_id]
            return send_file(
                io.BytesIO(file_info["data"]),
                mimetype=file_info["mime"],
                as_attachment=True,
                download_name=file_info["filename"]
            )

        # 3. 首页状态
        @self.app.route('/')
        def index():
            return (f"<h1>文件服务器运行中</h1>"
                    f"<li>图片缓存: {len(self.image_cache)}</li>"
                    f"<li>文件缓存: {len(self.file_cache)}</li>")

    def start(self):
        # 线程启动 Flask
        t = threading.Thread(
            target=lambda: self.app.run(host=self.host, port=self.port, threaded=True, debug=False, use_reloader=False),
            daemon=True
        )
        t.start()
        time.sleep(1)
        print(f"🚀 混合文件服务器已在端口 {self.port} 开启")

    def add_image(self, img_buffer: io.BytesIO) -> str:
        """ 存入内存图片并返回预览 URL """
        if len(self.image_cache) > 50:
            first_key = next(iter(self.image_cache))
            del self.image_cache[first_key]
            
        image_id = uuid.uuid4().hex
        img_buffer.seek(0)
        self.image_cache[image_id] = img_buffer.read()
        
        return f"http://{local_host}:{self.port}/image/{image_id}"

    def upload_local_file(self, local_path: str) -> str:
        """ 存入任意文件并返回下载 URL """
        if not os.path.exists(local_path):
            return f"错误：文件 {local_path} 不存在"

        # 缓存清理
        if len(self.file_cache) > 10:
            first_key = next(iter(self.file_cache))
            del self.file_cache[first_key]

        filename = os.path.basename(local_path)
        mime_type, _ = mimetypes.guess_type(local_path)
        mime_type = mime_type or "application/octet-stream"

        with open(local_path, "rb") as f:
            file_data = f.read()

        file_id = uuid.uuid4().hex
        self.file_cache[file_id] = {
            "data": file_data,
            "filename": filename,
            "mime": mime_type
        }

        return f"http://{local_host}:{self.port}/download/{file_id}"
# --- 使用示例 (配合 Matplotlib) ---

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. 启动服务器 (换成 8080 避开 6666 坑点)
    server = MemoryFileServer(port=6760)
    server.start()

    # 2. 模拟 Matplotlib 绘图并保存到 BytesIO
    plt.figure(figsize=(5, 4))
    plt.plot([1, 2, 3], [4, 5, 2], marker='o', color='r')
    plt.title("Memory Buffer Test")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close() # 释放绘图资源

    # 3. 生成 URL
    url = server.add_image(buf)
    print(f"\n🔗 图片已生成在内存中，请访问:\n{url}\n")

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("停止服务")
# if __name__ == "__main__":
#     server = ImageServer(port=6660)
#     server.start()
    
#     # 测试
#     test_img = "./figures/structures/sample.png" 
#     url = server.generate_url(test_img)
#     print(f"🔗 尝试在浏览器打开: {url}")
    
#     try:
#         while True: time.sleep(1)
#     except KeyboardInterrupt:
#         pass