import os, loadenv
import uuid
import time
import shutil
import threading
from flask import Flask, send_from_directory, send_file, abort
import io


config = loadenv.Config()
local_host = config.get_ip()

class ImageServer:
    def __init__(self, port=8080, folder="web_cache"):
        self.port = port
        self.upload_folder = os.path.abspath(folder)
        self.app = Flask(__name__)
        
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)

        @self.app.route('/image/<filename>')
        def serve_image(filename):
            # 加上 mimetype 强制浏览器识别为图片
            return send_from_directory(self.upload_folder, filename, mimetype='image/png')

        @self.app.route('/')
        def index():
            return "<h1>WSL Image Server is Running</h1>"

    def start(self):
        # 关键修改：直接在主线程启动测试，或者确保 host 绑定正确
        # 这里使用 0.0.0.0 是为了让 Windows 能通过虚拟网卡 IP 访问
        def run_flask():
            self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
        
        self.thread = threading.Thread(target=run_flask, daemon=True)
        self.thread.start()
        time.sleep(2) # 给 WSL 更多响应时间
        print(f"🚀 WSL 内部服务已启动，端口: {self.port}")

    def generate_url(self, local_path):
        # ... (之前的复制逻辑保持不变)
        ext = os.path.splitext(local_path)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        shutil.copy2(local_path, os.path.join(self.upload_folder, unique_name))
        
        # 重点：在 WSL 环境下，建议直接尝试 localhost，
        # 如果 localhost 不行，再手动换成终端显示的 172.x.x.x IP
        return f"http://{local_host}:{self.port}/image/{unique_name}"



class MemoryImageServer:
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        # 用字典在内存里存图片数据: { "uuid": b'binary_data' }
        self.image_cache = {}
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/image/<image_id>')
        def serve_image(image_id):
            if image_id not in self.image_cache:
                abort(404)
            
            # 从内存中读取二进制流并返回
            img_data = self.image_cache[image_id]
            return send_file(
                io.BytesIO(img_data),
                mimetype='image/png',
                download_name=f"{image_id}.png"
            )

        @self.app.route('/')
        def index():
            count = len(self.image_cache)
            return f"<h1>内存图片服务器</h1><p>当前缓存图片数: {count}</p>"

    def start(self):
        t = threading.Thread(
            target=lambda: self.app.run(host=self.host, port=self.port, threaded=True, debug=False, use_reloader=False),
            daemon=True
        )
        t.start()
        time.sleep(1)
        print(f"🚀 内存图片服务器已在端口 {self.port} 开启")

    def add_image(self, img_buffer: io.BytesIO) -> str:
        """
        输入一个 BytesIO 对象，存入内存并返回 URL
        """

        # 如果缓存超过 100 张，删除最早的一张（先进先出）
        if len(self.image_cache) > 50:
            first_key = next(iter(self.image_cache))
            del self.image_cache[first_key]
        image_id = uuid.uuid4().hex
        # 获取二进制数据存入字典
        img_buffer.seek(0)
        self.image_cache[image_id] = img_buffer.read()
        
        # 返回访问链接
        return f"http://{local_host}:{self.port}/image/{image_id}"

# --- 使用示例 (配合 Matplotlib) ---

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. 启动服务器 (换成 8080 避开 6666 坑点)
    server = MemoryImageServer(port=6760)
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