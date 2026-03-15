import os, loadenv
import io,time
import sys
import uuid
import threading
import webbrowser
from collections import deque
from flask import Flask, render_template_string, send_file, jsonify
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


config = loadenv.Config()
local_host = config.get_ip()


app = Flask(__name__)

# 配置
MAX_STRUCTURES = 10
STRUCTURE_STORAGE = {}
STRUCTURE_QUEUE = deque(maxlen=MAX_STRUCTURES)

# --- 核心逻辑：修复后的 HTML 模板 ---
COMPLETE_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{{ info.reduced_formula }} - 可视化</title>
    <style>
        body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f0f2f5; color: #333; overflow: hidden; }
        .container { max-width: 1600px; margin: 15px auto; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden; height: calc(100vh - 30px); display: flex; flex-direction: column; }
        
        .header { background: #2c3e50; color: white; padding: 12px 25px; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }
        
        /* 主布局：左侧固定，右侧滚动 */
        .content { display: grid; grid-template-columns: 1fr 420px; gap: 0; flex-grow: 1; overflow: hidden; }
        
        /* 左侧 3D 区域 */
        .vis-container { background: #fff; border-right: 1px solid #eee; height: 100%; }
        
        /* 右侧侧边栏：整体滚动 */
        .info-section { 
            padding: 20px; 
            overflow-y: auto; /* 允许纵向滚动 */
            background: #fafafa;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .card { background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 18px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
        .card h3 { margin: 0 0 15px 0; color: #3498db; font-size: 1.1em; border-left: 4px solid #3498db; padding-left: 12px; }
        
        .data-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .data-item { background: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #f0f0f0; }
        .label { font-size: 0.75em; color: #7f8c8d; display: block; margin-bottom: 4px; }
        .value { font-weight: 600; font-family: 'Consolas', monospace; font-size: 0.95em; color: #2c3e50; }
        
        table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
        th { text-align: left; color: #7f8c8d; padding: 10px 8px; border-bottom: 2px solid #eee; }
        td { padding: 10px 8px; border-bottom: 1px solid #f5f5f5; color: #444; }

        .btn-cif { background: #3498db; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13px; font-weight: 600; transition: background 0.2s; }
        .btn-cif:hover { background: #2980b9; }

        /* 美化滚动条 */
        .info-section::-webkit-scrollbar { width: 6px; }
        .info-section::-webkit-scrollbar-thumb { background: #ccc; border-radius: 10px; }
        .info-section::-webkit-scrollbar-track { background: transparent; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h2 style="margin:0; font-size: 1.4em;">{{ info.reduced_formula }}</h2>
                <span style="font-size: 0.85em; opacity: 0.8;">空间群: {{ info.space_group_symbol }} (No. {{ info.space_group_number }})</span>
            </div>
            <a href="/download_cif/{{ info.id }}" class="btn-cif">📄 下载 CIF 文件</a>
        </div>

        <div class="content">
            <div class="vis-container">
                <iframe src="/get_3d_html/{{ info.id }}" style="width:100%; height:100%; border:none;"></iframe>
            </div>

            <div class="info-section">
                <div class="card">
                    <h3>晶格参数</h3>
                    <div class="data-grid">
                        <div class="data-item"><span class="label">a (Å)</span><span class="value">{{ info.lattice.a }}</span></div>
                        <div class="data-item"><span class="label">b (Å)</span><span class="value">{{ info.lattice.b }}</span></div>
                        <div class="data-item"><span class="label">c (Å)</span><span class="value">{{ info.lattice.c }}</span></div>
                        <div class="data-item"><span class="label">α (°)</span><span class="value">{{ info.lattice.alpha }}</span></div>
                        <div class="data-item"><span class="label">β (°)</span><span class="value">{{ info.lattice.beta }}</span></div>
                        <div class="data-item"><span class="label">γ (°)</span><span class="value">{{ info.lattice.gamma }}</span></div>
                        <div class="data-item"><span class="label">体积 (Å³)</span><span class="value">{{ info.lattice.volume }}</span></div>
                        <div class="data-item"><span class="label">密度 (g/cm³)</span><span class="value">{{ info.density }}</span></div>
                    </div>
                </div>

                <div class="card">
                    <h3>基本信息</h3>
                    <div class="data-grid">
                        <div class="data-item"><span class="label">化学式</span><span class="value">{{ info.reduced_formula }}</span></div>
                        <div class="data-item"><span class="label">完整化学式</span><span class="value">{{ info.full_formula }}</span></div>
                        <div class="data-item"><span class="label">空间群</span><span class="value">{{ info.space_group_symbol }}</span></div>
                        <div class="data-item"><span class="label">空间群编号</span><span class="value">{{ info.space_group_number }}</span></div>
                        <div class="data-item"><span class="label">原子总数</span><span class="value">{{ info.num_sites }}</span></div>
                        <div class="data-item"><span class="label">是否有序</span><span class="value">{{ '是' if info.is_ordered else '否' }}</span></div>
                </div>

                <div class="card">
                    <h3>原子位点 (Sites)</h3>
                    <table>
                        <thead><tr><th>元素</th><th>分数坐标 (x, y, z)</th></tr></thead>
                        <tbody>
                            {% for site in info.sites %}
                            <tr><td><b>{{ site.element }}</b></td><td>{{ site.coords|join(', ') }}</td></tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# --- 路由与逻辑 ---

@app.route('/view/<struct_id>')
def view_structure(struct_id):
    if struct_id not in STRUCTURE_STORAGE:
        return "<h3>Error 404: 结构已失效</h3><p>可能因为加载了太多新结构，旧结构已被清理。</p>", 404
    
    data = STRUCTURE_STORAGE[struct_id]
    s = data['structure']
    lat = s.lattice
    sg = s.get_space_group_info()
    
    info = {
        'id': struct_id,
        'reduced_formula': s.reduced_formula,
        'full_formula': s.formula,
        'space_group_symbol': sg[0],
        'space_group_number': sg[1],
        'num_sites': len(s),
        'is_ordered': s.is_ordered,
        'density': round(s.density, 4),
        'lattice': {
            'a': round(lat.a, 4), 'b': round(lat.b, 4), 'c': round(lat.c, 4),
            'alpha': round(lat.alpha, 2), 'beta': round(lat.beta, 2), 'gamma': round(lat.gamma, 2),
            'volume': round(lat.volume, 2)
        },
        'sites': [{'element': str(site.specie), 'coords': [round(c, 4) for c in site.frac_coords]} for site in s.sites]
    }
    return render_template_string(COMPLETE_TEMPLATE, info=info)

@app.route('/get_3d_html/<struct_id>')
def get_3d_html(struct_id):
    path = STRUCTURE_STORAGE.get(struct_id, {}).get('vis_path')
    if path and os.path.exists(path):
        return send_file(os.path.abspath(path))
    return "3D File Not Found", 404

@app.route('/download_cif/<struct_id>')
def download_cif(struct_id):
    struct = STRUCTURE_STORAGE.get(struct_id, {}).get('structure')
    if not struct: return "Not Found", 404
    sio = io.BytesIO(str(CifWriter(struct)).encode())
    return send_file(sio, as_attachment=True, download_name=f"{struct.reduced_formula}.cif")

# --- 管理器 ---

class CrystalManager:
    _server_started = False
    _port = 6750

    def __init__(self):
        """初始化时仅负责启动后台服务"""
        if not CrystalManager._server_started:
            # 启动 Flask 守护线程
            thread = threading.Thread(
                target=lambda: app.run(host="0.0.0.0", port=CrystalManager._port, debug=False, use_reloader=False),
                daemon=True
            )
            thread.start()
            CrystalManager._server_started = True
            # 给服务一点启动时间，防止第一次调用 show 太快导致 404
            time.sleep(0.5) 
            print(f"🚀 可视化后台服务已在端口 {CrystalManager._port} 开启")

    def show(self, structure, html_file_path, auto_open=False):
        """注册结构并生成访问链接"""
        # 1. 生成唯一 ID
        structure_id = str(uuid.uuid4())
        
        # 2. 内存清理机制
        if len(STRUCTURE_QUEUE) >= MAX_STRUCTURES:
            old_id = STRUCTURE_QUEUE.popleft() # deque 建议用 popleft() 更高效
            STRUCTURE_STORAGE.pop(old_id, None)
        
        # 3. 存储数据
        STRUCTURE_STORAGE[structure_id] = {
            'structure': structure, 
            'vis_path': html_file_path
        }
        STRUCTURE_QUEUE.append(structure_id)
        
        # 4. 生成链接
        url = f"http://{local_host}:{CrystalManager._port}/view/{structure_id}"
        print(f"✅ 结构已就绪 [{structure.reduced_formula}]: {url}")
        
        # 5. 自动打开浏览器
        if auto_open:
            webbrowser.open(url)
            
        return url


# --- 测试样例 ---
if __name__ == '__main__':
    # 请确保路径正确
    cif = "cifs/La3S4-mp-567.cif"
    html = "cifs/images/La3S4-mp-567_3d.html"
    
    if os.path.exists(cif) and os.path.exists(html):
        struct = Structure.from_file(cif)
        # 实例化即运行，无需 .run()
        cm = CrystalManager()
        cm.show(struct, html)
        
        # 阻止主程序退出
        import time
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            print("退出中...")