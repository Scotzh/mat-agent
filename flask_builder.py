from flask import Flask, render_template, request, jsonify, send_file
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import json
import os
import threading
import time
import signal
import sys
import tempfile
import io

class CrystalStructureVisualizer:
    def __init__(self, structure, html_file_path):
        """
        初始化晶体结构可视化器
        
        参数:
            structure: pymatgen的Structure对象
            html_file_path: 3D可视化HTML文件的路径
        """
        self.structure = structure
        self.html_file_path = html_file_path
        self.app = Flask(__name__)
        self.shutdown_flag = False
        
        # 设置路由
        self.app.route('/')(self.index)
        self.app.route('/get_3d_html')(self.get_3d_html)
        self.app.route('/shutdown', methods=['POST'])(self.shutdown)
        self.app.route('/get_cif_text')(self.get_cif_text)
        self.app.route('/download_cif')(self.download_cif)
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)

    def create_structure_visualization_page(self):
        """
        创建显示3D结构可视化和结构数据的网页，包含结束进程按钮
        """
        # 提取结构信息
        lattice = self.structure.lattice
        space_group_info = self.structure.get_space_group_info()
        
        structure_info = {
            'formula': self.structure.formula,
            'reduced_formula': self.structure.reduced_formula,
            'space_group_symbol': space_group_info[0] if space_group_info else "未知",
            'space_group_number': space_group_info[1] if space_group_info else "未知",
            'lattice_parameters': {
                'a': round(lattice.a, 4),
                'b': round(lattice.b, 4),
                'c': round(lattice.c, 4),
                'alpha': round(lattice.alpha, 2),
                'beta': round(lattice.beta, 2),
                'gamma': round(lattice.gamma, 2),
                'volume': round(lattice.volume, 4)
            },
            'number_of_sites': len(self.structure),
            'density': round(self.structure.density, 4),
            'is_ordered': self.structure.is_ordered,
            'sites': [{'element': str(site.specie), 'frac_coords': [round(c, 4) for c in site.frac_coords]} for site in self.structure.sites]
        }
        
        # 生成HTML内容
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{structure_info['reduced_formula']} 晶体结构可视化</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
            position: relative;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }}
        
        .shutdown-btn {{
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s;
        }}
        
        .shutdown-btn:hover {{
            background: #c0392b;
        }}
        
        .shutdown-btn:disabled {{
            background: #95a5a6;
            cursor: not-allowed;
        }}
        
        .content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 30px;
        }}
        
        @media (max-width: 1200px) {{
            .content {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .visualization-section {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 0;  /* 修改为0，移除内边距 */
            height: 600px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);  /* 添加阴影效果 */
        }}
        
        .iframe-container {{
            width: 100%;
            height: 100%;
            border: none;
            border-radius: 8px;
            background: white;
            display: block;  /* 确保iframe是块级元素 */
        }}
        
        .info-section {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            overflow-y: auto;
            max-height: 600px;
        }}
        
        .section-title {{
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }}
        
        .info-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .info-item {{
            background: #e8f4fc;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .info-value {{
            color: #34495e;
            font-size: 1.1em;
        }}
        
        .iframe-container {{
            width: 100%;
            height: 100%;
            border: none;
            border-radius: 8px;
            background: white;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 5px;
        }}
        
        .badge-primary {{
            background: #3498db;
            color: white;
        }}
        
        .badge-success {{
            background: #27ae60;
            color: white;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }}
        
        .modal-content {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            max-width: 400px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{structure_info['reduced_formula']}</h1>
            <div class="subtitle">
                晶体结构可视化与详细信息
                <span class="badge badge-primary">空间群: {structure_info['space_group_symbol']}</span>
                <span class="badge badge-success">编号: {structure_info['space_group_number']}</span>
            </div>
            <button class="shutdown-btn" onclick="showShutdownConfirmation()">
                🔴🔴 结束进程
            </button>
            <button class="shutdown-btn" onclick="showShutdownConfirmation()">
                    🔴🔴 结束进程
                </button>
                <button class="shutdown-btn" style="right:160px; background:#2d9cdb;" onclick="openCifModal()">
                    📄 查看 / 下载 CIF
                </button>
        <div class="content">
            <div class="visualization-section">
                <div class="section-title">3D 结构可视化</div>
                <iframe src="/get_3d_html" class="iframe-container" 
                        title="3D Crystal Structure Visualization"></iframe>
            </div>
            
            <div class="info-section">
                <div class="section-title">结构信息</div>
                
                <div class="info-card">
                    <h3>晶格参数</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <div class="info-label">a (Å)</div>
                            <div class="info-value">{structure_info['lattice_parameters']['a']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">b (Å)</div>
                            <div class="info-value">{structure_info['lattice_parameters']['b']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">c (Å)</div>
                            <div class="info-value">{structure_info['lattice_parameters']['c']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">α (°)</div>
                            <div class="info-value">{structure_info['lattice_parameters']['alpha']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">β (°)</div>
                            <div class="info-value">{structure_info['lattice_parameters']['beta']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">γ (°)</div>
                            <div class="info-value">{structure_info['lattice_parameters']['gamma']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">体积 (Å³)</div>
                            <div class="info-value">{structure_info['lattice_parameters']['volume']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">密度 (g/cm³)</div>
                            <div class="info-value">{structure_info['density']}</div>
                        </div>
                    </div>
                </div>
                
                <div class="info-card">
                    <h3>基本信息</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <div class="info-label">化学式</div>
                            <div class="info-value">{structure_info['reduced_formula']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">完整化学式</div>
                            <div class="info-value">{structure_info['formula']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">空间群</div>
                            <div class="info-value">{structure_info['space_group_symbol']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">空间群编号</div>
                            <div class="info-value">{structure_info['space_group_number']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">原子总数</div>
                            <div class="info-value">{structure_info['number_of_sites']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">是否有序</div>
                            <div class="info-value">{'是' if structure_info['is_ordered'] else '否'}</div>
                        </div>
                    </div>
                </div>
                <div class="info-card">
                    <h3>原子位点信息</h3>
                    <!-- 新增 sites 信息展示 -->
                    <div style="margin-top:20px;">
                        <h3 style="color:black;">原子位点 (sites)</h3>
                        <table style="width:100%; border-collapse:collapse; color:black; font-size:1em;">
                            <thead>
                                <tr style="background:#e8f4fc;">
                                    <th style="padding:6px; border-bottom:1px solid #ccc;">元素</th>
                                    <th style="padding:6px; border-bottom:1px solid #ccc;">分数坐标 (x, y, z)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([
                                    f"<tr><td style='padding:6px; border-bottom:1px solid #eee;'>{site['element']}</td>"
                                    f"<td style='padding:6px; border-bottom:1px solid #eee;'>({site['frac_coords'][0]}, {site['frac_coords'][1]}, {site['frac_coords'][2]})</td></tr>"
                                    for site in structure_info['sites']
                                ])}
                            </tbody>
                        </table>
                    </div>
                </div>
        </div>
    </div>

    <!-- 确认关闭模态框 -->
    <div id="shutdownModal" class="modal">
        <div class="modal-content">
            <h3 style="color:black;">确认结束进程</h3>
            <p style="color:black;" >确定要结束当前服务进程吗？</p>
            <div style="margin-top: 20px;">
                <button onclick="shutdownServer()" style="
                    padding: 10px 20px;
                    background: #e74c3c;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    margin-right: 10px;
                ">确认结束</button>
                <button onclick="hideShutdownConfirmation()" style="
                    padding: 10px 20px;
                    background: #95a5a6;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                ">取消</button>
            </div>
        </div>
    </div>

    <!-- CIF 查看/下载模态框 -->
                <div id="cifModal" class="modal">
                    <div class="modal-content" style="max-width:800px; text-align:left;">
                        <h3 style="color:black; margin-top:0;">CIF 文件内容</h3>
                        <div style="margin-bottom:10px;">
                            <button onclick="copyCif()" style="
                                padding:8px 14px;
                                background:#27ae60;
                                color:white;
                                border:none;
                                border-radius:5px;
                                cursor:pointer;
                                margin-right:10px;
                            ">复制到剪贴板</button>
                            <button onclick="downloadCif()" style="
                                padding:8px 14px;
                                background:#3498db;
                                color:white;
                                border:none;
                                border-radius:5px;
                                cursor:pointer;
                            ">下载 CIF 文件</button>
                            <button onclick="closeCifModal()" style="
                                padding:8px 10px;
                                background:#95a5a6;
                                color:white;
                                border:none;
                                border-radius:5px;
                                cursor:pointer;
                                float:right;
                            ">关闭</button>
                        </div>
                        <pre id="cifContent" style="background:#fff; color:black; padding:12px; border-radius:6px; max-height:400px; overflow:auto; white-space:pre-wrap;"></pre>
                    </div>
                </div>
    <script>
        function showShutdownConfirmation() {{
            document.getElementById('shutdownModal').style.display = 'flex';
        }}
        
        function openCifModal() {{
            document.getElementById('cifModal').style.display = 'flex';
            const pre = document.getElementById('cifContent');
            pre.textContent = '加载中...';
            fetch('/get_cif_text')
                .then(r => r.json())
                .then(data => {{
                    if (data.cif) {{
                        pre.textContent = data.cif;
                    }} else {{
                        pre.textContent = '获取 CIF 失败: ' + (data.error || '未知错误');
                    }}
                }})
                .catch(err => {{
                    pre.textContent = '网络错误: ' + err;
                }});
        }}
        function closeCifModal() {{
            document.getElementById('cifModal').style.display = 'none';
        }}
        function copyCif() {{
            const text = document.getElementById('cifContent').innerText;
            navigator.clipboard.writeText(text).then(() => {{
                alert('已复制到剪贴板');
            }}).catch(err => {{
                alert('复制失败: ' + err);
            }});
        }}
        function downloadCif() {{
            // 直接跳转到下载路由，浏览器会触发下载
            window.location.href = '/download_cif';
        }}
        
        function hideShutdownConfirmation() {{
            document.getElementById('shutdownModal').style.display = 'none';
        }}
        
        function shutdownServer() {{
            const btn = document.querySelector('.shutdown-btn');
            btn.disabled = true;
            btn.textContent = '正在结束...';
            
            fetch('/shutdown', {{ method: 'POST' }})
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        document.body.innerHTML = `
                            <div style="
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                height: 100vh;
                                flex-direction: column;
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                color: white;
                                text-align: center;
                            ">
                                <h1>进程已结束</h1>
                                <p>服务已安全关闭，您可以关闭浏览器窗口。</p>
                                <p style="margin-top: 20px; font-size: 0.9em; opacity: 0.8;">
                                    如果浏览器窗口没有自动关闭，请手动关闭。
                                </p>
                            </div>
                        `;
                        
                        // 3秒后尝试关闭窗口
                        setTimeout(() => {{
                            window.close();
                        }}, 3000);
                    }} else {{
                        alert('结束进程失败: ' + data.message);
                        btn.disabled = false;
                        btn.textContent = '🔴🔴 结束进程';
                    }}
                }})
                .catch(error => {{
                    alert('网络错误: ' + error);
                    btn.disabled = false;
                    btn.textContent = '🔴🔴 结束进程';
                }});
                
            hideShutdownConfirmation();
        }}
    </script>
</body>
</html>
        """
        
        return html_content

    def index(self):
        """主页面 - 显示结构可视化"""
        html_content = self.create_structure_visualization_page()
        return html_content

    def get_3d_html(self):
        """提供3D HTML文件"""
        return send_file(self.html_file_path)

    def get_cif_text(self):
        """返回 CIF 文本（用于在页面显示）"""
        try:
            # 生成临时 CIF 文件并读取
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.cif')
            tmp.close()
            CifWriter(self.structure).write_file(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                cif_text = f.read()
            os.unlink(tmp.name)
            return jsonify({'cif': cif_text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def download_cif(self):
        """触发 CIF 文件下载"""
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.cif')
            tmp.close()
            CifWriter(self.structure).write_file(tmp.name)
            with open(tmp.name, 'rb') as f:
                data = f.read()
            os.unlink(tmp.name)
            bio = io.BytesIO(data)
            bio.seek(0)
            filename = f"{self.structure.reduced_formula}.cif"
            return send_file(bio, as_attachment=True, download_name=filename, mimetype='chemical/x-cif')
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def shutdown(self):
        """结束进程的API端点"""
        try:
            self.shutdown_flag = True
            print("收到关闭请求，正在准备关闭服务...")
            
            # 使用线程来延迟关闭，确保响应先返回给客户端
            def delayed_shutdown():
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGINT)
            
            threading.Thread(target=delayed_shutdown).start()
            
            return jsonify({
                'success': True,
                'message': '进程将在几秒后关闭'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            }), 500

    def graceful_shutdown(self, signum, frame):
        """优雅关闭处理"""
        print("\n正在关闭服务...")
        sys.exit(0)

    def run(self, port=5000, debug=False):
        """
        运行Flask应用
        
        参数:
            port: 端口号，默认为5000
            debug: 是否启用调试模式，默认为False
        """
        print("启动Flask服务...")
        print(f"访问 http://localhost:{port}/ 查看结构")
        print("点击页面右上角的红色按钮可以结束进程")
        
        self.app.run(debug=debug, port=port, use_reloader=False)

# 使用示例
if __name__ == '__main__':
    # 检查文件是否存在
    if not os.path.exists("cifs/La3S4-mp-567.cif"):
        print("错误: CIF文件不存在!")
        print("请确保 cifs/La3S4-mp-567.cif 文件存在")
        sys.exit(1)
    
    if not os.path.exists("cifs/images/La3S4-mp-567_3d.html"):
        print("错误: 3D HTML文件不存在!")
        print("请确保 cifs/images/La3S4-mp-567_3d.html 文件存在")
        print("你可以先运行 generate_structure.py 来生成3D HTML文件")
        sys.exit(1)

    # 创建实例并运行
    structure = Structure.from_file("cifs/La3S4-mp-567.cif")
    visualizer = CrystalStructureVisualizer(structure, "cifs/images/La3S4-mp-567_3d.html")
    visualizer.run()