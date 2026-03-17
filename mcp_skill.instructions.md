# MCP Agent Skill Documentation

这是一个基于你当前 `mpmcp.py` 工具集合的 agent skill 文档。它不直接执行代码，而是告诉 agent 如何调用你的 MCP 服务和主要工具。

## 1) Skill 目的
- 通过 MCP 服务执行材料数据查询、材料结构获取与可视化、远程 VASP 任务管理、计算结果提取等操作
- 在 agent 中作为“技能说明”被调用，帮你构建自然语言→MCP工具调用桥梁

## 2) MCP 服务地址
- 默认地址：`http://127.0.0.1:8000`
- 你需要先启动 `mpmcp.py`，然后 agent 才能调用工具

## 3) 工具（tools）列表与输入输出
你在 `mpmcp.py` 中已经定义了这些 `@mcp.tool()`：

- `get_time()` -> 返回当前时间字符串
- `get_material_project_page(material_id)` -> 返回 MP 网址
- `search_materials(...)` -> 通过 MP API 查询材料
- `get_band_gap(material_id)` -> 返回带隙
- `get_material_structure(material_id,get_sites,get_plot,download)` -> 返回晶体结构
- `build_structure(...)` -> 自定义晶体结构并可保存可视化链接
- `create_task(formula,cif_path)` -> 创建远程 VASP 任务
- `list_task_directories()` -> 列出任务目录
- `check_squeue()` -> 查询Slurm队列
- `submit_opt_mission(task_directory)` -> 提交优化任务
- `extract_opt_info(task_directory,get_plot,visualize)` -> 提取优化结果
- `submit_scf_mission(task_directory,custom_incar)` -> 提交SCF任务
- `extract_scf_info(task_directory)` -> 提取SCF结果
- `submit_band_mission(task_directory)` -> 提交能带
- `extract_band_info(task_directory,plot_band)` -> 提取能带结果
- `excute_command(command)` -> 远程执行命令（有安全过滤）

## 4) Agent 调用示例（建议在 skill 中用）

### Python 容器（可在 agent 所处环境）
```python
from fastmcp.client import Client

async def run_example():
    async with Client("http://127.0.0.1:8000") as c:
        # 1) 列出工具
        tools = await c.list_tools()
        print(tools)

        # 2) 获取时间
        out = await c.call_tool("get_time")
        print(out)

        # 3) 查询材料带隙
        gap = await c.call_tool("get_band_gap", {"material_id": "mp-149"})
        print(gap)

import asyncio
asyncio.run(run_example())
```

### 自定义 agent 
- 如果你用 agent 框架（如 LangChain + FastMCP）
- 把この技能说明当作 prompt 或 `skill` 文档，agent 通过 `client.call_tool(...)` 调用对应工具

## 5) 推荐最佳实践
1. 先调用 `get_time()` 或 `list_tools()` 测试服务是否可用。
2. 关键函数传参最好写成 JSON 字典。
3. 任务流：`create_task` -> `submit_opt_mission` -> `extract_opt_info` -> `submit_scf_mission` -> `extract_scf_info` -> `submit_band_mission` -> `extract_band_info`。
4. 若要可视化，设置 `get_plot=True` 或 `visualize=True`。

## 6) 可能遇到的问题
- `ConnectError`：确认 `mpmcp.py` 已运行且 `127.0.0.1:8000` 可用
- `API key` 未设置：需在 `.env`/环境变量中配置 `MP_API_KEY`

---

你现在已经有了一个完整的 agent skill 文档（MD），可直接给你的 agent 或对话模型使用。