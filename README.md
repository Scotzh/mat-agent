# MatAgent 工具使用指南

本 README 介绍 `mpmcp.py` 中可用的每个工具函数、参数格式和推荐执行规范。请在启动 `mpmcp.py` 并正确设置 MP_API_KEY 后调用。

## 1. 基础辅助工具

### `get_time()`
- 功能：获取当前系统时间
- 输入：无
- 返回：`YYYY-MM-DD HH:MM:SS` 字符串
- 场景：记录实验日志、任务时间点。

### `get_material_project_page(material_id: str)`
- 功能：生成 Materials Project 页面链接
- 参数：
  - `material_id`：例如 `mp-149`
- 返回：`{"material_id":..., "url":..., "message":...}`

## 2. MP 材料查询与信息获取

### `search_materials(...)`
- 功能：调用 MP API 搜索材料
- 关键参数：
  - `elements: list[str] | None`：包含元素列表，例如 `['Li', 'Cl']`
  - `exclude_elements: list[str] | None`：排除元素
  - `chemsys: str | list[str] | None`：化学系统，如 `Li-Fe-O` 或 `['Li-O', 'Fe-P']`
  - `band_gap: tuple[float,float] | None`：带隙范围 (min, max)
  - `num_elements: tuple[int,int] | None`：元素数范围
  - `formula: str | list[str] | None`：公式或通配公式，例如 `Li*Cl*`
  - `chunk_size: int | None`：返回数量，默认 25，最大 1000
- 返回：材料列表字典（material_id, formula_pretty, band_gap, symmetry）
- 注意：需要 `MP_API_KEY` 环境变量

### `get_band_gap(material_id: str)`
- 功能：获取指定材料的带隙
- 输入：`material_id`（如 `mp-1234`）
- 返回：`{"material_id":..., "band_gap":..., "formula":...}` or 错误信息

### `get_material_structure(material_id: str, get_sites=False, get_plot=False, download=False)`
- 功能：获取结构并可选生成 CIF/图
- 参数：
  - `material_id`: MP ID
  - `get_sites`: 是否返回原子位置信息
  - `get_plot`: 是否生成3D可视化网页链接
  - `download`: 是否保存 CIF 文件到 `cifs/`
- 返回：结构信息、CIF 路径、图片 URL（取决于参数）

### `get_material_all_infomation_by_id(material_id: str)`
- 功能：获取指定材料的全量 MP 数据
- 输入：`material_id`
- 返回：材料全字段字典

## 3. 自定义结构建模与可视化

### `build_structure(...)`
- 功能：根据晶格参数和坐标构建晶体结构
- 参数：
  - `a,b,c,alpha,beta,gamma`：晶格常数与角度
  - `elements: list[str]`：元素列表
  - `frac_coord: list[list[float]]`：对应分数坐标
  - `scaling_matrix: int | list = 1`：超胞扩展因子
  - `save_to_cif: bool = False`：是否保存 CIF
  - `add_to_database: str | None`：是否写入数据库
- 返回：`structure_dict`、可视化 URL、图片 URL

### `get_structure_plot(structure: Structure, repeat=True, rotation='10x,10y,0z')`
- 功能：为给定 pymatgen `Structure` 生成 2D 图片并上传到 image server
- 返回：`{"Image": URL, "error": None}`

### `visualize_structure(structure: Structure)`
- 功能：生成 ASE HTML 3D 交互网页，并返回临时链接

## 4. 远程 VASP 任务管理（SSH 集成）
> 这些工具依赖 `tryssh.VaspTaskInitializer(HOST, USERNAME, PASSWORD, PORT)` 实例 `connection`。

### `create_task(formula: str, cif_path: str)`
- 功能：在远程服务器上创建任务目录并上传 CIF
- 输入：材料公式和本地 CIF 文件路径
- 返回：任务状态字典

### `list_task_directories()`
- 功能：列出远程任务目录
- 返回：目录列表

### `check_squeue()`
- 功能：检查远程 Slurm 任务队列状态
- 返回：队列信息

### `submit_opt_mission(task_directory: str)`
- 功能：提交结构优化任务

### `extract_opt_info(task_directory: str, get_plot=True, visualize=False)`
- 功能：提取结构优化结果，支持绘图和可视化

### `submit_scf_mission(task_directory: str, custom_incar: dict = None)`
- 功能：提交 SCF 任务，可传 `custom_incar` 覆盖默认 INCAR

### `extract_scf_info(task_directory: str)`
- 功能：提取自洽计算结果

### `submit_band_mission(task_directory: str)`
- 功能：提交能带计算任务

### `extract_band_info(task_directory: str, plot_band=True)`
- 功能：提取能带结果并（可选）绘图

### `excute_command(command: str)`
- 功能：在远程服务器执行 Linux 命令（仅限安全命令）
- 返回：命令输出

## 5. 机器学习预测

### `predict_band_gap(formula: str | list[str])`
- 功能：调用内置 ML 模型预测带隙
- 输入：公式或公式列表
- 返回：预测结果

## 6. 项目流程和进度管理

### `set_task_progress(project_name: str, description: str = "", step_name: str = "", status: str = "")`
- 功能：记录/更新项目流程状态，写入 `material_workflow.json`
- 参数：项目名、描述、步骤名、状态（Pending/Running/Completed/Failed）

### `list_all_projects()`
- 功能：列出当前项目名称

### `get_project_workflow(project_name: str)`
- 功能：查看指定项目任务进度详情

### `read_file(file_path: str)`
- 功能：读取服务器文件内容（会进行异常处理）
- 返回：`{"success":True,"content":...}` 或错误信息

## 7. 推荐执行规范（SOP）
1. **初始化**：新材料或项目前先 `list_all_projects()`，若无则 `set_task_progress(..., step_name='Init', status='Pending')`。
2. **检索优先**：先 `search_materials(...)` 或 `get_material_structure(...)` 获取候选结构，再决定是否 `build_structure(...)`。
3. **提交任务**：每次 `submit_*_mission` 后，立即 `set_task_progress(..., status='Running')`。
4. **检查进度**：任务期间定时 `check_squeue()`，如发现失败或超时，设置对应步骤 `Failed` 并记录原因。
5. **结果入库**：`extract_*_info(...)` 成功后更新 `Completed` 并记录关键结果（例如能量、带隙）。
6. **异常处理**：若某步抛错，返回错误字典并建议“检查 API key、SSH 连接、文件路径、INCAR 参数”。


祝你使用顺利！
