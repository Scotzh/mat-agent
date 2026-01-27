MaterialDataServer 工具使用指南
1. 基础信息类
get_time

功能：获取当前的系统时间。

场景：记录实验日志或计算任务的启动时间。

get_material_project_page

功能：生成指向 Materials Project 官网特定材料详情页的 URL。

输入：material_id (如 "mp-149")。

2. 数据库检索与结构获取
search_materials (核心搜索工具)

功能：在 MP 数据库中筛选符合条件的材料。

参数说明：

elements: 包含的元素列表，如 ["Li", "Cl"]。

exclude_elements: 排除的元素，如 ["O"]。

band_gap: 带隙范围元组，如 (0.5, 3.0)。

formula: 支持通配符，如 Li*Cl*。

注意：单次最多返回 25 条数据。

get_material_structure

功能：提取材料的晶体学信息。

参数说明：

get_sites: 设为 True 时返回所有原子的分数坐标。

get_plot: 设为 True 时会启动后台进程，生成 3D 交互网页链接。

download: 设为 True 会将结构保存为本地 cifs/ 目录下的 .cif 文件。

get_material_all_infomation_by_id

功能：获取该材料在 MP 数据库中的全量原始数据（字典格式）。

get_band_gap

功能：快速查询特定 material_id 的带隙值。

3. 结构建模与设计
build_structure (自定义建模)

功能：根据你设计的化学式和晶格参数手动构建一个全新的晶体。

必填项：

晶格参数：a, b, c, alpha, beta, gamma。

原子信息：elements (元素列表) 和 frac_coord (对应坐标列表)。

场景：当你通过 AI 迭代出新的卤化物电解质配方时，用此工具生成模型。

4. 远程 VASP 计算管理 (SSH 集成)
create_task

功能：在远程计算集群上创建任务文件夹，并将本地 CIF 文件上传过去。

流程：这是所有 VASP 计算的第一步。

check_squeue

功能：查看服务器上排队或运行中的任务状态。

submit_opt_mission / submit_scf_mission

功能：提交结构优化 (Opt) 或静态自洽 (SCF) 计算。

注意：SCF 支持 custom_incar 参数，你可以通过字典修改计算精度或收敛标准。

extract_opt_info / extract_scf_info

功能：计算完成后，提取能量、力、费米能级或优化后的新结构。

自动可视化：extract_opt_info 默认会将优化后的结构再次生成 3D 预览图。

5. 机器学习预测
predict_band_gap

功能：输入化学式（如 Li3InCl6），直接通过内置的 ML 模型返回预测带隙。

优势：毫秒级响应，无需提交 VASP 任务，适合初步筛选。

6. 任务清单管理
📋 任务清单管理工具 (Workload Management)该模块用于追踪材料研发项目的进度，确保跨会话的实验数据不丢失。1. list_all_projects功能：获取当前数据库中记录的所有材料研发项目清单。使用场景：当用户询问“我最近在做什么项目”或“有哪些任务还没完成”时，首先调用此工具。输出：项目名称列表 ["Li3InCl6_SSE", "Halide_Project_A", ...]。2. get_project_workflow功能：查看特定材料项目的详细任务清单、每个步骤的状态及操作时间。参数：project_name (项目唯一标识符)。输出：包含步骤、状态（Pending/Running/Completed）和时间戳的字典。使用场景：在深入处理某个具体材料前，先确认它算到了哪一步。3. set_task_progress功能：创建新项目或更新现有项目的任务进度。参数：project_name: 项目名称。step_name: 具体步骤（如："Structure_Search", "VASP_Opt", "Bandgap_Pred"）。status: 状态更新（Pending, Running, Completed, Failed）。重要逻辑：当你完成一次 search_materials 或 build_structure 后，应自动更新进度。当你通过 submit_opt_mission 提交任务后，必须将状态设为 Running。当你通过 extract_opt_info 获取结果后，必须将状态设为 Completed。🤖 Agent 执行规范 (Operational Rules)为了表现得像一个专业的材料科学助理，请遵循以下 SOP（标准作业程序）：项目初始化：在用户提到一个新化学式（如 "帮我研究一下 $Li_2ZrCl_6$"）时，先尝试 list_all_projects 检查是否已有记录，若无则调用 set_task_progress 初始化项目。分步查看：不要试图一次性展示所有数据。先 list_all_projects 让用户选择项目，再根据名称调用 get_project_workflow 展示详情。自动同步：上传成功 $\rightarrow$ 更新步骤 "HPC_Upload" 为 Completed。计算中 $\rightarrow$ 调用 check_squeue 确认任务在排队后，更新 "VASP_Simulation" 为 Running。提取成功 $\rightarrow$ 更新状态并记录关键结果（如能量或带隙）。错误处理：如果 submit_opt_mission 报错，请将对应步骤标记为 Failed，并主动询问用户是否需要检查 INCAR 设置或远程服务器连接。