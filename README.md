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