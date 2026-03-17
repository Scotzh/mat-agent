from datetime import datetime

from fastmcp import FastMCP
import asyncio
import os
import pandas as pd
from mp_api.client import MPRester
from pydantic import BaseModel
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import matplotlib.pyplot as plt
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from ase.visualize import view
import multiprocessing
import flask_builder
import flask_plot
import duckdb
import pickle
import loadenv
import databasemanage
import tryssh
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams["font.family"] = ["serif"]  # 指定默认字体为SimHei
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像时负号'-'显示为方块的问题
mcp = FastMCP(name="MatAgent")
import tempfile
import shutil
import atexit
import signal
# ...existing code...

# 全局子进程列表（初始化），存储元组 (Process, temp_dir)
child_processes: list[tuple[multiprocessing.Process, str]] = []

def cleanup_child_processes():
    """在主进程退出时尝试优雅终止所有子进程并删除临时文件目录"""
    for p, temp_dir in list(child_processes):
        try:
            if p.is_alive():
                p.terminate()
                p.join(3)  # 等待 3 秒优雅退出
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
                    p.join(1)
        except Exception:
            pass
        # 删除临时目录（如果存在）
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            child_processes.remove((p, temp_dir))
        except ValueError:
            pass

# 注册退出清理
atexit.register(cleanup_child_processes)

# 响应终止信号时也清理
def _handle_exit(signum, frame):
    cleanup_child_processes()
    os._exit(0)

signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)


config = loadenv.Config()
if not config.validate_config():
    raise EnvironmentError("请设置必要的环境变量")
MY_API_KEY = config.get_api_key()
IP = config.get_ip()
IMAGE_URL = "http://" + IP + ":5000"
HOST = config.get_host()
PORT = config.get_port()
USERNAME = config.get_username()
PASSWORD = config.get_password()


@mcp.tool()
async def get_time() -> str:
    """
    获取当前时间
    Returns:
        当前时间字符串
    """
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
async def get_material_project_page(material_id: str) -> str:
    """
    获取指定材料的Material Project页面链接"https://next-gen.materialsproject.org/materials/{material_id}/"
    
    Args:
        material_id: 材料ID (如"mp-1234")
            
    Returns:    
        Material Project页面链接
    """
    if not material_id:
        return {"error": "材料ID不能为空", "message": "请提供有效的材料ID"}
    
    # 构建Material Project页面链接
    url = f"https://next-gen.materialsproject.org/materials/{material_id}/"
    return {"material_id": material_id, "url": url, "message": f"获取材料 {material_id} 的Material Project页面链接成功"}

@mcp.tool()
async def search_materials(
    elements: list[str] | None = None,
    exclude_elements: list[str] | None = None,
    chemsys: str | list[str] | None = None,
    band_gap: tuple[float, float] | None = None,
    num_elements: tuple[int, int] | None = None,
    formula: str | list[str] | None = None,
    chunk_size: int | None = 25
) -> list[dict]:
    """
    Material Project数据查询工具,参数里不要加fields,查询的参数有元素符号,带隙范围和原子位点数范围,每次最多返回25条数据。
    
    Args:
        elements: 元素符号列表(如["O", "Si"])
        exclude_elements: 排除的元素符号列表(如["H"])
        chemsys: A chemical system or list of chemical systems (e.g., Li-Fe-O, Si-*, [Si-O, Li-Fe-P])
        band_gap: 带隙范围(如(0.0, 1.5))
        num_elements: 元素个数范围(如(1, 10))
        formula: A formula including anonymized formula or wild cards (e.g., Fe2O3, ABO3, Si*). A list of chemical formulas can also be passed (e.g., [Fe2O3, ABO3]).
        chunk_size: 每次查询返回的结果数量,默认25,最大1000
    Returns:
        材料的基本信息,
        返回的结果是一个字典列表,每个字典包含以下字段:
            - material_id: 材料ID (如"mp-1234")
            - formula_pretty: 美化后的化学式
            - band_gap: 带隙值
            - symmetry: 对称性信息

    """

    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("API密钥未设置")
    try:
        with MPRester(API_KEY) as mpr:
            # 使用正确的search参数格式
            criteria = {}
            if elements:
                criteria["elements"] = elements
            if exclude_elements:
                criteria["exclude_elements"] = exclude_elements
            if chemsys:
                criteria["chemsys"] = chemsys
            if band_gap:
                criteria["band_gap"] = band_gap
            if num_elements:
                criteria["num_elements"] = num_elements
            if formula:
                criteria["formula"] = formula
                
            results = mpr.summary.search(
                **criteria,
                fields=["material_id", "formula_pretty", "band_gap", "symmetry"],
                chunk_size = chunk_size if chunk_size and chunk_size <= 1000 else 25,
                num_chunks = 1
            )
            print(f"查询到 {len(results)} 个材料")
        return [{
        "material_id": r.material_id,
        "formula_pretty": r.formula_pretty,
        "band_gap": r.band_gap,
        "symmetry": r.symmetry,  # 对称性信息
        # 可扩展其他字段
    } for r in results]
    
    except Exception as e:
        return {"error": str(e), "message": "查询材料数据失败"}

@mcp.tool()
async def get_band_gap(material_id: str) -> dict:
    """
    获取指定材料的带隙值
    
    Args:
        material_id: 材料ID (如"mp-1234")
            
    Returns:
        材料的带隙值
    """ 
    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("MP_API_KEY环境变量未设置")
    try:
        with MPRester(API_KEY) as mpr:
            results = mpr.summary.search(
                material_ids=material_id,
                fields=["band_gap","formula_pretty"]
            )
            if not results:
                raise ValueError(f"未找到材料ID为 {material_id} 的材料")
            else:
                print(f"获取材料 {material_id} 的带隙值成功")
            band_gap = results[0].band_gap
            formula = results[0].formula_pretty
        return {"material_id": material_id, "band_gap": band_gap, "formula": formula}
    except Exception as e:
        return {"error": str(e), "message": f"获取材料 {material_id} 的带隙值失败"}


@mcp.tool()
async def get_material_structure(material_id: str, 
                                get_sites: bool = False,
                                get_plot: bool = False, 
                                download: bool = False)-> dict | list:
    """
    获取指定材料的晶体结构数据,并保存为CIF文件,生成晶体结构图
    
    Args:
        material_id: 材料ID (如"mp-1234")
        get_sites: 是否获取原子位点信息,默认False
        get_plot: 是否生成晶体结构图,默认False,如果你只想获取位点信息,可以设置为False
        download: 是否下载CIF文件,默认False
    
    Returns:
        材料的晶体结构,包括空间群符号、空间群编号、化学式等信息,以及CIF文件路径和图片路径,以及3d交互式网页地址
    """
    # 获取API密钥
    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("MP_API_KEY环境变量未设置")
    os.makedirs("cifs", exist_ok=True)
    os.makedirs("cifs/images", exist_ok=True)
    message = []
    # 执行MP API查询
    try:
        with MPRester(API_KEY) as mpr:
            structure = mpr.get_structure_by_material_id(material_id, conventional_unit_cell=True)
            lattice = structure.lattice
            space_group_info = structure.get_space_group_info()
            formula = structure.formula
            reduced_formula = structure.composition.reduced_formula
            structure_info = {
                'formula': formula,
                'reduced_formula': reduced_formula,
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
                'number_of_sites': len(structure),
                'density': round(structure.density, 4),
                'is_ordered': structure.is_ordered,
            }
            message.append(f"材料 {material_id} 的晶体结构信息: {structure_info}")
            if get_sites:
                structure_info['sites'] = [{
                    'element': site.species_string,
                    'fractional_coordinates': [round(coord, 4) for coord in site.frac_coords],
                } for site in structure.sites]
                message.append(f"材料 {material_id} 的原子位点信息已包含在返回结果中")
            # 保存CIF文件
            if download:
                CifWriter(structure).write_file(f"cifs/{reduced_formula}-{material_id}.cif")
                print(f"获取材料 {material_id} 的晶体结构成功，已保存为cif文件")
                message.append(f"材料 {material_id} 的晶体结构已保存为cif文件，路径为'cifs/{reduced_formula}-{material_id}.cif'")   
            # 生成晶体结构图
            if get_plot:
                structure_url = visualize_structure(structure)
                message.append("生成了2d结构预览图和3d可视化交互式网页，请点击查看晶体结构图")
                message.append(f"3d_image_url: {structure_url}")
                res = get_structure_plot(structure)
                if not res["error"]:
                    image = res["Image"]
                    return {"image_url": image, "structure_dict":structure_info, "message": message,
                }
                else:
                    message.append(res["error"])
                    return {"structure_dict":structure_info, "message": message}

        return {"structure_dict":structure_info, "message": message,
                }
    except Exception as e:
        return {"error": str(e), "message": f"获取材料 {material_id} 的晶体结构失败"}


@mcp.tool()
async def build_structure(a: float,
                          b: float,
                          c: float,
                          alpha: float,
                          beta: float,
                          gamma: float,
                          elements: list[str],
                          frac_coord: list[list[float]],
                          scaling_matrix: int | list = 1,
                          save_to_cif: bool = False,
                          add_to_database: str = None,) -> dict | list:
    """
    构建晶体结构并保存为CIF文件,生成晶体结构图
    
    Args:
        a: 晶格参数a
        b: 晶格参数b
        c: 晶格参数c
        alpha: 晶格参数alpha
        beta: 晶格参数beta
        gamma: 晶格参数gamma
        elements: 元素符号列表，有多少个原子就要写多少个 (如["Si","O", "O"])
        frac_coord: 分数坐标列表，与上面的原子一一对应(如[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]])
        scaling_matrix: 超胞，默认整数（int）：表示在 a, b, c 三个方向进行相同的扩胞。例如 scaling_matrix=2表示构建 2×2×2 的超胞。
                    列表（list）：长度为 3 的列表，分别表示 a, b, c 方向的扩胞倍数。例如 scaling_matrix=[2, 1, 1]表示构建 2×1×1 的超胞。
        save_to_cif: 是否保存cif，默认不保存，需要时开启
        add_to_database: 是否将结构添加到数据库,默认None
    """
    try:
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        structure = Structure(lattice, elements, frac_coord)
        structure = structure.make_supercell(scaling_matrix = scaling_matrix)
        formula = structure.composition.reduced_formula
        os.makedirs("custom_structures", exist_ok=True)
        os.makedirs("custom_structures/images", exist_ok=True)
        message = []
        current_date = datetime.now().strftime("%Y%m%d%H%M")
        if save_to_cif:
            CifWriter(structure).write_file(f"custom_structures/{formula}_custom_{current_date}.cif")
            message.append(f"自定义晶体结构已保存为 ./custom_structures/{formula}_custom_{current_date}.cif")
        structure_url =  visualize_structure(structure)
        message.append("3d晶体结构可视化交互式网页，请点击查看晶体结构图")

        if add_to_database:
            db = databasemanage.DatabaseManager(add_to_database)
            db.add_material(formula=formula, structure=structure, band_gap=None, material_id=None)
            db.close()
            message.append(f"自定义晶体结构已添加到数据库 {add_to_database}")

        res = get_structure_plot(structure)
        if not res["error"]:
            image = res["Image"]
            return {
                "image": image,
                "3d_image_url": structure_url,
                "message": message
                }
        else:
            message.append(res["error"])
            return {
                "3d_image_url": structure_url,
                "message": message
                }

    except Exception as e:
        return {"error": str(e), "message": "构建晶体结构失败"}


def visualize_structure(structure: Structure) -> str:
    """
    可视化晶体结构的3D交互式网页（使用临时文件，进程结束后自动删除）
    Args:
        structure: pymatgen Structure对象
    """
    formula = structure.composition.reduced_formula
    atoms = AseAtomsAdaptor.get_atoms(structure)

    # 生成临时目录并写入 HTML（临时目录在子进程结束或清理时删除）
    temp_dir = tempfile.mkdtemp(prefix=f"{formula}_custom_")
    html_path = os.path.join(temp_dir, f"{formula}_custom_3d.html")
    write(html_path, atoms, format='html')

    url = crystalmanager.show(structure, html_path)
    return url


import matplotlib.pyplot as plt
import io
from fastmcp.utilities.types import Image
from ase.build import bulk
from ase.visualize.plot import plot_atoms
from ase import Atoms
from itertools import product
import numpy as np
def get_structure_plot(structure: Structure,
                          repeat: bool = True, rotation: str ='10x,10y,0z') -> dict:
    """
    输入指定的晶体结构并返回预览图。
    参数:
    - struture: Pymatgen.Structure对象
    - repeat: 为了让图片看起来更像“晶格”，可以重复一下晶胞 (可选)
    - rotation: 观测角度默认为 '10x,10y,0z'
    """
    try:
        # 转换为 ASE Atoms 对象
        atoms = structure.to_ase_atoms()
        atoms.wrap()
        # if repeat:
        #     atoms = atoms.repeat((2, 2, 2))  # 2x2x2 supercell
        def _enhance_for_plot(atoms: Atoms, tolerance: float = 0.05) -> Atoms:
            """
            专门为可视化增强 Atoms：将边界原子复制到相对的边界、棱和顶点。
            """
            # 1. 获取原始信息
            cell = atoms.get_cell()
            scaled_positions = atoms.get_scaled_positions()
            symbols = atoms.get_chemical_symbols()
            
            new_scaled = []
            new_symbols = []
            
            # 2. 定义平移矢量 (0, 1) 的所有组合 (共8个方向，足以覆盖 1x1x1 晶格的所有边界)
            # 如果原原子在 0 附近，偏移 +1 就能补全对侧
            offsets = list(product([0, 1], repeat=3))
            
            for pos, symbol in zip(scaled_positions, symbols):
                # 检查该原子靠近哪些边界
                # near_zero[i] 为 True 表示该原子在第 i 维靠近 0
                near_zero = np.isclose(pos, 0, atol=tolerance)
                
                for off in offsets:
                    # 如果偏移量 off 在某个维度为 1，但原子在该维度并不靠近 0，则跳过
                    if any(o == 1 and not nz for o, nz in zip(off, near_zero)):
                        continue
                        
                    # 基础位置 (off 为 (0,0,0)) 已经在循环中包含
                    new_scaled.append(pos + off)
                    new_symbols.append(symbol)
                    
            # 3. 构建新对象
            # 保持原始 cell 不变，这样绘图软件能正确渲染晶格线
            enhanced = Atoms(symbols=new_symbols, 
                            scaled_positions=new_scaled, 
                            cell=cell, 
                            pbc=True) # 绘图时开启 PBC 通常效果更好
            return enhanced

        enhanced_atoms = _enhance_for_plot(atoms=atoms)
        # 使用 ASE 绘图
        fig, ax = plt.subplots(figsize=(16,16))
        
        # 使用 ASE 的 plot_atoms 函数
        plot_atoms(
            enhanced_atoms,
            ax,
            rotation=rotation,
            show_unit_cell=2,
        )
        
        # 分析对称性
        analyzer = SpacegroupAnalyzer(structure)
        spacegroup = analyzer.get_space_group_symbol()
        
        # 添加结构信息
        a, b, c = structure.lattice.a, structure.lattice.b, structure.lattice.c
        alpha, beta, gamma = structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma
        formula = structure.composition.formula
        
        info_text = (
            f"Formula: {formula}\n"
            f"Space group: {spacegroup}\n"
            f"Lattice parameters: a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å\n"
            f"Angles: α={alpha:.2f}°, β={beta:.2f}°, γ={gamma:.2f}°\n"
            f"Atoms in unit cell: {len(structure)}\n"
            f"Total atoms shown: {len(atoms)}"
        )
        
        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_axis_off()
        ax.set_title(f"Crystal Structure Visualization of {structure.composition.reduced_formula}", 
                    fontsize=14, fontweight='bold')
        
        # 保存到内存
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        return {"Image": get_plot_url(img_buffer), "error": None}
        
    except Exception as e:
        # 如果输入的参数不合法（如元素符号错误），返回一个简单的报错图或抛出异常
        return {"Image": get_plot_url(_create_error_image(f"构建失败: {str(e)}")), "error": e} 

def _create_error_image(error_message: str):
    """创建错误信息图片"""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, f"❌ {error_message}",
            ha='center', va='center', fontsize=12, color='red')
    ax.set_axis_off()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return img_buffer


def get_plot_url(img_buffer):
    return server.add_image(img_buffer)



@mcp.tool()
async def get_material_all_infomation_by_id(material_id: str) -> dict:
    """
    获取指定材料的所有信息
    
    Args:
        material_id: 材料ID (如"mp-1234")
    
    Returns:
        材料的所有信息
    """
    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("MP_API_KEY环境变量未设置")

    try:
        with MPRester(API_KEY) as mpr:
            # 获取材料的所有信息
            with mpr.materials as materials:
                material = materials.search(
                    material_ids = material_id)
                if not material:
                    raise ValueError(f"未找到材料ID为 {material_id} 的材料")
                else:
                    print(f"获取材料 {material_id} 的所有信息成功")
            material_dict = material[0]
        return material_dict
    except Exception as e:
        return {"error": str(e), "message": f"获取材料 {material_id} 的所有信息失败"}

# @mcp.tool()
# async def get_dos_by_material_id(material_id: str) -> dict:
#     """
#     获取指定材料的态密度数据
    
#     Args:
#         material_id: 材料ID (如"mp-1234")
    
#     Returns:
#         材料的态密度数据
#     """
#     API_KEY = MY_API_KEY
#     if not API_KEY:
#         raise ValueError("MP_API_KEY环境变量未设置")
#     os.makedirs("dos_plots", exist_ok=True)
#     try:
#         with MPRester(API_KEY) as mpr:
#             dos = mpr.get_dos_by_material_id(material_id)
#             dos_dict = dos.as_dict()
#         # 创建DOS绘图器
#         plotter = pymatgen.electronic_structure.plotter.DosPlotter()
        
#         # 添加总态密度
#         plotter.add_dos("Total DOS", dos)
        
#         # 添加轨道分解态密度
#         for orb, dos_obj in dos.get_spd_dos().items():
#             plotter.add_dos(f"{orb.name} DOS", dos_obj)
        
#         # 绘制图像
#         plt.figure(figsize=(10, 7))
#         ax = plotter.get_plot()
        
#         # 设置图像标题和标签
#         fermi_level = dos.efermi
#         plt.axvline(fermi_level, color='r', linestyle='--', alpha=0.8)
#         plt.title(f"{material_id} - Density of States", fontsize=16)
#         plt.xlabel("Energy (eV)", fontsize=14)
#         plt.ylabel("DOS (States/eV)", fontsize=14)
#         plt.legend(fontsize=12, loc='best')
#         plt.annotate(f'E$_F$ = {fermi_level:.3f} eV', 
#                      xy=(fermi_level, 0), 
#                      xytext=(fermi_level + 0.5, 0.5),
#                      arrowprops=dict(facecolor='red', shrink=0.05),
#                      fontsize=12)
#         plt.savefig(f"dos_plots/{material_id}-dos.png", dpi=300)
#         plt.close()  # 关闭图像以释放内存
#         dos_dict["file_path"] = f"dos_plots/{material_id}-dos.png"
#         print(f"获取材料 {material_id} 的态密度数据成功，已保存态密度图")
#         return dos_dict
#     except Exception as e:
#         return {"error": str(e), "message": f"获取材料 {material_id} 的态密度数据失败"}




# # 数据库模块
# @mcp.tool()
# async def add_material_to_database_by_material_id(material_id: str, database: str = "material_database.db") -> dict:
#     """
#     根据id将材料信息添加到数据库
    
#     Args:
#         material_id: 材料ID
#         database: 数据库文件名,默认material_database.db
#     """
#     # 获取API密钥
#     API_KEY = MY_API_KEY
#     if not API_KEY:
#         raise ValueError("MP_API_KEY环境变量未设置")

#     try:
#         with MPRester(API_KEY) as mpr:
#             results = mpr.summary.search(
#                 material_ids=material_id,
#                 fields=["formula_pretty", "band_gap", "structure"]
#             )
#             if not results:  # 修复：检查results是否为空
#                 raise ValueError(f"未找到材料ID为 {material_id} 的材料")
#             else:
#                 print(f"获取材料 {material_id} 的信息成功")
            
#             material_dict = dict(results[0])  # 直接使用results[0]
            
#         # 提取所需信息
#         formula = material_dict.get("formula_pretty", "")
#         band_gap = material_dict.get("band_gap", None)
#         structure = material_dict.get("structure", None)
        
#         db = databasemanage.DatabaseManager(database)
#         db.add_material(formula=formula, structure=structure, band_gap=band_gap, material_id=material_id)
#         db.close()
#         return {"message": f"材料 {material_id} 已成功添加到数据库{database}"}
#     except Exception as e:
#         return {"error": str(e), "message": f"添加材料 {material_id} 到数据库失败{database}"}

# @mcp.tool()
# async def list_all_materials_from_database(page:int = 1, database: str = "material_database.db") -> list:
#     """
#     从数据库中列出所有材料信息
#     Args:
#         page: 数据库页码,默认1
#         database: 数据库文件名,默认material_database.db
#     Returns:
#         所有材料的列表
#     """
#     try:
#         db = databasemanage.DatabaseManager(database)
#         results = db.list_all_materials_by_pages(page=page, page_size=10)
#         db.close()
#         return results
#     except Exception as e:
#         return {'error': str(e)}
    

# @mcp.tool()
# async def get_material_from_database_by_mpid(material_id: str, database: str = "material_database.db") -> dict:
#     """
#     根据材料的materials project ID从数据库中获取指定材料信息
    
#     Args:
#         material_id: 材料的materials project ID
#         database: 数据库文件名,默认material_database.db
#     Returns:
#         指定材料的信息
#     """
#     try:
#         db = databasemanage.DatabaseManager(database)
#         material = db.get_material_by_material_id(material_id)
#         db.close()
#         if not material:
#             return {"error": f"在{database}中未找到材料ID为 {material_id} 的材料", "message": f"请检查材料ID或数据库名称是否正确"}
#         return material
#     except Exception as e:
#         return {"error": str(e), "message": f"获取材料 {material_id} 的信息失败，请检查材料ID或数据库名称是否正确"}

# @mcp.tool()
# async def get_material_from_database_by_ID(ID: str, database: str = "material_database.db") -> dict:
#     """
#     根据ID从数据库中获取指定材料信息
    
#     Args:
#         ID: 材料在数据库中的ID
#         database: 数据库文件名,默认material_database.db
#     Returns:
#         指定材料的信息
#     """
#     try:
#         db = databasemanage.DatabaseManager(database)
#         material = db.get_material_by_ID(ID)
#         db.close()
#         if not material:
#             return {"error": f"在{database}中未找到材料ID为 {ID} 的材料", "message": f"请检查材料ID是否正确或数据库名称是否正确"}
#         return material
#     except Exception as e:
#         return {"error": str(e), "message": f"获取材料 {ID} 的信息失败，请检查材料ID或数据库名称是否正确"}


# @mcp.tool()
# async def get_material_from_database_by_elements(formula: str, 
#                                     database: str = "material_database.db", 
#                                     page: int = 1, 
#                                     page_size: int = 25) -> dict:
#     """
#     根据化学组成从数据库中获取指定材料信息
    
#     Args:
#         formula: 化学式字符串 (如"SiO2")
#         database: 数据库文件名,默认material_database.db
#         page: 页码,默认1
#         page_size: 每页数量,默认25 
#     Returns:
#         指定材料的信息
#     """
#     try:
#         db = databasemanage.DatabaseManager(database)
#         material = db.get_material_by_elements(formula, page, page_size)
#         db.close()
#         if not material:
#             return {"error": f"在{database}中，未找到材料组成为 {formula} 的材料", "message": f"请检查材料组成是否正确或数据库名称是否正确"}
#         return material
#     except Exception as e:
#         return {"error": str(e), "message": f"获取材料  {formula} 的信息失败，请检查数据库名称是否正确"}

# @mcp.tool()
# async def remove_material_from_database_by_ID(ID: str, database: str = "material_database.db") -> dict:
#     """
#     根据ID从数据库中删除指定材料信息
#     Args:
#         ID: 材料在数据库中的ID
#         database: 数据库文件名,默认material_database.db
#     Returns:
#         删除结果
#     """
#     try:
#         db = databasemanage.DatabaseManager(database)
#         material = db.get_material_by_ID(ID)
#         if not material:
#             db.close()
#             return {"error": f"在{database}中未找到ID为 {ID} 的材料", "message": f"请检查ID或数据库名称是否正确"}
#         db.remove_material(ID)
#         db.close()
#         return {"message": f"材料 ID {ID} 已成功从数据库{database}中删除"}
#     except Exception as e:
#         return {"error": str(e), "message": f"从数据库{database}中删除材料 ID {ID} 失败，请检查ID或数据库名称是否正确"}


# @mcp.tool()
# async def list_databases() -> list:
#     """
#     列出当前目录下的所有数据库文件(.db)
    
#     Returns:
#         数据库文件列表
#     """
#     db_files = [f for f in os.listdir('.') if f.endswith('.db')]
#     return db_files 



# 任务投送模块
@mcp.tool()
async def create_task(formula: str, cif_path: str) -> dict:
    """
    在远程服务器上创建任务文件夹并上传CIF文件
    
    Args:
        formula: 化学式字符串 (如"SiO2")
        cif_path: CIF文件路径
    
    Returns:
        任务结果
    """
    try:
        with connection as vasp_task:
            base_dir = config.get_base_dir()
            if not base_dir:
                raise ValueError("base_dir环境变量未设置")
            result = None
            for _ in range(3):
                result = vasp_task.create_task(formula, cif_path, base_dir)
                if result:
                    break
            if result:
                return {"message": f"任务目录已创建并上传CIF文件", "task_directory": result}
            else:
                return {"error": "任务创建失败", "message": "请再试一次"}
    except Exception as e:
        return {"error": str(e), "message": "任务创建失败"}

@mcp.tool()
async def list_task_directories() -> dict:
    """
    列出远程服务器上的所有任务目录
    
    Returns:
        任务目录列表
    """
    try:
        with connection as vasp_task:
            base_dir = config.get_base_dir()
            if not base_dir:
                raise ValueError("base_dir环境变量未设置")
            result = None
            for _ in range(3):
                result = vasp_task.get_task_directories(base_dir)
                if result:
                    break
            if result:
                return {"task_directories": result}
            else:
                return {"error": "获取任务目录失败", "message": "请检查服务器连接是否正常"}
    except Exception as e:
        return {"error": str(e), "message": "获取任务目录失败"}

@mcp.tool()
async def check_squeue() -> dict:
    """
    检查远程服务器上的任务队列
    
    Returns:
        任务队列信息
    """
    try:
        with connection as vasp_task:
            result = None
            for _ in range(3):
                result = vasp_task.check_squeue()
                if result:
                    break
            if result:
                return {"squeue": result}
            else:
                return {"error": "检查任务队列失败", "message": "请检查服务器连接是否正常"}
    except Exception as e:
        return {"error": str(e), "message": "检查任务队列失败"}

@mcp.tool()
async def submit_opt_mission(task_directory: str) -> dict:
    """
    提交结构优化任务到远程服务器
    
    Args:
        task_directory: 任务目录路径
    
    Returns:
        任务提交结果
    """
    try:
        with connection as vasp_task:
            result = None
            for _ in range(3):
                result = vasp_task.opt(task_directory)
                if result:
                    break
            return result
    except Exception as e:
        return {"error": str(e), "message": "任务提交失败"}

@mcp.tool()
async def extract_opt_info(task_directory: str, get_plot: bool = True, visualize: bool = False) -> dict:
    """
    提取结构优化任务的结果信息
    Args:
        task_directory: 任务目录路径
        get_plot: 是否生成结构预览图
        visualize: 是否生成3D可视化结构图
    Returns:
        结构优化结果信息
    """
    try:
        with connection as vasp_task:
            result = None
            for _ in range(3):
                result = vasp_task.extract_opt_info(task_directory)
                if result:
                    break
            if visualize:
                structure_url = visualize_structure(result['structure'])
                result["3d_image_url"] = structure_url
            if get_plot:
                res = get_structure_plot(result['structure'])
                image = res["Image"]
                result["error"] = res['error']
                result["image_url"] = image
            result.pop("structure")  
            return result
    except Exception as e:
        return {"error": str(e), "message": "提取任务结果失败"}


@mcp.tool()
async def submit_scf_mission(task_directory: str, custom_incar: dict = None) -> dict:
    """
    提交自洽计算任务到远程服务器
    
    Args:
        task_directory: 任务目录路径
        custom_incar: 自定义INCAR参数字典，会覆盖默认参数，默认None,默认的自洽计算INCAR参数如下
        default_incar_dict = {
            "SYSTEM": "SCF Calculation",
            "ENCUT": encut,        # 平面波截断能量
            "ISMEAR": 0,           # 高斯展宽
            "SIGMA": 0.05,         # 展宽宽度
            "EDIFF": 1E-6,         # 电子步收敛精度
            "LWAVE": True,         # 输出WAVECAR
            "LCHARG": True,        # 输出CHGCAR
            "NSW": 0,              # 离子步数为0（自洽计算）
            "IBRION": -1,          # 不进行离子弛豫
            "ISIF": 2,             # 固定晶胞
            "PREC": "Accurate",    # 精度设置
            "ALGO": "Normal",      # 电子优化算法
            "NELM": 100,           # 最大电子步数
        }
    Returns:
        任务提交结果
    """
    try:
        with connection as vasp_task:
            result = vasp_task.scf(task_directory, custom_incar=None)
            return result
    except Exception as e:
        return {"error": str(e), "message": "任务提交失败"}
        
@mcp.tool()
async def extract_scf_info(task_directory: str) -> dict:
    """
    提取自洽计算任务的结果信息
    Args:
        task_directory: 任务目录路径
    Returns:
        自洽计算结果信息
    """
    try:
        with connection as vasp_task:
            result = vasp_task.extract_scf_info(task_directory)
            return result
    except Exception as e:
        return {"error": str(e), "message": "提取任务结果失败"}


@mcp.tool()
async def submit_band_mission(task_directory: str) -> dict:
    """
    提交能带计算任务到远程服务器
    
    Args:
        task_directory: 任务目录路径
    Returns:
        任务提交结果
    """
    try:
        with connection as vasp_task:
            result = None
            for _ in range(3):
                result = vasp_task.band_calc(task_directory)
                if result:
                    break
            return result
    except Exception as e:
        return {"error": str(e), "message": "任务提交失败"}


def plot_vasp_band(xml_path, kpoints_path):
    """
    使用 Pymatgen 绘制高质量能带图
    """
    from pymatgen.io.vasp import Vasprun
    from pymatgen.electronic_structure.plotter import BSPlotter

    try:
        # 1. 加载数据 (指定 KPOINTS 以获得正确的标签)
        run = Vasprun(xml_path, parse_projected_eigen=False)
        # 注意：对于金属，get_band_structure 依然有效
        bs = run.get_band_structure(kpoints_filename=kpoints_path, line_mode=True)

        # 6. 提取详细物理量 (处理金属态空值问题)
        is_metal = bs.is_metal()
        gap_info = bs.get_band_gap()
        
        # 如果是金属，VBM/CBM 通常定义在费米面交叉处
        # 这里做一个简单的安全提取
        results = {
            "is_metal": is_metal,
            "gap": gap_info['energy'],
            "fermi_energy": bs.efermi,
        }

        # 2. 初始化绘制器
        plotter = BSPlotter(bs)
        
        # 3. 获取画布
        # get_plot() 实际上返回的是 matplotlib.pyplot 模块
        plt_module = plotter.get_plot()
        fig = plt.gcf() 
        ax = plt.gca()
        

        # --- 标签深度美化 ---
        # 获取当前刻度位置和原始文本
        xticks = ax.get_xticks()
        labels = [label.get_text() for label in ax.get_xticklabels()]
        
        # 替换 GAMMA 为标准大写希腊字母 \Gamma
        # 如果你确定想要小写，请把 \Gamma 改为 \gamma
        fixed_labels = [l.replace('GAMMA', r'$\Gamma$') for l in labels]
        
        # 重新设置刻度，确保对齐
        ax.set_xticks(xticks)
        ax.set_xticklabels(fixed_labels, fontsize=20)
        
        # 美化 Y 轴标签
        ax.set_ylabel(r'$E - E_f$ (eV)', fontsize=20)
        ax.set_title('Band Structure', fontsize=22, pad=20)

        # 画一条更醒目的费米面红线
        ax.axhline(y=0, color='#d62728', linestyle='--', linewidth=2, zorder=1)

        # 导出
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
            
        
        return {"Image": get_plot_url(buf), "data": results, "error": None}

    except Exception as e:
        return {"Image": None, "error": str(e)}


@mcp.tool()
async def extract_band_info(task_directory: str, plot_band: bool = True) -> dict:
    """
    提取能带计算任务的结果信息
    Args:
        task_directory: 任务目录路径
        plot_band: 是否绘制能带图
    Returns:
        能带计算结果信息
    """
    try:
        with connection as vasp_task:
            result = vasp_task.extract_band_info(task_directory)
            if plot_band:
                res = plot_vasp_band(xml_path= result['local_files']['vasprun.xml'],kpoints_path= result['local_files']['KPOINTS'])
                if not res["error"]:
                    image = res["Image"]
                    res.pop("Image")
                    result.update({"image_url": image, "plot_info": res, "message": "绘图成功"})
                else:
                    res.pop("Image")
                    result.update({"plot_info": res, "message": "绘图失败"})
            return result
    except Exception as e:
        return {"error": str(e), "message": "提取任务结果失败"}
    
@mcp.tool()
async def submit_band_mission(task_directory: str) -> dict:
    """
    提交能带计算任务到远程服务器
    
    Args:
        task_directory: 任务目录路径
    Returns:
        任务提交结果
    """
    try:
        with connection as vasp_task:
            result = None
            for _ in range(3):
                result = vasp_task.band_calc(task_directory)
                if result:
                    break
            return result
    except Exception as e:
        return {"error": str(e), "message": "任务提交失败"}
    

@mcp.tool()
async def excute_command(command: str) -> dict:
    """
    在计算服务器上执行linux命令（注意计算服务器和mcp服务器不是同一个服务器）
    若要执行python
    Args:
        command: 命令（严禁使用危险命令）
    Returns:
        执行的结果
    """
    try:
        with connection as vasp_task:
            result = vasp_task.excute_command(command)
            return result
    except Exception as e:
        return {"error": str(e), "message": "命令提交或执行失败"}
    
# 机器学习模块
@mcp.tool()
async def predict_band_gap(formula:str | list[str]) -> dict:
    """
    使用预训练模型预测指定材料的带隙值
    
    Args:
        formula: 化学式字符串或列表 (如"SiO2"或["SiO2", "Fe2O3"])
    Returns:
        带隙预测结果
    """
    from myml import bandgap_predict as mm
    try:
        result = mm.predict_bandgap(formula)
        return {
            "formula": formula,
            "predicted_band_gap": result
        }
    except Exception as e:
        return {"error": str(e), "message": f"预测材料 {formula} 的带隙值失败"}



import json

WORKFLOW_FILE = "material_workflow.json"


def load_workflows():
    try:
        with open(WORKFLOW_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_workflows(data):
    with open(WORKFLOW_FILE, "w") as f:
        json.dump(data, f, indent=4)

@mcp.tool()
async def set_task_progress(project_name: str, description: str = "", step_name: str = "", status: str = "") -> str:
    """
    用简单的字典记录项目进度。
    Args:
        project_name: 项目或材料名称 (如 "MoS2_Bandgap_Study")
        description: 项目描述 (如 "本项目旨在研究MoS2的带隙,将进行晶体建模、结构优化和自洽计算，目前已进行到结构优化阶段，下一步是自洽计算")
        step_name: 步骤名称 (如 "VASP_Opt")
        status: 状态 (如 "Pending", "Running", "Completed", "Failed")
    """
    db = load_workflows()
    if project_name not in db:
        db[project_name] = {}
    if description:
        db[project_name]["description"] = description
    if step_name and status:
        db[project_name][step_name] = {
            "status": status,
            "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        }
    save_workflows(db)
    return f"项目 {project_name} 的步骤 {step_name} 已更新为 {status}。项目描述：{description}。"

@mcp.tool()
async def list_all_projects() -> list[str]:
    """
    列出当前所有材料研发项目的名称。
    用于在开始工作前确认有哪些正在进行的项目。
    """
    db = load_workflows()
    return list(db.keys())

@mcp.tool()
async def get_project_workflow(project_name: str) -> dict:
    """
    根据项目名称查看具体的任务清单和进度。
    
    Args:
        project_name: 项目名称（如 "Li3InCl6_Optimization"）
    """
    db = load_workflows()
    if project_name not in db:
        return {"error": f"未找到名为 '{project_name}' 的项目", "current_projects": list(db.keys())}
    
    return {
        "project": project_name,
        "workflow": db[project_name]
    }

@mcp.tool()
async def read_file(file_path: str) -> dict:
    """
    读取mcp服务器的文件
    
    Args:
        file_path: 文件的路径
    """
    try:
        # 这里应该包含实际的文件读取逻辑
        # 例如：检查文件是否存在，验证文件路径安全性等
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return {
            "success": True,
            "content": content,
            "file_path": file_path
        }
    
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"文件未找到: {file_path}",
            "file_path": file_path
        }
    except PermissionError:
        return {
            "success": False, 
            "error": f"权限不足: {file_path}",
            "file_path": file_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"读取文件时出错: {str(e)}",
            "file_path": file_path
        }









if __name__ == "__main__":
    try:
        # 启动MCP服务器
        connection = tryssh.VaspTaskInitializer(HOST, USERNAME, PASSWORD, PORT)
        for i in range(5):
            try:
                with connection as vasp_task:
                    if vasp_task.link():
                        print("已成功连接到远程服务器")
                        break
            except Exception as e:
                print(f"连接远程服务器失败，正在重试... ({i+1}/5)")
                if i == 4:
                    raise e
        server = flask_plot.MemoryImageServer(port=6760)
        server.start()
        crystalmanager = flask_builder.CrystalManager()
        mcp.run(
            # transport="streamable-http",
            transport="sse",
            host="0.0.0.0",
            port=8000
        )
    except Exception as e:
        print(f"服务器运行出错: {e}")
        exit()