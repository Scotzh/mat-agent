from fastmcp import FastMCP
import asyncio
import os
import pandas as pd
from mp_api.client import MPRester
from pydantic import BaseModel
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
# import pymatgen.electronic_structure
# from pymatgen.electronic_structure.plotter import DosPlotter
# from pymatgen.electronic_structure.dos import CompleteDos
import matplotlib.pyplot as plt
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from ase.visualize import view
import multiprocessing
import flask_builder
import duckdb
import pickle
import loadenv
import databasemanage
import tryssh
plt.style.use('seaborn-v0_8-whitegrid')
mcp = FastMCP(name="MaterialDataServer")
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
                                download: bool = False) -> dict:
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
                visualize_structure(structure)
                message.append("3d晶体结构可视化交互式网页，请点击查看晶体结构图")
                message.append(f"3d_image_url: {IMAGE_URL}")

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
                          add_to_database: bool = False,
                          database: str = 'custom_structures.db') -> dict:
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
        add_to_database: 是否将结构添加到数据库,默认False
        database: 数据库文件名,默认'custom_structures.db'
    """
    try:
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        structure = Structure(lattice, elements, frac_coord)
        formula = structure.composition.reduced_formula
        os.makedirs("custom_structures", exist_ok=True)
        os.makedirs("custom_structures/images", exist_ok=True)
        CifWriter(structure).write_file(f"custom_structures/{formula}_custom.cif")
        message = [f"自定义晶体结构已保存为 custom_structures/{formula}_custom.cif"]
        visualize_structure(structure)
        message.append("3d晶体结构可视化交互式网页，请点击查看晶体结构图")
        if add_to_database:
            db = databasemanage.DatabaseManager(database)
            db.add_material(formula=formula, structure=structure, band_gap=None, material_id=None)
            db.close()
            message.append(f"自定义晶体结构已添加到数据库 {database}")

        return {"cif_file_path": f"custom_structures/{formula}_custom.cif",
                "3d_image_url": IMAGE_URL,
                "message": message
                }

    except Exception as e:
        return {"error": str(e), "message": "构建晶体结构失败"}


def visualize_structure(structure: Structure) -> None:
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

    htmlviewer = flask_builder.CrystalStructureVisualizer(structure, html_path)

    # 先检查有没有正在运行的子程序，如果有，先停止并删除其临时目录
    if child_processes:
        for p, tdir in list(child_processes):
            try:
                if p.is_alive():
                    p.terminate()
                    p.join(3)
                    if p.is_alive():
                        try:
                            p.kill()
                        except Exception:
                            pass
                        p.join(1)
            except Exception:
                pass
            # 删除该子进程对应的临时目录
            try:
                if tdir and os.path.exists(tdir):
                    shutil.rmtree(tdir, ignore_errors=True)
            except Exception:
                pass
            try:
                child_processes.remove((p, tdir))
            except ValueError:
                pass

    # 启动新进程并记录 (process, temp_dir)
    p = multiprocessing.Process(target=htmlviewer.run)
    p.start()
    child_processes.append((p, temp_dir))

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
async def extract_opt_info(task_directory: str, visualize: bool = True) -> dict:
    """
    提取结构优化任务的结果信息
    Args:
        task_directory: 任务目录路径
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
                visualize_structure(result['structure'])
                result["3d_image_url"] = IMAGE_URL
            result.pop("structure")  # 删除structure对象，避免序列化问题
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


# 机器学习模块
@mcp.tool()
async def predict_band_gap(formula: str) -> dict:
    """
    使用预训练模型预测指定材料的带隙值
    
    Args:
        formula: 化学式字符串 (如"SiO2")
    
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

# @mcp.tool()
# async def predict_halid_ionic_conductivity(formula: str) -> dict:
#     """
#     使用预训练模型预测指定卤化物材料的离子电导率(25℃)，主要针对卤化物锂离子固态电解质材料
    
#     Args:
#         formula: 化学式字符串 (如"Li3InCl6")
    
#     Returns:
#         离子电导率预测结果
#     """
#     from myml import ion_conductivity as icp
#     try:
#         result = icp.predict_ionic_conductivity(formula)
#         return {
#             "formula": formula,
#             "predicted_ionic_conductivity": result
#         }
#     except Exception as e:
#         return {"error": str(e), "message": f"预测材料 {formula} 的离子电导率失败"}

# @mcp.tool()
# async def predict_mixed_halid_ionic_conductivity(formula1:str, formula2:str, ratios:list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) -> dict:
#     """
#     使用预训练模型预测两种卤化物材料混合后的离子电导率(25℃)，主要针对卤化物固锂离子态电解质材料
    
#     Args:
#         formula1: 第一种卤化物化学式字符串 (如"Li3InCl6")
#         formula2: 第二种卤化物化学式字符串 (如"Li3YCl6")
#         ratios: 混合比例列表，表示formula1的比例，默认[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#     Returns:
#         混合离子电导率预测结果
#     """
#     from myml import ion_conductivity as micp
#     try:
#         result = micp.predict_mixed_ionic_conductivity(formula1, formula2, ratios)
#         return {
#             "formula1": formula1,
#             "formula2": formula2,
#             "predicted_mixed_ionic_conductivity": result
#         }
#     except Exception as e:
#         return {"error": str(e), "message": f"预测材料 {formula1} 和 {formula2} 混合后的离子电导率失败"}


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
        mcp.run(
            transport="sse",
            host="127.0.0.1",
            port=8000
        )
    except Exception as e:
        print(f"服务器运行出错: {e}")
        exit()