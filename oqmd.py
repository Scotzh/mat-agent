import requests
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime
from pymatgen.core import Structure
from io import StringIO
from typing import Optional, Dict, Any, List
# 请求头配置
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}
def search_oqmd(
    fields: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    sort_by: Optional[str] = None,
    desc: bool = False,
    noduplicate: bool = False,
    icsd: Optional[bool] = None
) -> Dict[str, Any]:
    """
    搜索OQMD材料数据库
    
    Args:
        fields: 需要返回的字段列表，如 ["name", "entry_id", "band_gap"]
        filter_expr: 筛选表达式，如 "element_set=(Al-Fe),O AND stability=0"
        limit: 每页最大记录数，默认50
        offset: 偏移量（用于翻页），默认0
        sort_by: 排序字段，如 "delta_e", "stability"
        desc: 是否降序排序，默认False
        noduplicate: 是否排除重复条目，默认False
        icsd: 是否仅限ICSD收录的结构
        
    Returns:
        包含查询结果的字典，包含数据和元信息
    """
    params = {}
    if fields:
        params["fields"] = ",".join(fields)
    if filter_expr:
        params["filter"] = filter_expr
    if limit:
        params["limit"] = limit
    if offset:
        params["offset"] = offset
    if sort_by:
        params["sort_by"] = sort_by
    if desc:
        params["desc"] = str(desc).lower()
    if noduplicate:
        params["noduplicate"] = str(noduplicate).lower()
    if icsd is not None:
        params["icsd"] = str(icsd).lower()
    
    params["format"] = "json"
    BASE_URL = "http://oqmd.org/oqmdapi/formationenergy"
    try:
        response = requests.get(BASE_URL, params=params, timeout=90)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "error": f"请求失败: {str(e)}",
            "status": "error"
        }
    except Exception as e:
        return {
            "error": f"处理数据时发生错误: {str(e)}",
            "status": "error"
        }

def safe_get(url, retries=5, delay=5, timeout=100):
    """安全的HTTP GET请求，包含重试机制"""
    for i in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                print(f"⚠️ 429 Too Many Requests. Retrying in {delay}s...", flush=True)
            elif resp.status_code in [502, 503, 504]:
                print(f"⚠️ Server error {resp.status_code}. Retrying in {delay}s...", flush=True)
            else:
                resp.raise_for_status()

        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
            print(f"⏱️ Timeout. Retrying in {delay}s...", flush=True)
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}", flush=True)

        # 渐进式延迟
        if i == 3:
            time.sleep(120)
        elif i == 4:
            time.sleep(300)
        else:
            time.sleep(delay)

    raise Exception(f"Failed to download after {retries} retries: {url}")


def get_poscar_content(entry_id: int, mode="conventional"):
    """
    通过entry_id获取POSCAR文件内容
    
    参数:
        entry_id: OQMD材料条目的ID
        mode: 下载模式，"conventional"或"primitive"
    
    返回:
        POSCAR内容的字符串
    """
    # 构建材料详情页URL
    url = f"https://oqmd.org/materials/entry/{entry_id}"
    resp = safe_get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # 查找POSCAR下载链接
    target_prefix = f"/materials/export/{mode}/poscar/"
    
    for a in soup.find_all("a", href=True):
        if a["href"].startswith(target_prefix):
            poscar_url = "https://oqmd.org" + a["href"]
            break
    else:
        raise ValueError(f"POSCAR download URL not found for entry {entry_id}")
    
    # 下载POSCAR内容
    poscar_resp = requests.get(poscar_url)
    if poscar_resp.status_code != 200:
        raise ValueError(f"Failed to download POSCAR: HTTP {poscar_resp.status_code}")
    
    return poscar_resp.text


def parse_poscar_with_pymatgen(entry_id: int, mode="conventional"):
    """
    主函数：获取POSCAR并用pymatgen解析
    
    参数:
        entry_id: OQMD材料条目的ID
        mode: 下载模式，"conventional"或"primitive"
    
    返回:
        pymatgen Structure对象
    """
    try:
        # 1. 获取POSCAR内容
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{time_str} 📥 开始处理 entry_id: {entry_id}", flush=True)
        
        poscar_content = get_poscar_content(entry_id, mode)
        print(f"✅ 成功获取POSCAR内容，长度: {len(poscar_content)} 字符", flush=True)
        
        # 2. 使用pymatgen解析POSCAR
        # 方法1: 从字符串解析
        structure = Structure.from_str(poscar_content, fmt="poscar")
        
        # 方法2: 也可以从StringIO解析（如果pymatgen版本需要文件对象）
        # from io import StringIO
        # structure = Structure.from_file(StringIO(poscar_content))
        
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{time_str} ✅ 成功解析为Structure对象", flush=True)
        # lattice = structure.lattice
        # space_group_info = structure.get_space_group_info()
        formula = structure.formula
        # reduced_formula = structure.composition.reduced_formula
        # structure_info = {
        #     'formula': formula,
        #     'reduced_formula': reduced_formula,
        #     'space_group_symbol': space_group_info[0] if space_group_info else "未知",
        #     'space_group_number': space_group_info[1] if space_group_info else "未知",
        #     'lattice_parameters': {
        #         'a': round(lattice.a, 4),
        #         'b': round(lattice.b, 4),
        #         'c': round(lattice.c, 4),
        #         'alpha': round(lattice.alpha, 2),
        #         'beta': round(lattice.beta, 2),
        #         'gamma': round(lattice.gamma, 2),
        #         'volume': round(lattice.volume, 4)
        #     },
        #     'number_of_sites': len(structure),
        #     'density': round(structure.density, 4),
        #     'is_ordered': structure.is_ordered,
        # }
        
        return {"success": True, "structure": structure, "formula":formula}
        
    except Exception as e:
        print(f"❌ 处理 entry_id {entry_id} 失败: {e}", flush=True)
        return {"success": False,"error" :f"处理 entry_id {entry_id} 失败: {e}"}


# 使用示例
if __name__ == "__main__":
    # 示例1: 单个entry_id测试
    test_entry_id = 12345  # 替换为你要测试的entry_id
    
    try:
        # 获取并解析结构
        structure = parse_poscar_with_pymatgen(test_entry_id, mode="conventional")
        
        # 现在你可以使用pymatgen的Structure对象进行各种操作
        print(f"\n🎯 结构信息摘要:")
        print(f"   晶格参数: {structure.lattice.parameters}")
        print(f"   密度: {structure.density:.4f} g/cm³")
        print(f"   带隙信息: 可通过structure.get_band_structure()进一步分析")
        
        # 保存到文件（可选）
        # structure.to(fmt="poscar", filename=f"structure_{test_entry_id}.vasp")
        
    except Exception as e:
        print(f"主程序错误: {e}")
    
    # 示例2: 批量处理（如果需要）
    # entry_ids = [12345, 67890, 11111]  # 你的ID列表
    # structures = {}
    # for eid in entry_ids:
    #     try:
    #         structure = parse_poscar_with_pymatgen(eid)
    #         structures[eid] = structure
    #         time.sleep(1)  # 防止请求过快
    #     except Exception as e:
    #         print(f"跳过 entry_id {eid}: {e}")