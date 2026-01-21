import os
from mp_api.client import MPRester

# 也可以从环境变量读取
MP_API_KEY = os.getenv("MP_API_KEY")

def search_materials_logic(elements=None, band_gap=None, formula=None, num_elements=None, **kwargs):
    """
    核心业务逻辑：对接 Materials Project API
    """
    try:
        with MPRester(MP_API_KEY) as mpr:
            # 转换参数格式 (例如将列表转为元组)
            criteria = {}
            if elements: criteria["elements"] = elements
            if band_gap: criteria["band_gap"] = tuple(band_gap)
            if formula:  criteria["formula"] = formula
            if num_elements: criteria["num_elements"] = num_elements
            print(f"正在调用工具查询:{criteria}")
            results = mpr.summary.search(
                **criteria,
                fields=["material_id", "formula_pretty", "band_gap", "symmetry"],
                chunk_size=10,
                num_chunks=1
            )
            print(f"查询到{len(results)}条结果")
            return [{
                "material_id": str(r.material_id),
                "formula_pretty": r.formula_pretty,
                "band_gap": r.band_gap,
                "symmetry": str(r.symmetry.crystal_system)
            } for r in results]
    except Exception as e:
        return {"error": str(e)}

# 导出给 DeepSeek 的工具定义 (Schema)
DEEPSEEK_TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_materials",
            "description": "查询 Materials Project 数据库中的材料信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "elements": {"type": "array", "items": {"type": "string"}},
                    "band_gap": {"type": "array", "items": {"type": "number"}, "description": "[最小值, 最大值]"},
                    "formula": {"type": "string"},
                    "num_elements": {"type": "integer"}
                }
            }
        }
    }
]
