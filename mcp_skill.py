"""MCP Agent Skill

这个文件定义了一个可被 agent 直接调用的 skill 类：
- 连接你的 mpmcp 服务（默认 http://127.0.0.1:8000）
- 列出工具、调用工具
- 常用材料和计算任务接口封装

使用方式：
    from mcp_skill import MCPAgentSkill
    skill = MCPAgentSkill(mcp_url="http://127.0.0.1:8000")
    result = skill.get_band_gap("mp-149")
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastmcp.client import Client


class MCPAgentSkill:
    """Agent Skill wrapper for your mcp service."""

    def __init__(self, mcp_url: str = "http://127.0.0.1:8000"):
        self.mcp_url = mcp_url
        self.client = Client(mcp_url)

    async def _call_tool_async(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        arguments = arguments or {}
        async with self.client as c:
            result = await c.call_tool(tool_name, arguments)
        # `call_tool` returns a list of MCPContent; unify.
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result

    async def _list_tools_async(self) -> List[str]:
        async with self.client as c:
            tools = await c.list_tools()
        # 如果返回对象含 name 字段，则提取
        if isinstance(tools, list):
            return [t.name if hasattr(t, "name") else str(t) for t in tools]
        return [str(tools)]

    def _run_async(self, coro):
        return asyncio.run(coro)

    def list_tools(self) -> List[str]:
        """列出当前 mcp 服务中的可用工具名称。"""
        return self._run_async(self._list_tools_async())

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """通用调用任意 MCP 工具。"""
        return self._run_async(self._call_tool_async(tool_name, arguments))

    def get_time(self) -> str:
        return self.call_tool("get_time")

    def get_material_project_page(self, material_id: str) -> dict:
        return self.call_tool("get_material_project_page", {"material_id": material_id})

    def search_materials(
        self,
        elements: Optional[List[str]] = None,
        exclude_elements: Optional[List[str]] = None,
        chemsys: Optional[str | List[str]] = None,
        band_gap: Optional[tuple[float, float]] = None,
        num_elements: Optional[tuple[int, int]] = None,
        formula: Optional[str | List[str]] = None,
        chunk_size: Optional[int] = 25,
    ) -> Any:
        args = {
            "elements": elements,
            "exclude_elements": exclude_elements,
            "chemsys": chemsys,
            "band_gap": band_gap,
            "num_elements": num_elements,
            "formula": formula,
            "chunk_size": chunk_size,
        }
        # 移除 None 参数
        args = {k: v for k, v in args.items() if v is not None}
        return self.call_tool("search_materials", args)

    def get_band_gap(self, material_id: str) -> Any:
        return self.call_tool("get_band_gap", {"material_id": material_id})

    def get_material_structure(
        self,
        material_id: str,
        get_sites: bool = False,
        get_plot: bool = False,
        download: bool = False,
    ) -> Any:
        return self.call_tool(
            "get_material_structure",
            {
                "material_id": material_id,
                "get_sites": get_sites,
                "get_plot": get_plot,
                "download": download,
            },
        )

    def build_structure(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float,
        beta: float,
        gamma: float,
        elements: List[str],
        frac_coord: List[List[float]],
        scaling_matrix: int | List[int] = 1,
        save_to_cif: bool = False,
        add_to_database: Optional[str] = None,
    ) -> Any:
        return self.call_tool(
            "build_structure",
            {
                "a": a,
                "b": b,
                "c": c,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "elements": elements,
                "frac_coord": frac_coord,
                "scaling_matrix": scaling_matrix,
                "save_to_cif": save_to_cif,
                "add_to_database": add_to_database,
            },
        )

    def create_task(self, formula: str, cif_path: str) -> Any:
        return self.call_tool("create_task", {"formula": formula, "cif_path": cif_path})

    def list_task_directories(self) -> Any:
        return self.call_tool("list_task_directories")

    def check_squeue(self) -> Any:
        return self.call_tool("check_squeue")

    def submit_opt_mission(self, task_directory: str) -> Any:
        return self.call_tool("submit_opt_mission", {"task_directory": task_directory})

    def extract_opt_info(self, task_directory: str, get_plot: bool = True, visualize: bool = False) -> Any:
        return self.call_tool(
            "extract_opt_info",
            {
                "task_directory": task_directory,
                "get_plot": get_plot,
                "visualize": visualize,
            },
        )

    def submit_scf_mission(self, task_directory: str, custom_incar: Optional[Dict[str, Any]] = None) -> Any:
        return self.call_tool(
            "submit_scf_mission",
            {
                "task_directory": task_directory,
                "custom_incar": custom_incar,
            },
        )

    def extract_scf_info(self, task_directory: str) -> Any:
        return self.call_tool("extract_scf_info", {"task_directory": task_directory})

    def submit_band_mission(self, task_directory: str) -> Any:
        return self.call_tool("submit_band_mission", {"task_directory": task_directory})

    def extract_band_info(self, task_directory: str, plot_band: bool = True) -> Any:
        return self.call_tool("extract_band_info", {"task_directory": task_directory, "plot_band": plot_band})

    def excute_command(self, command: str) -> Any:
        return self.call_tool("excute_command", {"command": command})


if __name__ == "__main__":
    skill = MCPAgentSkill()
    print("[MCP Skill] 已连接到", skill.mcp_url)
    print("[MCP Skill] 可用工具：", skill.list_tools())
    print("[示例] get_time ->", skill.get_time())
