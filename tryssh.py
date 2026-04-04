import paramiko
import os
from datetime import datetime
from pymatgen.core import Structure
from pymatgen.io import vasp
from typing import Dict, List, Optional, Union
import loadenv
import re
import numpy as np


class VaspTaskInitializer:
    def __init__(self, hostname, username, password=None, port=22, key_filename=None):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.port = port
        self.key_filename = key_filename
        self.ssh = None
        self.sftp = None

    def __enter__(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if self.key_filename:
            self.ssh.connect(self.hostname, port=self.port, username=self.username, key_filename=self.key_filename)
        else:
            self.ssh.connect(self.hostname, port=self.port, username=self.username, password=self.password)
        self.sftp = self.ssh.open_sftp()
        print("SSH和SFTP连接已建立")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()
        print("SSH和SFTP连接已关闭")

    def link(self):
        if self.ssh:
            return True
        return False

    def create_task(self, chemical_formula, local_cif_path, base_dir):
        current_date = datetime.now().strftime("%Y%m%d")
        task_dir = os.path.join(base_dir, f"{chemical_formula}_{current_date}")
        try:
            # 1. 远程创建主目录和子目录（无论本地如何）
            self.ssh.exec_command(f"mkdir -p '{task_dir}'")
            subfolders = ["自洽计算", "结构优化", "态密度计算", "能带计算"]
            for folder in subfolders:
                stdin, stdout, stderr = self.ssh.exec_command(f"mkdir -p '{os.path.join(task_dir, folder)}'")
                stdout.channel.recv_exit_status()  # 等待每个命令完成

            # 2. 上传CIF文件
            cif_filename = os.path.basename(local_cif_path)
            remote_cif_path = os.path.join(task_dir, cif_filename)
            self.sftp.put(local_cif_path, remote_cif_path)
            print(f"CIF文件已上传到远程服务器: {remote_cif_path}")
            return task_dir
        except Exception as e:
            print(f"创建任务目录或上传CIF文件时出错: {e}")
            return None

    def get_task_directories(self, base_dir):
        # 获取所有任务目录
        stdin, stdout, stderr = self.ssh.exec_command(f"ls -d {base_dir}/*/")
        dirs = stdout.read().decode().splitlines()
        return dirs


    def check_squeue(self):
        stdin, stdout, stderr = self.ssh.exec_command("squeue -u $USER")
        output = stdout.read().decode()
        error = stderr.read().decode()
        if error:
            print(f"Error checking squeue: {error}")
            return None
        return output

    def relax(self, task_dir):
        command = f"cd '{task_dir}' && ./../auto_opt.sh"
        stdin, stdout, stderr = self.ssh.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        print(output)
        print(error)
        return {"status": "结构优化任务已提交",
                "command": command,
                'stdout': str(output),
                'stderr': str(error)}

    def extract_relax_info(self, task_dir):
        def _extract_outcar(outcar_path):
            """从OUTCAR提取信息"""
            outcar_info = {}
            try:
                outcar = vasp.Outcar(outcar_path)
                if outcar.final_energy is not None:
                    outcar_info['free_energy'] = outcar.final_energy
                
                # 获取 energy without entropy (energy at sigma->0)
                if hasattr(outcar, 'final_energy_wo_entrp') and outcar.final_energy_wo_entrp is not None:
                    outcar_info['energy_without_entropy'] = outcar.final_energy_wo_entrp
                
                # 计算熵贡献 (T*S)
                if 'free_energy' in outcar_info and 'energy_without_entropy' in outcar_info:
                    outcar_info['entropy'] = outcar_info['energy_without_entropy'] - outcar_info['free_energy']
                
                # 计算每个原子的能量
                if 'free_energy' in outcar_info and outcar.natoms > 0:
                    outcar_info['free_energy_per_atom'] = outcar_info['free_energy'] / outcar.natoms
                    outcar_info['num_atoms'] = outcar.natoms
                
                 # 最后一步的力
                if outcar.forces is not None and len(outcar.forces) > 0:
                    forces = outcar.forces
                    outcar_info['final_forces'] = forces[-1].tolist()

                # 获取应力（最后一步的应力）
                if outcar.stress is not None and len(outcar.stress) > 0:
                    stress = outcar.stress
                    outcar_info['final_stress'] = stress[-1].tolist()
            except Exception as e:
                print(f"Error reading OUTCAR: {e}")
            return outcar_info

        def _extract_crystal_structure(contcar_path):
            """从CONTCAR提取晶体结构信息"""
            structure_info = {}
            
            try:
                structure = Structure.from_file(contcar_path)
                lattice = structure.lattice
                space_group_info = structure.get_space_group_info()
                structure_info = {
                    'formula': structure.formula,
                    'reduced_formula': structure.reduced_formula,
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
                    'sites': [{'element': str(site.specie), 'frac_coords': [round(c, 4) for c in site.frac_coords]} for site in structure.sites]
                }
            except Exception as e:
                print(f"Error reading CONTCAR: {e}")
            return structure_info, structure

        # 从"结构优化"目录中提取'OUTCAR', 'CONTCAR'中的自由能和晶体结构信息
        remote_outcar = os.path.join(task_dir, "结构优化", "OUTCAR")
        remote_contcar = os.path.join(task_dir, "结构优化", "CONTCAR")
        
        file_name = os.path.basename(task_dir.rstrip('/'))
        
        # 修正：使用 os.makedirs 而不是 os.mkdir
        output_dir = "./calculation_output/relaxation"
        os.makedirs(output_dir, exist_ok=True)
        
        calculation_outcar = os.path.join(output_dir, f"{file_name}_OUTCAR")
        calculation_contcar = os.path.join(output_dir, f"{file_name}_CONTCAR")
        
        # 下载文件
        try:

            self.sftp.get(remote_outcar, calculation_outcar)
            print(f"OUTCAR文件已下载到本地: {calculation_outcar}")
        except Exception as e:
            print(f"下载OUTCAR失败: {e}")
            return None
        
        try:
            self.sftp.get(remote_contcar, calculation_contcar)
            print(f"CONTCAR文件已下载到本地: {calculation_contcar}")
        except Exception as e:
            print(f"下载CONTCAR失败: {e}")
            return None
        
        # 提取信息
        outcar_info = _extract_outcar(calculation_outcar)
        structure_info = _extract_crystal_structure(calculation_contcar)
        
        return {
            "structure": structure_info[1],
            "outcar_info": outcar_info, 
            "structure_info": structure_info[0],
            "local_files": {
                "outcar": calculation_outcar,
                "contcar": calculation_contcar
            }
        }

        

        

    def scf(self, task_dir, custom_incar: dict = None):
        """
        运行自洽计算
        
        Args:
            task_dir: 任务目录路径
            custom_incar: 自定义INCAR参数字典，会覆盖默认参数
        
        Returns:
            dict: 包含任务状态和执行信息的字典
        """
        # 运行自洽计算脚本(生成文件夹并且提取INCAR)
        command1 = f"cd '{task_dir}' && ./../auto_scf_step1.sh"
        stdin, stdout, stderr = self.ssh.exec_command(command1)
        stdout.read()  # 等待命令执行完成

        # 默认的自洽计算INCAR参数
        default_incar_dict = {
            "SYSTEM": "SCF Calculation",
            "ENCUT": 520,
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
        
        # 如果提供了自定义参数，则覆盖默认参数
        if custom_incar:
            default_incar_dict.update(custom_incar)
            print(f"使用自定义INCAR参数: {custom_incar}")
        
        # 生成新的INCAR对象
        new_incar = vasp.Incar(default_incar_dict)
        
        # 保存到本地临时文件
        new_incar_path = "./temp_new_INCAR"
        new_incar.write_file(new_incar_path)
        print(f"新INCAR文件已生成: {new_incar_path}")
        
        # 上传新的INCAR文件到远程服务器
        remote_new_incar = os.path.join(task_dir, "自洽计算", "INCAR")
        try:
            self.sftp.put(new_incar_path, remote_new_incar)
            print(f"新INCAR文件已上传到: {remote_new_incar}")
        except Exception as e:
            print(f"上传INCAR失败: {e}")
            return {"status": "失败", "error": str(e)}
        
        # 执行第二步脚本（提交计算任务）
        command2 = f"cd '{task_dir}' && ./../auto_scf_step2.sh"
        stdin, stdout, stderr = self.ssh.exec_command(command2)
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        print("=== 标准输出 ===")
        print(output)
        if error:
            print("=== 错误输出 ===")
            print(error)
        
        # 清理临时文件
        try:
            os.remove(new_incar_path)
            print("临时文件已清理")
        except Exception as e:
            print(f"清理临时文件失败: {e}")
        
        return {
            "status": "自洽计算任务已提交",
            "command1": command1,
            "command2": command2,
            "stdout": output,
            "stderr": error,
            "incar_params": default_incar_dict
        }

    def extract_scf_info(self, task_dir):
        remote_vasprun = os.path.join(task_dir, "自洽计算", "vasprun.xml")
        file_name = os.path.basename(task_dir.rstrip('/'))
        
        # 修正：使用 os.makedirs 而不是 os.mkdir
        vasprun_dir = "./calculation_output/scf"
        os.makedirs(vasprun_dir, exist_ok=True)
        
        calculation_vasprun = os.path.join(vasprun_dir, f"{file_name}_vasprun.xml")
        
        # 下载文件
        try:
            self.sftp.get(remote_vasprun, calculation_vasprun)
            print(f"OUTCAR文件已下载到本地: {calculation_vasprun}")
        except Exception as e:
            print(f"下载vasprun.xml失败: {e}")
            return None
        
        # 提取信息
        vasprun_info = {}
        try:
            vasprun = vasp.Vasprun(calculation_vasprun)
            # 能量信息
            print(f"最终能量: {vasprun.final_energy} eV")
            print(f"每原子能量: {vasprun.final_energy/len(vasprun.final_structure)} eV/atom")
            vasprun_info['final_energy'] = vasprun.final_energy
            vasprun_info['energy_per_atom'] = vasprun.final_energy / len(vasprun.final_structure)

            # 电子信息
            print(f"费米能级: {vasprun.efermi} eV")
            print(f"能隙: {vasprun.eigenvalue_band_properties[0]} eV")
            vasprun_info['efermi'] = vasprun.efermi
            vasprun_info['band_gap'] = vasprun.eigenvalue_band_properties[0]

        except Exception as e:
            print(f"Error reading vasprun.xml: {e}")

        return {
            "vasprun_info": vasprun_info,
            "local_files": {
                "vasprun": calculation_vasprun
            }
        }

    def band_calc(self, task_dir):
        """
        进行能带结构计算
        """
        command = f"cd '{task_dir}' && ./../auto_band.sh"
        stdin, stdout, stderr = self.ssh.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        print(output)
        print(error)
        return {"status": "能带计算任务已提交",
                "command": command,
                'stdout': str(output),
                'stderr': str(error)}
            
    def extract_band_info(self, task_dir):
        """
        提取能带计算结果并下载数据文件
        """
        band_dir = os.path.join(task_dir, "能带计算")
        
        # 1. 远程调用 VASPKIT 211 导出格式化数据 (BAND.dat, KLINES.dat)
        # 这样我们可以直接下载处理好的文件用于本地绘图
        extract_cmd = f"cd '{band_dir}' && echo -e '21\\n211' | vaspkit"
        self.ssh.exec_command(extract_cmd)

        # 2. 准备本地目录
        local_output = "./calculation_output/band"
        os.makedirs(local_output, exist_ok=True)
        file_prefix = os.path.basename(task_dir.rstrip('/'))

        # 3. 定义需要下载的文件列表
        files_to_download = {
            "vasprun.xml": f"{file_prefix}_vasprun.xml",
            "REFORMATTED_BAND.dat": f"{file_prefix}_BAND.dat",
            "KLINES.dat": f"{file_prefix}_KLINES.dat",
            "BAND_GAP": f"{file_prefix}_BAND_GAP",
            "KPOINTS": f"{file_prefix}_KPOINTS",
            "BAND.jpg": f"{file_prefix}_BAND.jpg",
        }

        downloaded_info = {}
        for remote_name, local_name in files_to_download.items():
            remote_f = os.path.join(band_dir, remote_name)
            local_f = os.path.join(local_output, local_name)
            try:
                self.sftp.get(remote_f, local_f)
                downloaded_info[remote_name] = local_f
            except:
                print(f"提醒: 未能下载 {remote_name}，可能计算未完成或出错。")

        # 4. 解析基本带隙信息 (从下载的 vasprun.xml)
        band_info = {}
        if "vasprun.xml" in downloaded_info:
            try:
                v = vasp.Vasprun(downloaded_info["vasprun.xml"])
                bs = v.get_band_structure(downloaded_info["KPOINTS"], line_mode=True)
                # 获取带隙、VBM、CBM等
                # band_info['band'] = v.eigenvalue_band_properties
                band_info['is_metal'] = bs.is_metal()
                band_info['gap'] = bs.get_band_gap()
                band_info['vbm'] = bs.get_vbm()
                band_info['cbm'] = bs.get_cbm()
            except Exception as e:
                print(f"解析能带XML失败: {e}")

        return {
            "status": "提取完成",
            "band_info": band_info,
            "local_files": downloaded_info
        }
    
    def dos_calc(self, task_dir):
        """
        进行态密度计算
        """
        command = f"cd '{task_dir}' && ./../auto_dos.sh"
        stdin, stdout, stderr = self.ssh.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        print(output)
        print(error)
        return {"status": "态密度计算任务已提交",
                "command": command,
                'stdout': str(output),
                'stderr': str(error)}
    
    def extract_dos_info(self, task_dir):
        """
        提取态密度计算结果并下载数据文件
        参考VASPKIT文档：功能31用于提取DOS数据
        """
        dos_dir = os.path.join(task_dir, "态密度计算")
        
        # 1. 远程调用 VASPKIT 31 提取DOS数据
        # 31: DOS数据处理，311: 提取总态密度(TDOS)
        extract_cmd = f"cd '{dos_dir}' && echo -e '31\\n311' | vaspkit"
        self.ssh.exec_command(extract_cmd)
        
        # 2. 可选：提取特定原子的投影态密度(PDOS)
        # 这里可以添加交互式选择，简化版先提取所有原子的总PDOS
        # extract_pdos_cmd = f"cd '{dos_dir}' && echo -e '31\\n312\\nall\\nall' | vaspkit"
        # self.ssh.exec_command(extract_pdos_cmd)
        
        # 3. 准备本地目录
        local_output = "./calculation_output/dos"
        os.makedirs(local_output, exist_ok=True)
        file_prefix = os.path.basename(task_dir.rstrip('/'))
        
        # 4. 定义需要下载的文件列表
        files_to_download = {
            "vasprun.xml": f"{file_prefix}_vasprun.xml",
            "DOSCAR": f"{file_prefix}_DOSCAR",
            "TDOS.dat": f"{file_prefix}_TDOS.dat",
            "PDOS.dat": f"{file_prefix}_PDOS.dat",
            "DOS.jpg": f"{file_prefix}_DOS.jpg",
            "OUTCAR": f"{file_prefix}_OUTCAR",
            "INCAR": f"{file_prefix}_INCAR",
        }
        
        # 5. 检查并下载PDOS相关文件（如果存在）
        # 根据VASPKIT文档，PDOS文件命名格式：PDOS_A1.dat, PDOS_A2.dat等
        try:
            stdin, stdout, stderr = self.ssh.exec_command(f"cd '{dos_dir}' && ls PDOS_*.dat 2>/dev/null || echo ''")
            pdos_files = stdout.read().decode().strip().split()
            for pdos_file in pdos_files:
                if pdos_file:
                    local_name = f"{file_prefix}_{pdos_file}"
                    files_to_download[pdos_file] = local_name
        except:
            pass
        
        downloaded_info = {}
        for remote_name, local_name in files_to_download.items():
            remote_f = os.path.join(dos_dir, remote_name)
            local_f = os.path.join(local_output, local_name)
            try:
                self.sftp.get(remote_f, local_f)
                downloaded_info[remote_name] = local_f
            except:
                print(f"提醒: 未能下载 {remote_name}，可能文件不存在或计算未完成。")
        
        # 6. 解析DOS信息
        dos_info = {}
        
        # 6.1 从OUTCAR解析费米能级
        if "OUTCAR" in downloaded_info:
            try:
                with open(downloaded_info["OUTCAR"], 'r') as f:
                    content = f.read()
                    # 查找费米能级
                    fermi_lines = [line for line in content.split('\n') if 'E-fermi' in line]
                    if fermi_lines:
                        fermi_line = fermi_lines[-1]  # 取最后一个
                        efermi = float(fermi_line.split())
                        dos_info['fermi_energy'] = efermi
                        print(f"费米能级: {efermi} eV")
            except Exception as e:
                print(f"解析OUTCAR失败: {e}")
        
        # 6.2 从TDOS.dat解析DOS数据
        if "TDOS.dat" in downloaded_info:
            try:
                # 读取TDOS数据
                tdos_data = np.loadtxt(downloaded_info["TDOS.dat"])
                if len(tdos_data.shape) == 2:
                    energies = tdos_data[:, 0]
                    dos_values = tdos_data[:, 1] if tdos_data.shape[1] > 1 else tdos_data[:, 1:]
                    
                    # 计算积分态密度（电子数）
                    if 'fermi_energy' in dos_info:
                        efermi = dos_info['fermi_energy']
                        # 找到费米能级附近的索引
                        fermi_idx = np.argmin(np.abs(energies - efermi))
                        # 计算费米能级以下的积分
                        if fermi_idx > 0:
                            integrated_dos = np.trapezoid(dos_values[:fermi_idx], energies[:fermi_idx])
                            dos_info['integrated_dos'] = integrated_dos
                            print(f"费米能级以下积分态密度: {integrated_dos:.4f} 电子数")
                    
                    # 计算DOS最大值和位置
                    max_dos_idx = np.argmax(dos_values)
                    dos_info['max_dos'] = dos_values[max_dos_idx]
                    dos_info['max_dos_energy'] = energies[max_dos_idx]
                    print(f"最大DOS值: {dos_values[max_dos_idx]:.4f} states/eV at {energies[max_dos_idx]:.4f} eV")
                    
                    # 计算DOS宽度（超过阈值的能量范围）
                    threshold = 0.01 * dos_info['max_dos']  # 最大值的1%作为阈值
                    significant_indices = np.where(dos_values > threshold)
                    if len(significant_indices) > 0:
                        dos_info['dos_width'] = energies[significant_indices[-1]] - energies[significant_indices[0]]
                        print(f"DOS宽度: {dos_info['dos_width']:.4f} eV")
            except Exception as e:
                print(f"解析TDOS.dat失败: {e}")
        
                    
            except ImportError:
                print("pymatgen未安装，跳过详细解析")
            except Exception as e:
                print(f"pymatgen解析失败: {e}")
        return {
            "status": "提取完成",
            "dos_info": dos_info,
            "local_files": downloaded_info,
        }

    def extract_file(self, file_path: str) -> dict:
        """
        从计算服务器中提取指定文件并返回本地保存地址。
        """
        if not isinstance(file_path, str) or not file_path.strip():
            return {
                "status": "error",
                "message": "file_path 不能为空"
            }

        remote_file_path = file_path.strip()
        output_dir = "./calculation_output/any"
        os.makedirs(output_dir, exist_ok=True)
        local_file_path = os.path.join(output_dir, os.path.basename(remote_file_path))

        # 允许覆盖：如果本地文件已存在，则先删除
        if os.path.exists(local_file_path):
            try:
                os.remove(local_file_path)
            except Exception as e:
                # 删除失败也不终止下载，继续尝试覆盖
                print(f"警告：无法删除旧文件 {local_file_path}：{e}")

        try:
            self.sftp.get(remote_file_path, local_file_path)
        except Exception as e:
            return {
                "status": "error",
                "message": f"下载文件失败: {e}",
                "remote_file": remote_file_path,
                "local_file": local_file_path
            }

        return {
            "status": "ok",
            "message": "文件已保存",
            "remote_file": remote_file_path,
            "local_file": local_file_path
        }

    def execute_command(self, command: str) -> dict:

            """
            在远程服务器执行命令
            """
            if not isinstance(command, str) or not command.strip():
                return {
                    "status": "error",
                    "message": "命令为空或无效"
                }

            normalized = command.strip().lower()

            # 禁止的危险命令关键字/模式
            deny_patterns = [
                r"\brm\s+-rf\b",
                r"\brm\s+-r\b",
                r"\brm\s+-f\b",
                r"\bshutdown\b",
                r"\breboot\b",
                r"\bpoweroff\b",
                r"\bsystemctl\s+reboot\b",
                r"\bsystemctl\s+poweroff\b",
                r"\binit\s+0\b",
                r"\bmkfs\b",
                r"\bdd\b",
                r"\bchmod\s+[a-z0-9]*777\b",
                r"\bchown\b",
                r"\bpasswd\b",
                r"\buseradd\b",
                r"\buserdel\b",
                r"\bsudo\b",
                r"\bcurl\b.*\|.*sh",
                r"\bwget\b.*\|.*sh",
                r"\bpython\s+-c\b",
                r"\bperl\s+-e\b",
                r"\bphp\s+-r\b",
                r"\bpkill\b",
                r"\bkillall\b"
            ]

            for pat in deny_patterns:
                if re.search(pat, normalized):
                    return {
                        "status": "rejected",
                        "message": "检测到危险命令，已拒绝执行",
                        "reason": f"危险命令匹配: {pat}"
                    }

            # 禁止常见的直接删除根目录或写入系统文件
            if re.search(r"\brm\s+-rf\s+/\b", normalized) or re.search(r"\b>\s*/etc/", normalized):
                return {
                    "status": "rejected",
                    "message": "检测到危险命令，已拒绝执行",
                    "reason": "禁止删除根目录或覆盖系统文件"
                }

            try:
                stdin, stdout, stderr = self.ssh.exec_command(command)
                out = stdout.read().decode(errors='ignore')
                err = stderr.read().decode(errors='ignore')
                return {
                    "status": "ok",
                    "command": command,
                    "stdout": out,
                    "stderr": err
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": "远程执行命令失败",
                    "error": str(e)
                }

    def excute_python(self, command: str) -> dict:
        """
        在远程服务器上执行 Python 命令（当作 shell 命令执行）
        """
        if not isinstance(command, str) or not command.strip():
            return {
                "status": "error",
                "message": "Python 命令为空或无效"
            }

        normalized = command.strip().lower()
        unsafe_patterns = [
            r"\bimport\s+os\b",
            r"\bimport\s+subprocess\b",
            r"\bos\.system\b",
            r"\bsubprocess\.Popen\b",
            r"\bsubprocess\.call\b",
            r"\bopen\(.*['\"/]etc['\"]",
            r"\b__import__\b",
            r"\beval\b",
            r"\bexec\b",
            r"\bcompile\b",
            r"\bimportlib\b",
            r"\bsys\.exit\b",
            r"\bexit\(\)\b"
        ]

        for pat in unsafe_patterns:
            if re.search(pat, normalized):
                return {
                    "status": "rejected",
                    "message": "检测到危险 Python 操作，已拒绝执行",
                    "reason": f"危险模式匹配: {pat}"
                }

        # 如果需要只允许有限的 Python 语句，下面可以额外添加白名单逻辑
        try:
            # 这里通过在远程 shell 执行 python -c 来运行
            safe_payload = command.replace('"', '\\"').replace('$', '\\$')
            remote_cmd = f"python3 -c \"{safe_payload}\""
            stdin, stdout, stderr = self.ssh.exec_command(remote_cmd)
            out = stdout.read().decode(errors='ignore')
            err = stderr.read().decode(errors='ignore')
            return {
                "status": "ok",
                "command": remote_cmd,
                "stdout": out,
                "stderr": err
            }
        except Exception as e:
            return {
                "status": "error",
                "message": "远程执行 Python 命令失败",
                "error": str(e)
            }

    # ==================== 新增方法：分步任务控制 ====================

    def create_relax_mission(self, task_directory: str) -> dict:
        """
        仅创建结构优化输入文件（执行 auto_relax_step1.sh）
        
        Args:
            task_directory: 任务目录路径
            
        Returns:
            执行结果字典
        """
        cmd = f"cd '{task_directory}' && bash ../auto_relax_step1.sh"
        return self.execute_command(cmd)

    def create_scf_mission(self, task_directory: str) -> dict:
        """
        仅创建自洽计算输入文件（执行 auto_scf_step1.sh）
        
        Args:
            task_directory: 任务目录路径
            
        Returns:
            执行结果字典
        """
        cmd = f"cd '{task_directory}' && bash ../auto_scfc_step1.sh"
        return self.execute_command(cmd)

    def create_band_mission(self, task_directory: str) -> dict:
        """
        仅创建能带计算输入文件（执行 auto_band_step1.sh）
        
        Args:
            task_directory: 任务目录路径
            
        Returns:
            执行结果字典
        """
        cmd = f"cd '{task_directory}' && bash ../auto_band_step1.sh"
        return self.execute_command(cmd)

    def create_dos_mission(self, task_directory: str) -> dict:
        """
        仅创建态密度计算输入文件（执行 auto_dos_step1.sh）
        
        Args:
            task_directory: 任务目录路径
            
        Returns:
            执行结果字典
        """
        cmd = f"cd '{task_directory}' && bash ../auto_dos_step1.sh"
        return self.execute_command(cmd)

    def submit_relax_calculation(self, task_directory: str) -> dict:
        """
        仅提交结构优化计算任务（执行 auto_relax_step2.sh）
        
        Args:
            task_directory: 任务目录路径
            
        Returns:
            执行结果字典，包含作业ID（如果提交成功）
        """
        cmd = f"cd '{task_directory}' && bash ../auto_relax_step2.sh"
        result = self.execute_command(cmd)
        # 尝试从输出中提取作业ID
        if result.get("status") == "ok":
            stdout = result.get("stdout", "")
            import re
            match = re.search(r"作业ID:\s*(\d+)", stdout)
            if match:
                result["job_id"] = match.group(1)
            elif "Submitted batch job" in stdout:
                match = re.search(r"Submitted batch job (\d+)", stdout)
                if match:
                    result["job_id"] = match.group(1)
        return result

    def submit_scf_calculation(self, task_directory: str) -> dict:
        """
        仅提交自洽计算任务（执行 auto_scf_step2.sh）
        
        Args:
            task_directory: 任务目录路径
            
        Returns:
            执行结果字典
        """
        cmd = f"cd '{task_directory}' && bash ../auto_scfc_step2.sh"
        result = self.execute_command(cmd)
        if result.get("status") == "ok":
            stdout = result.get("stdout", "")
            import re
            if "Submitted batch job" in stdout:
                match = re.search(r"Submitted batch job (\d+)", stdout)
                if match:
                    result["job_id"] = match.group(1)
        return result

    def submit_band_calculation(self, task_directory: str) -> dict:
        """
        仅提交能带计算任务（执行 auto_band_step2.sh）
        
        Args:
            task_directory: 任务目录路径
            
        Returns:
            执行结果字典
        """
        cmd = f"cd '{task_directory}' && bash ../auto_band_step2.sh"
        result = self.execute_command(cmd)
        if result.get("status") == "ok":
            stdout = result.get("stdout", "")
            import re
            if "Submitted batch job" in stdout:
                match = re.search(r"Submitted batch job (\d+)", stdout)
                if match:
                    result["job_id"] = match.group(1)
        return result

    def submit_dos_calculation(self, task_directory: str) -> dict:
        """
        仅提交态密度计算任务（执行 auto_dos_step2.sh）
        
        Args:
            task_directory: 任务目录路径
            
        Returns:
            执行结果字典
        """
        cmd = f"cd '{task_directory}' && bash ../auto_dos_step2.sh"
        result = self.execute_command(cmd)
        if result.get("status") == "ok":
            stdout = result.get("stdout", "")
            import re
            if "Submitted batch job" in stdout:
                match = re.search(r"Submitted batch job (\d+)", stdout)
                if match:
                    result["job_id"] = match.group(1)
        return result

    def modify_incar_file(self, task_directory: str, mission: str, read_mode: bool = True, 
                          new_params: dict = None) -> dict:
        """
        读写修改INCAR文件
        
        Args:
            task_directory: 任务目录路径
            mission: 计算类型 ('relax', 'scf', 'band', 'dos')
            read_mode: True表示读取INCAR参数，False表示写入新参数
            new_params: 写入模式时的新参数字典
            
        Returns:
            读取模式：返回INCAR参数字典
            写入模式：返回操作结果
        """
        # 确定子目录
        subdir_map = {
            "relax": "结构优化",
            "scf": "自洽计算",
            "band": "能带计算",
            "dos": "态密度计算"
        }
        if mission not in subdir_map:
            return {
                "status": "error",
                "message": f"未知的计算类型: {mission}，可选: {list(subdir_map.keys())}"
            }
        
        subdir = subdir_map[mission]
        remote_path = f"{task_directory}/{subdir}/INCAR"
        
        if read_mode:
            # 读取INCAR文件
            try:
                # 下载文件到本地临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.incar', delete=False) as tmp:
                    tmp_path = tmp.name
                self.sftp.get(remote_path, tmp_path)
                
                # 使用pymatgen解析INCAR
                from pymatgen.io.vasp import Incar
                incar = Incar.from_file(tmp_path)
                params = incar.as_dict()
                
                # 清理临时文件
                import os
                os.unlink(tmp_path)
                
                return {
                    "status": "ok",
                    "mission": mission,
                    "incar_params": params,
                    "file_path": remote_path
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"读取INCAR文件失败: {str(e)}",
                    "error": str(e)
                }
        else:
            # 写入模式：更新INCAR参数
            if new_params is None:
                return {
                    "status": "error",
                    "message": "写入模式需要提供new_params参数"
                }
            
            try:
                # 下载当前INCAR
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.incar', delete=False) as tmp:
                    tmp_path = tmp.name
                self.sftp.get(remote_path, tmp_path)
                
                # 解析并更新
                from pymatgen.io.vasp import Incar
                incar = Incar.from_file(tmp_path)
                
                # 更新参数
                for key, value in new_params.items():
                    incar[key] = value
                
                # 保存回临时文件
                incar.write_file(tmp_path)
                
                # 上传更新后的文件
                self.sftp.put(tmp_path, remote_path)
                
                # 清理临时文件
                import os
                os.unlink(tmp_path)
                
                return {
                    "status": "ok",
                    "message": f"INCAR文件已更新，修改了 {len(new_params)} 个参数",
                    "updated_params": list(new_params.keys()),
                    "file_path": remote_path
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"更新INCAR文件失败: {str(e)}",
                    "error": str(e)
                }


if __name__ == "__main__":
    pass
