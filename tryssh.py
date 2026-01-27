import paramiko
import os
from datetime import datetime
from pymatgen.core import Structure
from pymatgen.io import vasp
from typing import Dict, List, Optional, Union
import loadenv
import re


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
            subfolders = ["自洽计算", "结构优化", "态密度计算", "能带结构计算"]
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

    def opt(self, task_dir):
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

    def extract_opt_info(self, task_dir):
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
            



if __name__ == "__main__":
    pass
