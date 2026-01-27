import pandas as pd
import numpy as np
from itertools import chain
from collections import defaultdict

class ImprovedMolecularOrbitals:
    """
    使用NIST原子轨道数据库计算材料能带边缘(HOMO/LUMO)
    改进点：
    1. 使用精确的原子轨道能量数据
    2. 考虑不同原子的电荷状态
    3. 更准确的电子填充算法
    4. 处理部分填充轨道的金属/非金属判断
    """
    
    def __init__(self, formula: dict, orbital_data: pd.DataFrame = None):
        """
        Args:
            orbital_data: 包含轨道能量的DataFrame(必须包含'e_*'和'occ_*'列)
            formula: 化学式字典，如 {'Sr':1, 'Ti':1, 'O':3}
        """
        self.orbital_data = orbital_data if orbital_data is not None else pd.read_csv("./myml/nist_atomic_data_lda(eV).csv")
        self.formula = formula
        self.elements = list(formula.keys())
        self.composition = {elem: int(count) for elem, count in formula.items()}
        
        # 处理原子轨道数据
        self.prepare_atomic_data()
        
        # 计算能带边缘
        self.band_edges = self.calculate_band_edges()
    
    def prepare_atomic_data(self):
        """预处理轨道数据，构建每种元素的轨道列表"""
        self.atomic_orbitals = {}
        self.electronegativity = {}
        
        # 获取所有轨道能量列
        energy_cols = [c for c in self.orbital_data.columns if c.startswith('e_')]
        
        for element in self.elements:
            # 从数据中获取该元素的行
            element_row = self.orbital_data[self.orbital_data['element'] == element]
            
            if element_row.empty:
                raise ValueError(f"元素 {element} 不在数据库中找到")
            
            # 获取电负性
            # self.electronegativity[element] = Element(element).X
            
            # 构建轨道列表
            orbitals = []
            for e_col in energy_cols:
                # 解析轨道名称（如'1s'）
                orb_name = e_col.split('_')[1]
                energy = element_row[e_col].values[0]
                
                # 获取占据数列名
                occ_col = 'occ_' + orb_name
                if occ_col in self.orbital_data.columns:
                    occupation = element_row[occ_col].values[0]
                else:
                    # 如果没有占据数，根据轨道类型估计
                    orb_type = orb_name[-1]
                    occupation = {'s': 2, 'p': 6, 'd': 10, 'f': 14}.get(orb_type, 0)
                
                orbitals.append({
                    'name': orb_name,
                    'energy': energy,
                    'occupation': occupation,
                    'capacity': self.get_orbital_capacity(orb_name)
                })
            
            # 按能量排序
            orbitals.sort(key=lambda x: x['energy'])
            self.atomic_orbitals[element] = orbitals
    
    @staticmethod
    def get_orbital_capacity(orb_name):
        """返回轨道的最大电子容量"""
        orb_type = orb_name[-1]
        return {
            's': 2, 'p': 6, 'd': 10, 'f': 14
        }.get(orb_type, 0)
    
    def calculate_total_electrons(self):
        """计算材料的总电子数"""
        total_electrons = 0
        for element, count in self.composition.items():
            # 从数据中获取原子序数
            element_row = self.orbital_data[self.orbital_data['element'] == element]
            atomic_number = element_row['atomic_number'].values[0]
            total_electrons += atomic_number * count
        return total_electrons
    
    def build_composite_orbitals(self):
        """构建材料的复合轨道列表"""
        composite_orbitals = []
        
        for element, count in self.composition.items():
            orbitals = self.atomic_orbitals[element]
            # 对于每种原子，根据化学计量数复制其轨道
            for _ in range(count):
                for orb in orbitals:
                    # 创建新的轨道对象，避免引用问题
                    composite_orbitals.append({
                        'element': element,
                        'name': orb['name'],
                        'energy': orb['energy'],
                        'capacity': orb['capacity'],
                        'assigned': 0  # 初始化为0分配电子
                    })
        
        # 按能量排序
        composite_orbitals.sort(key=lambda x: x['energy'])
        return composite_orbitals
    
    def calculate_band_edges(self):
        """精确计算能带边缘"""
        # 计算总电子数
        total_electrons = self.calculate_total_electrons()
        orbitals = self.build_composite_orbitals()
        remaining_electrons = total_electrons
        
        # 电子填充算法
        for i, orb in enumerate(orbitals):
            if remaining_electrons <= 0:
                break
                
            # 可分配电子数 = min(轨道剩余容量, 剩余电子数)
            to_assign = min(orb['capacity'] - orb['assigned'], remaining_electrons)
            orb['assigned'] += to_assign
            remaining_electrons -= to_assign
        
        # 寻找HOMO和LUMO
        homo = None
        lumo = None
        
        # 先找到最后一个被占据的轨道 (HOMO)
        for i in range(len(orbitals) - 1, -1, -1):
            if orbitals[i]['assigned'] > 0:
                homo = {
                    'element': orbitals[i]['element'],
                    'orbital': orbitals[i]['name'],
                    'energy': orbitals[i]['energy'],
                    'occupation': orbitals[i]['assigned']
                }
                homo_index = i
                break
        
        # 再找到第一个未完全占据的轨道 (LUMO)
        for i in range(len(orbitals)):
            if orbitals[i]['assigned'] < orbitals[i]['capacity']:
                lumo = {
                    'element': orbitals[i]['element'],
                    'orbital': orbitals[i]['name'],
                    'energy': orbitals[i]['energy'],
                    'occupation': orbitals[i]['assigned']
                }
                lumo_index = i
                break
        
        # 判断是否为金属
        # 1. 存在部分填充轨道 (HOMO和LUMO相同)
        # 2. HOMO和LUMO能量差接近0
        is_metal = False
        if homo and lumo:
            # 如果电子没有填满，或者HOMO和LUMO是同一个轨道
            is_metal = (lumo_index == homo_index) or (abs(lumo['energy'] - homo['energy']) < 0.01)
        
        return {
            'HOMO': homo,
            'LUMO': lumo,
            'metal': is_metal,
            'total_electrons': total_electrons,
            'remaining_electrons': remaining_electrons
        }
    
    def get_band_edges_summary(self):
        """获取格式化的能带边缘信息"""
        if not self.band_edges:
            return "无法计算能带边缘"
        
        homo = self.band_edges['HOMO']
        lumo = self.band_edges['LUMO']
        metal = self.band_edges['metal']
        
        summary = f"材料 {'是' if metal else '不是'} 金属\n"
        summary += f"总电子数: {self.band_edges['total_electrons']}\n"
        
        if homo:
            summary += f"HOMO: {homo['element']} {homo['orbital']} (能量: {homo['energy']:.4f} eV, 占据数: {homo['occupation']})\n"
        else:
            summary += "HOMO: 未找到\n"
        
        if lumo:
            summary += f"LUMO: {lumo['element']} {lumo['orbital']} (能量: {lumo['energy']:.4f} eV, 占据数: {lumo['occupation']})\n"
        else:
            summary += "LUMO: 未找到\n"
        
        if homo and lumo:
            gap = lumo['energy'] - homo['energy']
            summary += f"带隙: {gap:.4f} eV"
        
        return summary
    
    def get_data(self) -> dict:
        """
        返回HOMO, LUMO, gap
        """
        if not self.band_edges:
            return 0
        homo = self.band_edges['HOMO']['energy']
        lumo = self.band_edges['LUMO']['energy']
        gap = lumo - homo
        return {'HOMO':homo, 'LUMO':lumo, 'gap':gap}


# 使用示例
if __name__ == "__main__":
    # 假设您已经从CSV加载了数据到dataframe
    df = pd.read_csv("./myml/nist_atomic_data_lda(eV).csv")
    
    # 测试数据（实际使用时应加载完整CSV）
    # test_data = [
    #     {'element': 'H', 'atomic_number': 1, 'e_1s': -13.6, 'occ_1s': 1},
    #     {'element': 'O', 'atomic_number': 8, 'e_1s': -543.1, 'e_2s': -41.6, 'e_2p': -18.6, 'occ_1s': 2, 'occ_2s': 2, 'occ_2p': 4},
    #     {'element': 'Ti', 'atomic_number': 22, 'e_1s': -4966.3, 'e_2s': -597.9, 'e_2p': -476.3, 'e_3s': -88.7, 'e_3p': -67.4, 'e_3d': -10.5, 'e_4s': -7.9, 
    #      'occ_1s': 2, 'occ_2s': 2, 'occ_2p': 6, 'occ_3s': 2, 'occ_3p': 6, 'occ_3d': 2, 'occ_4s': 2}
    # ]
    orbital_df = pd.DataFrame(df)
    
    # 计算SrTiO3的能带边缘
    formula = {'Sr': 1, 'Ti': 1, 'O': 3}
    calculator = ImprovedMolecularOrbitals(formula, orbital_df)
    
    print("="*50)
    print("材料: SrTiO3")
    print(calculator.get_band_edges_summary())
    print("="*50)
    print(calculator.get_data())