import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
from myml import featurizer as ft
# 离子符号改为原子符号后的完整数据字典（键名：ionic radius/ionic potential 保持不变）
ion_data = {
    # 金属原子（源自文档Table S1，离子符号→原子符号）
    "Li": {"ionic radius": 0.78, "ionic potential": 12.8, "charge": "+1"},
    "In": {"ionic radius": 0.8, "ionic potential": 37.5, "charge": "+3"},
    "Sc": {"ionic radius": 0.745, "ionic potential": 40.27, "charge": "+3"},
    "Zr": {"ionic radius": 0.72, "ionic potential": 55.56, "charge": "+4"},
    "Hf": {"ionic radius": 0.71, "ionic potential": 56.34, "charge": "+4"},
    "Ta": {"ionic radius": 0.64, "ionic potential": 78.13, "charge": "+5"},
    "Be": {"ionic radius": 0.45, "ionic potential": 27.78, "charge": "+2"},
    "Mg": {"ionic radius": 0.72, "ionic potential": 27.78, "charge": "+2"},
    "Al": {"ionic radius": 0.535, "ionic potential": 56.07, "charge": "+3"},
    "Ca": {"ionic radius": 1.0, "ionic potential": 20.0, "charge": "+2"},
    "Ti": {"ionic radius": 0.605, "ionic potential": 66.12, "charge": "+4"},
    "V": {"ionic radius": 0.64, "ionic potential": 46.88, "charge": "+3"},
    "Cr": {"ionic radius": 0.615, "ionic potential": 48.78, "charge": "+3"},
    "Mn": {"ionic radius": 0.67, "ionic potential": 29.85, "charge": "+2"},
    "Fe": {"ionic radius": 0.645, "ionic potential": 46.51, "charge": "+3"},
    "Co": {"ionic radius": 0.65, "ionic potential": 30.77, "charge": "+2"},
    "Ni": {"ionic radius": 0.69, "ionic potential": 28.99, "charge": "+2"},
    "Cu": {"ionic radius": 0.73, "ionic potential": 27.4, "charge": "+2"},
    "Zn": {"ionic radius": 0.74, "ionic potential": 27.03, "charge": "+2"},
    "Ga": {"ionic radius": 0.625, "ionic potential": 48.0, "charge": "+3"},
    "Ge": {"ionic radius": 0.53, "ionic potential": 75.47, "charge": "+4"},
    "Sr": {"ionic radius": 1.18, "ionic potential": 16.95, "charge": "+2"},
    "Y": {"ionic radius": 0.9, "ionic potential": 33.33, "charge": "+3"},
    "Nb": {"ionic radius": 0.64, "ionic potential": 78.13, "charge": "+5"},
    "Mo": {"ionic radius": 0.61, "ionic potential": 81.97, "charge": "+5"},
    "Tc": {"ionic radius": 0.645, "ionic potential": 62.02, "charge": "+4"},
    "Ru": {"ionic radius": 0.68, "ionic potential": 44.12, "charge": "+3"},
    "Rh": {"ionic radius": 0.665, "ionic potential": 45.11, "charge": "+3"},
    "Pd": {"ionic radius": 0.86, "ionic potential": 23.26, "charge": "+2"},
    "Ag": {"ionic radius": 1.15, "ionic potential": 8.7, "charge": "+1"},
    "Cd": {"ionic radius": 0.95, "ionic potential": 21.05, "charge": "+2"},
    "Sn": {"ionic radius": 0.69, "ionic potential": 57.97, "charge": "+4"},
    "Sb": {"ionic radius": 0.6, "ionic potential": 50.0, "charge": "+3"},
    "Ba": {"ionic radius": 1.35, "ionic potential": 14.81, "charge": "+2"},
    "La": {"ionic radius": 1.03, "ionic potential": 29.13, "charge": "+3"},
    "Ce": {"ionic radius": 1.01, "ionic potential": 29.7, "charge": "+3"},
    "Pr": {"ionic radius": 0.99, "ionic potential": 30.3, "charge": "+3"},
    "Nd": {"ionic radius": 0.983, "ionic potential": 30.52, "charge": "+3"},
    "Sm": {"ionic radius": 0.958, "ionic potential": 30.32, "charge": "+3"},
    "Eu": {"ionic radius": 0.947, "ionic potential": 31.68, "charge": "+3"},
    "Gd": {"ionic radius": 0.938, "ionic potential": 31.98, "charge": "+3"},
    "Tb": {"ionic radius": 0.923, "ionic potential": 32.5, "charge": "+3"},
    "Dy": {"ionic radius": 0.912, "ionic potential": 32.89, "charge": "+3"},
    "Ho": {"ionic radius": 0.901, "ionic potential": 33.3, "charge": "+3"},
    "Er": {"ionic radius": 0.89, "ionic potential": 30.71, "charge": "+3"},
    "Tm": {"ionic radius": 0.88, "ionic potential": 34.09, "charge": "+3"},
    "Yb": {"ionic radius": 0.868, "ionic potential": 34.56, "charge": "+3"},
    "Lu": {"ionic radius": 0.861, "ionic potential": 34.84, "charge": "+3"},
    "W": {"ionic radius": 0.6, "ionic potential": 100.0, "charge": "+6"},
    "Re": {"ionic radius": 0.63, "ionic potential": 63.49, "charge": "+4"},
    "Os": {"ionic radius": 0.63, "ionic potential": 63.49, "charge": "+4"},
    "Ir": {"ionic radius": 0.68, "ionic potential": 44.12, "charge": "+3"},
    "Pt": {"ionic radius": 0.625, "ionic potential": 64.0, "charge": "+4"},
    "Au": {"ionic radius": 1.37, "ionic potential": 7.3, "charge": "+1"},
    "Hg": {"ionic radius": 1.02, "ionic potential": 19.61, "charge": "+2"},
    "Tl": {"ionic radius": 1.5, "ionic potential": 6.67, "charge": "+1"},
    "Pb": {"ionic radius": 1.19, "ionic potential": 16.81, "charge": "+2"},
    "Bi": {"ionic radius": 1.03, "ionic potential": 29.13, "charge": "+3"},
    "Po": {"ionic radius": 0.94, "ionic potential": 31.91, "charge": "+3"},
    # 卤素原子（补充数据，电荷对应原始卤素离子价态）
    "F": {"ionic radius": 1.19, "ionic potential": 8.403, "charge": "-1"},
    "Cl": {"ionic radius": 1.67, "ionic potential": 5.988, "charge": "-1"},
    "Br": {"ionic radius": 1.82, "ionic potential": 5.495, "charge": "-1"},
    "I": {"ionic radius": 2.06, "ionic potential": 4.854, "charge": "-1"}
}
element_features = pd.read_csv('./myml/element_features.csv').set_index('element')
for element in ion_data.keys():
    ion_data[element]['electronegativity'] = element_features.loc[element, 'Electronegativity(Pauling)']
    ion_data[element]['ion_electronegativity_potential'] = ion_data[element]['electronegativity'] * ion_data[element]['ionic potential']
    ion_data[element]['ion_radius_electronegativity'] = ion_data[element]['ionic radius'] * ion_data[element]['electronegativity']
    ion_data[element]['ion_radius_potential_ratio'] = ion_data[element]['ionic radius'] / ion_data[element]['ionic potential']
    ion_data[element]['ion_potential_electronegativity_ratio'] = ion_data[element]['ionic potential'] / ion_data[element]['electronegativity']
# 示例：获取 HE-5 关键组成原子（In、Sc、Zr、Hf、Ta）的核心参数
# he5_key_atoms = ["In", "Sc", "Zr", "Hf", "Ta"]
# for atom in he5_key_atoms:
#     radius = ion_data[atom]["ionic radius"]
#     potential = ion_data[atom]["ionic potential"]
#     # print(f"Atom: {atom} | Ionic Radius: {radius} Å | Ionic Potential: {potential} nm⁻¹")


def new_formula(formula:str)-> Composition: 
    # 把原来的化学式转为标准化学式
    nm_formula = ft.normalize_formula(formula)
    # print(f"原始化学式：{formula}，标准化：{nm_formula}",end='，')
    nm_formula = str(''.join([f"{k}{v if v>1 else ''}" for k,v in nm_formula.items()]))
    # print(nm_formula)
    composition = Composition(nm_formula)
    return composition
# df["composition"] = df["formula"].apply(new_formula)
# df['normal_formula'] = df["formula"].apply(ft.normalize_formula)
# df.head()


def culculate_polarization_factors(normal_formula):
    """
    输入可以是 pymatgen Composition 或 dict-like（元素->计量数）。
    返回包含极化因子的一致字典（保证总是返回 dict）。
    """
    # 兼容 pymatgen Composition 和字典
    try:
        comp_dict = normal_formula.get_el_amt_dict()  # pymatgen Composition
    except Exception:
        comp_dict = dict(normal_formula)  # 已经是 dict 或类似结构

    phi_Li = 0.0
    r_ratio_Li = 0.0
    volume_ratio_Li = 0.0
    sum_phi_M = 0.0
    sum_phi_X = 0.0
    sum_r_ratio_M = 0.0
    sum_r_ratio_X = 0.0
    n = sum(comp_dict.values()) if len(comp_dict) > 0 else 0
    if n == 0:
        # 无成分，返回空特征（或全 NaN）
        return {
            'tau': np.nan, 'r_ratio': np.nan, 'volume_ratio': np.nan,
            'phi_Li': np.nan, 'r_ratio_Li': np.nan, 'volume_ratio_Li': np.nan,
            'sum_phi_M': np.nan, 'sum_r_ratio_M': np.nan, 'sum_volume_ratio_M': np.nan,
            'sum_phi_X': np.nan, 'sum_r_ratio_X': np.nan, 'sum_volume_ratio_X': np.nan,
        }

    for elem, coeff in comp_dict.items():
        # elem 已为字符串（来自 get_el_amt_dict）或可能是 Element 对象的 name
        elem_sym = elem if isinstance(elem, str) else str(elem)

        if elem_sym not in ion_data:
            # 找不到时跳过（或根据需要改为抛错）
            # print(f"警告: ion_data 中缺失元素 {elem_sym}，已忽略")
            continue

        r = ion_data[elem_sym]["ionic radius"]
        phi = ion_data[elem_sym]["ionic potential"]

        if elem_sym == "Li":
            phi_Li += phi * coeff / n
            r_ratio_Li += r * coeff / n
            volume_ratio_Li += 4 * np.pi / 3 * r**3 * coeff / n
        elif elem_sym in ["F", "Cl", "Br", "I"]:
            sum_phi_X += phi * coeff / n
            sum_r_ratio_X += r * coeff / n
        else:
            sum_phi_M += phi * coeff / n
            sum_r_ratio_M += r * coeff / n

    # 安全除零
    tau = (phi_Li + sum_phi_M) / sum_phi_X if sum_phi_X != 0 else np.nan
    Li_radius_ratio = r_ratio_Li / sum_r_ratio_X if sum_r_ratio_X != 0 else np.nan
    M_radius_ratio = sum_r_ratio_M / sum_r_ratio_X if sum_r_ratio_X != 0 else np.nan
    M_phi_ratio = sum_phi_M / sum_phi_X if sum_phi_X !=0 else np.nan
    radius_ratio = Li_radius_ratio + M_radius_ratio

    # Descriptor_ESB = Σ [ (q_Li * q_neighbor) / (r_Li + r_neighbor) ] * r_neighbor_ratio,锂离子与晶格骨架的等效静电束缚能,该值越小，表示键强越弱，​模拟的迁移率应越高。
    # Descriptor_bottleneck = (r_bottleneck - r_ion) / r_ion 是迁移路径"瓶颈"处的半径,该值越大，表示迁移通道越宽敞，几何约束越小，​模拟的迁移率应越高。
    Descriptor_ESB = 0.0
    Descriptor_bottleneck = 0.0
    for elem, coeff in comp_dict.items():
        # elem 已为字符串（来自 get_el_amt_dict）或可能是 Element 对象的 name
        elem_sym = elem if isinstance(elem, str) else str(elem)

        if elem_sym not in ion_data:
            # 找不到时跳过（或根据需要改为抛错）
            # print(f"警告: ion_data 中缺失元素 {elem_sym}，已忽略")
            continue
        r = ion_data[elem_sym]["ionic radius"]
        q = abs(int(ion_data[elem_sym]["charge"]))
        if elem_sym != "Li":
            r_ratio_neighbor = r / sum_r_ratio_X if sum_r_ratio_X != 0 else 0.0
            Descriptor_ESB += (1.0 * q) / (r_ratio_Li + r) * r_ratio_neighbor * (coeff / n)
            Descriptor_bottleneck += (r - r_ratio_Li) / r_ratio_Li * r_ratio_neighbor * (coeff / n)

    
    bottleneck_bse_ratio = Descriptor_bottleneck / Descriptor_ESB if Descriptor_ESB !=0 else np.nan
    features = {
        'tau': tau,
        'cation_radius_ratio': radius_ratio,
        'phi_Li': phi_Li,
        'Li/X_radius': Li_radius_ratio,
        'M/X_radius': M_radius_ratio,
        'M/X_phi': M_phi_ratio,
        'phi_M': sum_phi_M,
        'radius_ratio_M': sum_r_ratio_M,
        'phi_X': sum_phi_X,
        'radius_ratio_X': sum_r_ratio_X,
        'Descriptor_ESB': Descriptor_ESB,
        'Descriptor_bottleneck': Descriptor_bottleneck,
        'bottleneck_bse_ratio': bottleneck_bse_ratio
    }
    return features

def add_polarization_factors(df):
    """
    为数据框 df 添加极化因子特征列（从 composition 列计算）
    """
    feature_dicts = df['normal_formula'].apply(culculate_polarization_factors)
    feature_df = pd.DataFrame(feature_dicts.tolist())
    return pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

# df = add_polarization_factors(df)
# df.head()

from matminer.featurizers.composition.packing import AtomicPackingEfficiency
from matminer.featurizers.composition.element import Stoichiometry
# df = Stoichiometry().featurize_dataframe(df, 'composition',ignore_errors=True)
# df = AtomicPackingEfficiency(impute_nan=True).featurize_dataframe(df, 'composition',ignore_errors=True)

drops = ['Etot',
 'Etot/atomic_number^2',
 'Ekin',
 'Ekin/atomic_number^2',
 'Ecoul',
 'Ecoul/atomic_number^2',
 'Eenuc',
 'Eenuc/atomic_number^2',
 'Exc',
 'Exc/atomic_number^2','molar_vol','ENMAX(eV)',
 'EAUG(eV)',
 'EAUG/ENMAX',
 'ZVAL','molar_vol','electric_pol',
 'GGAU_Etot',
 'mus_fere',
 'FERE correction',
 'is_s',
 'is_p',
 'is_d',
 'is_ds',
 'is_f',
 'Row_1',
 'Row_2',
 'Row_3',
 'Row_4',
 'Row_5',
 'Row_6',
 'Row_7',
 'Column_1',
 'Column_2',
 'Column_3',
 'Column_4',
 'Column_5',
 'Column_6',
 'Column_7',
 'Column_8',
 'Column_9',
 'Column_10',
 'Column_11',
 'Column_12',
 'Column_13',
 'Column_14',
 'Column_15',
 'Column_16',
 'Column_17',
 'Column_18',
 'Column',
 'Row',
 'magnetic_electrons',
 'magnetic_moment']
# df = ft.get_all_features(df, drops = drops)

#为锂离子计算特殊离子性特征
electronegativity = element_features['Electronegativity(Pauling)'].to_dict()
def calculate_migration_ion_specific_features(composition, migration_ion='Li'):
    """
    计算与迁移离子相关的特殊离子性特征
    """
    composition = composition.as_dict()
    # print(composition)
    if migration_ion not in composition.keys():
        print(f"没有迁移离子 {migration_ion}，跳过计算特征")
        return {}  # 如果没有迁移离子，返回空字典
    
    migration_features = {}
    migration_chi = electronegativity.get(migration_ion, np.nan)
    if np.isnan(migration_chi):
        return {}
    # 迁移离子与其他元素的离子性
    migration_ionicities = []
    for elem, amount in composition.items():
        if elem == migration_ion:
            continue
        elem_chi = electronegativity.get(elem, np.nan)
        if np.isnan(elem_chi):
            continue
        delta_chi = abs(migration_chi - elem_chi)
        ionicity = 1 - np.exp(-0.25 * delta_chi**2)
        # amount 可能为 float（Composition），强制为 int 用于重复计数
        migration_ionicities.extend([ionicity] * int(amount))
    
    if migration_ionicities:
        migration_features.update({
            f'{migration_ion}_avg_ionicity': np.mean(migration_ionicities),
            f'{migration_ion}_max_ionicity': np.max(migration_ionicities),
            f'{migration_ion}_min_ionicity': np.min(migration_ionicities),
            f'{migration_ion}_std_ionicity': np.std(migration_ionicities),
            f'{migration_ion}_skew_ionicity': pd.Series(migration_ionicities).skew()
            
        })
    
    return migration_features


def add_migration_ion_features(df, migration_ion='Li'):
    """
    为数据框 df 添加与迁移离子相关的特征列
    """
    feature_dicts = df['composition'].apply(lambda comp: calculate_migration_ion_specific_features(comp, migration_ion))
    print(feature_dicts.head())
    feature_df = pd.DataFrame(feature_dicts.tolist())
    return pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

# df = add_migration_ion_features(df, migration_ion='Li')
# df.head()
def culculate_ion_statistics(normal_formula):
    """
    输入可以是 pymatgen Composition 或 dict-like（元素->计量数）。
    返回包含离子统计特征的一致字典（保证总是返回 dict）。
    """ 
    # 兼容 pymatgen Composition 和字典
    try:
        comp_dict = normal_formula.get_el_amt_dict()  # pymatgen Composition
    except Exception:
        comp_dict = dict(normal_formula)  # 已经是 dict 或类似结构
    features_list = [
        'ionic radius',
        'ionic potential',
        'ion_electronegativity_potential',
        'ion_radius_electronegativity',
        'ion_radius_potential_ratio',
        'ion_potential_electronegativity_ratio'
    ]
    
    features = {}
    for feature in features_list:
        values = []
        for elem, coeff in comp_dict.items():
            # elem 已为字符串（来自 get_el_amt_dict）或可能是 Element 对象的 name
            elem_sym = elem if isinstance(elem, str) else str(elem)

            if elem_sym not in ion_data:
                # 找不到时跳过（或根据需要改为抛错）
                # print(f"警告: ion_data 中缺失元素 {elem_sym}，已忽略")
                continue

            value = ion_data[elem_sym][feature]
            values.extend([value] * int(coeff))  # 根据计量数添加多次

        if len(values) == 0:
            # 无有效值，全部设为 NaN
            features[f'mean_{feature}'] = np.nan
            features[f'std_{feature}'] = np.nan
            features[f'skew_{feature}'] = np.nan
            features[f'kurt_{feature}'] = np.nan
        else:
            arr = np.array(values)
            features[f'mean_{feature}'] = np.mean(arr)
            features[f'std_{feature}'] = np.std(arr)
            features[f'skew_{feature}'] = pd.Series(arr).skew()
            features[f'kurt_{feature}'] = pd.Series(arr).kurtosis()
        
    return features



def add_ion_statistics(df):
    """
    为数据框 df 添加离子统计特征列（从 composition 列计算）
    """
    # 这里假设已有 culculate_ion_statistics 函数
    feature_dicts = df['normal_formula'].apply(culculate_ion_statistics)
    feature_df = pd.DataFrame(feature_dicts.tolist())
    return pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)


import numpy as np
from scipy.spatial.distance import pdist, squareform
from pymatgen.core import Composition

def ion_coulomb_matrix_from_composition(composition):
    """
    仅从化学式估算库仑矩阵特征
    
    Args:
        formula: 化学式，如 'Li3PS4'
        method: 距离估算方法 ('covalent', 'ionic', 'vdw')
    """
    
    comp = composition
    elements = list(comp.element_composition.keys())
    amounts = list(comp.element_composition.values())
    n_atoms = sum(amounts)
    
    # 原子序数字典
    atomic_numbers = {elem.symbol: elem.Z for elem in elements}
    
    # 估算平均距离矩阵
    n_elements = len(elements)
    avg_dist_matrix = np.zeros((n_elements, n_elements))
    
    for i, elem_i in enumerate(elements):
        for j, elem_j in enumerate(elements):
            sym_i, sym_j = elem_i.symbol, elem_j.symbol
            if i == j:
                # 同种元素：使用共价半径
                avg_dist_matrix[i, j] = ion_data[sym_i]['ionic radius'] if ion_data[sym_i]['ionic radius'] else 0.6
            else:
                # 不同元素：半径相加
                r_i = ion_data[sym_i]['ionic radius'] if ion_data[sym_i]['ionic radius'] else 0.6
                r_j = ion_data[sym_j]['ionic radius'] if ion_data[sym_j]['ionic radius'] else 0.6
                avg_dist_matrix[i, j] = r_i + r_j
    
    # 构建简化的库仑矩阵
    ion_cm_simple = np.zeros((n_elements, n_elements))
    for i, elem_i in enumerate(elements):
        for j, elem_j in enumerate(elements):
            sym_i, sym_j = elem_i.symbol, elem_j.symbol
            Z_i = elem_i.Z
            Z_j = elem_j.Z
            
            if i == j:
                ion_cm_simple[i, j] = 0.5 * Z_i **2.4
            else:
                if avg_dist_matrix[i, j] > 1e-8:
                    ion_cm_simple[i, j] = Z_i * Z_j / avg_dist_matrix[i, j]
    
    return ion_cm_simple

def ion_cm_to_statistical_features(cm):
    """
    从库仑矩阵提取统计特征
    """
    features = {}
    
    # 矩阵整体统计
    features.update({
        'ion_cm_mean': np.mean(cm),
        'ion_cm_std': np.std(cm),
        'ion_cm_max': np.max(cm),
        'ion_cm_min': np.min(cm),
        'ion_cm_trace': np.trace(cm),
        'ion_cm_frobenius_norm': np.linalg.norm(cm, 'fro')
    })
    
    # 行统计
    row_means = np.mean(cm, axis=1)
    features.update({
        'ion_cm_row_mean_mean': np.mean(row_means),
        'ion_cm_row_mean_std': np.std(row_means),
        'ion_cm_row_mean_max': np.max(row_means),
        'ion_cm_row_mean_min': np.min(row_means)
    })
    
    # 对角线统计
    diagonal = np.diag(cm)
    features.update({
        'ion_cm_diag_mean': np.mean(diagonal),
        'ion_cm_diag_std': np.std(diagonal)
    })
    
    # 非对角线统计
    off_diagonal = cm[np.triu_indices_from(cm, k=1)]
    features.update({
        'ion_cm_offdiag_mean': np.mean(off_diagonal),
        'ion_cm_offdiag_std': np.std(off_diagonal)
    })
    
    return features


def create_ion_coulomb_matrix_features(composition):
    """
    为给定化学式创建库仑矩阵统计特征
    """
    ion_cm = ion_coulomb_matrix_from_composition(composition)
    features = ion_cm_to_statistical_features(ion_cm)
    return features

def add_ion_coulomb_matrix_features(df):
    """
    为数据框 df 添加库仑矩阵统计特征列
    """
    feature_dicts = df['composition'].apply(create_ion_coulomb_matrix_features)
    feature_df = pd.DataFrame(feature_dicts.tolist())
    return pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

# df = add_ion_coulomb_matrix_features(df)
# df.head()

from pymatgen.core.periodic_table import Element
import math
def calculate_all_configurational_entropies(composition):
    """
    使用e_1s作为能量描述符计算七种构型熵的两种加权方式
    
    参数:
        composition: pymatgen的Composition对象
    返回:
        包含所有熵值的字典
    """
    # 获取元素组成信息
    elements = composition.elements
    amounts = [composition[el] for el in elements]
    total_atoms = sum(amounts)
    mole_fractions = [amt / total_atoms for amt in amounts]
    
    # 获取元素符号列表
    element_symbols = [el.symbol for el in elements]
    
    # 获取元素属性
    atomic_numbers = [el.Z for el in elements]
    atomic_masses = [el.atomic_mass for el in elements]
    atomic_radii = [el.atomic_radius for el in elements]
    pauling_electroneg = [el.X for el in elements]
    
    # 获取价电子数
    valence_electrons = []
    for el in elements:
        if el.is_transition_metal:
            valence_electrons.append(0 if el.group in [8, 9, 10] else el.group)
        else:
            valence_electrons.append(el.group)
    
    # 获取e_1s能量值
    e_1s_values = []
    for symbol in element_symbols:
        if symbol in element_features.index:
            e_1s_values.append(element_features.loc[symbol, 'e_1s'])
        else:
            # 如果找不到该元素的e_1s值，使用原子序数*10作为近似
            el = Element(symbol)
            e_1s_values.append(el.Z * 10)
    
    # 1. 传统原子分数熵
    atomic_entropy = 0
    for x in mole_fractions:
        if x > 0:
            atomic_entropy -= x * math.log(x)
    
    # 2. 原子序数分数熵
    z_total = sum(mole_fractions[i] * atomic_numbers[i] for i in range(len(elements)))
    z_fractions = [(mole_fractions[i] * atomic_numbers[i]) / z_total for i in range(len(elements))]
    
    z_entropy_atomic_weighted = 0
    z_entropy_z_weighted = 0
    for i in range(len(elements)):
        if z_fractions[i] > 0:
            z_entropy_atomic_weighted -= mole_fractions[i] * math.log(z_fractions[i])
            z_entropy_z_weighted -= z_fractions[i] * math.log(z_fractions[i])
    
    # 3. 原子质量分数熵
    mass_total = sum(mole_fractions[i] * atomic_masses[i] for i in range(len(elements)))
    mass_fractions = [(mole_fractions[i] * atomic_masses[i]) / mass_total for i in range(len(elements))]
    
    mass_entropy_atomic_weighted = 0
    mass_entropy_mass_weighted = 0
    for i in range(len(elements)):
        if mass_fractions[i] > 0:
            mass_entropy_atomic_weighted -= mole_fractions[i] * math.log(mass_fractions[i])
            mass_entropy_mass_weighted -= mass_fractions[i] * math.log(mass_fractions[i])
    
    # 4. 原子体积分数熵
    volume_factors = [r**3 for r in atomic_radii]
    volume_total = sum(mole_fractions[i] * volume_factors[i] for i in range(len(elements)))
    volume_fractions = [(mole_fractions[i] * volume_factors[i]) / volume_total for i in range(len(elements))]
    
    volume_entropy_atomic_weighted = 0
    volume_entropy_volume_weighted = 0
    for i in range(len(elements)):
        if volume_fractions[i] > 0:
            volume_entropy_atomic_weighted -= mole_fractions[i] * math.log(volume_fractions[i])
            volume_entropy_volume_weighted -= volume_fractions[i] * math.log(volume_fractions[i])
    
    # 5. 原子价电子分数熵
    vec_total = sum(mole_fractions[i] * valence_electrons[i] for i in range(len(elements)))
    vec_fractions = [(mole_fractions[i] * valence_electrons[i]) / vec_total for i in range(len(elements))]
    
    vec_entropy_atomic_weighted = 0
    vec_entropy_vec_weighted = 0
    for i in range(len(elements)):
        if vec_fractions[i] > 0:
            vec_entropy_atomic_weighted -= mole_fractions[i] * math.log(vec_fractions[i])
            vec_entropy_vec_weighted -= vec_fractions[i] * math.log(vec_fractions[i])
    
    # 6. 原子电负性分数熵
    en_total = sum(mole_fractions[i] * pauling_electroneg[i] for i in range(len(elements)))
    en_fractions = [(mole_fractions[i] * pauling_electroneg[i]) / en_total for i in range(len(elements))]
    
    en_entropy_atomic_weighted = 0
    en_entropy_en_weighted = 0
    for i in range(len(elements)):
        if en_fractions[i] > 0:
            en_entropy_atomic_weighted -= mole_fractions[i] * math.log(en_fractions[i])
            en_entropy_en_weighted -= en_fractions[i] * math.log(en_fractions[i])
    
    # 7. 原子能量分数熵 - 使用e_1s作为能量描述符
    # 注意：e_1s是负值，我们需要处理这种情况
    # 方案：取绝对值或使用偏移量，这里我们取绝对值
    e_1s_abs = [abs(e) for e in e_1s_values]  # 取绝对值确保正数
    
    energy_total = sum(mole_fractions[i] * e_1s_abs[i] for i in range(len(elements)))
    energy_fractions = [(mole_fractions[i] * e_1s_abs[i]) / energy_total for i in range(len(elements))]
    
    energy_entropy_atomic_weighted = 0
    energy_entropy_energy_weighted = 0
    for i in range(len(elements)):
        if energy_fractions[i] > 0:
            energy_entropy_atomic_weighted -= mole_fractions[i] * math.log(energy_fractions[i])
            energy_entropy_energy_weighted -= energy_fractions[i] * math.log(energy_fractions[i])
    
    # 乘以气体常数R (8.314 J/mol·K)
    R = 8.314
    
    return {
        # 传统原子分数熵
        "atomic_fraction_entropy": atomic_entropy * R,
        
        # 原子序数分数熵
        "atomic_number_entropy_atomic_weighted": z_entropy_atomic_weighted * R,
        # "atomic_number_entropy_z_weighted": z_entropy_z_weighted * R,
        
        # 原子质量分数熵
        "atomic_mass_entropy_atomic_weighted": mass_entropy_atomic_weighted * R,
        # "atomic_mass_entropy_mass_weighted": mass_entropy_mass_weighted * R,
        
        # 原子体积分数熵
        "atomic_volume_entropy_atomic_weighted": volume_entropy_atomic_weighted * R,
        # "atomic_volume_entropy_volume_weighted": volume_entropy_volume_weighted * R,
        
        # 原子价电子分数熵
        "valence_electron_entropy_atomic_weighted": vec_entropy_atomic_weighted * R,
        # "valence_electron_entropy_vec_weighted": vec_entropy_vec_weighted * R,
        
        # 原子电负性分数熵
        "electronegativity_entropy_atomic_weighted": en_entropy_atomic_weighted * R,
        # "electronegativity_entropy_en_weighted": en_entropy_en_weighted * R,
        
        # 原子能量分数熵 (使用e_1s)
        "energy_entropy_e1s_atomic_weighted": energy_entropy_atomic_weighted * R,
        # "energy_entropy_e1s_energy_weighted": energy_entropy_energy_weighted * R,
        
        # 添加e_1s统计信息用于分析
        # "e1s_mean": np.mean(e_1s_values),
        # "e1s_std": np.std(e_1s_values),
        # "e1s_range": max(e_1s_values) - min(e_1s_values)
    }


def add_configurational_entropy_features(df):
    """
    为数据框 df 添加构型熵特征列
    """
    feature_dicts = df['composition'].apply(calculate_all_configurational_entropies)
    feature_df = pd.DataFrame(feature_dicts.tolist())
    return pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

# df = add_configurational_entropy_features(df)
# df.head()





def _flatten_to_number(x):
    """把常见的非标量转为单个数值（尽量），失败返回 np.nan。"""
    try:
        # 直接数字（包括 numpy 标量）
        if isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool):
            return float(x)
        # 布尔转为 0/1
        if isinstance(x, bool):
            return float(int(x))
        # 序列：取平均（可按需改为其它聚合）
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            arr = np.array(list(x), dtype=float)
            if arr.size == 0:
                return np.nan
            return float(np.nanmean(arr))
        # 字符串尝试转数值
        if isinstance(x, str):
            try:
                return float(x)
            except Exception:
                return np.nan
        # dict 或其它，无法直接转换
        return np.nan
    except Exception:
        return np.nan

def sanitize_numeric_columns(df, exclude_cols=None, fill_strategy='median'):
    """
    尝试把 object 列转换为数值列：
      - 对每个 object 列，对值应用 _flatten_to_number，然后 pd.to_numeric(errors='coerce')
      - 用中位数或均值填充 NaN（默认中位数）
    exclude_cols: 列表，不处理的列（例如 ['formula','composition','normal_formula']）
    """
    if exclude_cols is None:
        exclude_cols = ['formula', 'composition', 'normal_formula', 'temperature']
    obj_cols = [c for c, dt in df.dtypes.items() if dt == 'object' and c not in exclude_cols]
    for c in obj_cols:
        try:
            ser = df[c].map(_flatten_to_number)
            ser = pd.to_numeric(ser, errors='coerce')
            if fill_strategy == 'median':
                fill_val = ser.median()
            elif fill_strategy == 'mean':
                fill_val = ser.mean()
            else:
                fill_val = 0.0
            # 若列全 NaN，则填 0
            if np.isnan(fill_val):
                fill_val = 0.0
            df[c] = ser.fillna(fill_val).astype(float)
        except Exception:
            # 若转换失败，直接删除该列（避免传入模型报错）
            df.drop(columns=[c], inplace=True)
    return df
def add_features_for_df(df):
    df["composition"] = df["formula"].apply(new_formula)
    df['normal_formula'] = df["formula"].apply(ft.normalize_formula)
    df = add_polarization_factors(df)
    df = ft.get_all_features(df, drops = drops)
    df = ft.calc_orbital(df)
    df = add_migration_ion_features(df, migration_ion='Li')
    df = add_ion_statistics(df)
    df = add_ion_coulomb_matrix_features(df)
    df = add_configurational_entropy_features(df)
    df = Stoichiometry().featurize_dataframe(df, 'composition',ignore_errors=True)
    df = AtomicPackingEfficiency(impute_nan=True).featurize_dataframe(df, 'composition',ignore_errors=True)
    df.rename(columns={'dist from 1 clusters |APE| < 0.010': 'APE_close_to_1'}, inplace = True)
    df.rename(columns={'dist from 3 clusters |APE| < 0.010': 'APE_close_to_3'}, inplace = True)
    df.rename(columns={'dist from 5 clusters |APE| < 0.010': 'APE_close_to_5'}, inplace = True)
    return df

def mix_materials(formula1, formula2, ratio1: float = 0.5):
    """
    以比率 ratio1 混合两个化学式，返回标准化的化学式字符串。
    - 用 Composition.get_el_amt_dict 得到元素计量（可能为浮点）
    - 按比例加权后过滤掉接近 0 的元素
    - 用 pymatgen.Composition 重新构造并返回规范化/约简的化学式
    """
    comp1 = new_formula(formula1)
    comp2 = new_formula(formula2)

    el_amt_dict1 = comp1.get_el_amt_dict()
    el_amt_dict2 = comp2.get_el_amt_dict()

    mixed_el_amt = {}
    for el in set(list(el_amt_dict1.keys()) + list(el_amt_dict2.keys())):
        amt1 = el_amt_dict1.get(el, 0.0) * ratio1
        amt2 = el_amt_dict2.get(el, 0.0) * (1.0 - ratio1)
        amt = float(amt1 + amt2)
        # 过滤接近 0 的元素
        if amt <= 1e-8:
            continue
        # 确保键为元素符号字符串
        sym = el if isinstance(el, str) else getattr(el, "symbol", str(el))
        mixed_el_amt[sym] = mixed_el_amt.get(sym, 0.0) + amt

    if not mixed_el_amt:
        return ""

    # 用 pymatgen Composition 规范化/约简化学式
    comp_mixed = Composition(mixed_el_amt)
    # 返回约简后的化学式字符串（例如 Li3InCl6）
    return comp_mixed.reduced_formula

# def add_temperature_feature(formula:str):
#     # temperature = [i for i in range(101)]
#     temperature = [25]
#     df = pd.DataFrame([{'formula':formula}])
#     df es(df)
#      # 在扩展为不同温度行之前，确保尽可能把 object 列变为数值或删除不可转列
#     df = sanitize_numeric_columns(df, exclude_cols=['formula','composition','normal_formula','temperature'])

    # # 使用 pandas repeat 保留列类型（避免 np.repeat 导致 object）
    # df = df.loc[df.index.repeat(len(temperature))].reset_index(drop=True)
    # df['lnT'] = np.log(np.array(temperature)+273)
    # df['1/T'] = 1/(np.array(temperature)+273)
    # df['temperature'] = np.array(temperature)
    # return df

def featurize_mixed_formulas(formula1, formula2, ratio1:float=0.5):
    mixed_formula = mix_materials(formula1, formula2, ratio1)
    df = add_features_for_df(mixed_formula)
    return df

def featurize_mixed_formulas_list(formula1, formula2, ratios:list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
    df = pd.DataFrame([{'formula':mix_materials(formula1,formula2,r)} for r in ratios])
    df = add_features_for_df(df)
    df['mixing_ratio'] = ratios
    df['temperature'] = 25
    df['lnT'] = np.log(25+273)
    df['1/T'] = 1/(25+273)
    return df


def predict_ionic_conductivity(formula):
    model = XGBRegressor()
    model.load_model('./myml/halide_xgb.json')
    df = pd.DataFrame([{'formula':formula}])
    df = add_features_for_df(df)
    df['temperature'] = 25
    df['lnT'] = np.log(25+273)
    df['1/T'] = 1/(25+273)
    features = df.drop(columns=['formula','composition','normal_formula'])
    predictions = model.predict(features[model.feature_names_in_])
    df['predicted_log_conductivity'] = predictions
    print(df[['formula','temperature','predicted_log_conductivity']])
    return df[['formula','temperature','predicted_log_conductivity']].to_dict(orient='records')

def predict_mixed_ionic_conductivity(formula1, formula2, ratios:list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
    model = XGBRegressor()
    model.load_model('./myml/halide_xgb.json')
    df = featurize_mixed_formulas_list(formula1, formula2, ratios)
    df['temperature'] = 25
    df['lnT'] = np.log(25+273)
    df['1/T'] = 1/(25+273)
    features = df.drop(columns=['formula','composition','normal_formula'])
    predictions = model.predict(features[model.feature_names_in_])
    df['predicted_log_conductivity'] = predictions
    print(df[['formula','mixing_ratio','temperature','predicted_log_conductivity']])
    return df[['formula','mixing_ratio','temperature','predicted_log_conductivity']].to_dict(orient='records')

import multiprocessing
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
import joblib
# from sklearn.svm import SVR
# from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    # multiprocessing.freeze_support()
    # formula = "LiCl"
    # df = add_temperature_feature(formula)
    # print(df.dtypes)
    # print(df.head())
    model = XGBRegressor()
    model.load_model('./myml/halide_xgb.json')
    # model = joblib.load('svr_model_v6.pkl')
    # features = df.drop(columns=['formula','composition','normal_formula','temperature'])
    # predictions = model.predict(features[model.feature_names_in_])
    # df['predicted_log_conductivity'] = predictions
    # print(df[['formula','temperature','predicted_log_conductivity']])
    # plt.plot(df['1/T'].apply(lambda x: x*1000), np.array(df['predicted_log_conductivity']), marker='o')
    # plt.xlabel('1000/T (1/K)')
    # plt.ylabel('Predicted log10 Conductivity (S/cm)')
    # plt.title(f'Predicted Ionic Conductivity vs Temperature for {formula}')
    # plt.grid()
    # #线性回归
    # coeffs = np.polyfit(df['1/T'].apply(lambda x: x*1000), np.array(df['predicted_log_conductivity']), 1)
    # poly_eqn = np.poly1d(coeffs)
    # plt.plot(df['1/T'].apply(lambda x: x*1000), poly_eqn(df['1/T'].apply(lambda x: x*1000)), color='red', linestyle='--', label='Linear Fit')
    # plt.legend()
    # plt.show()
    df = featurize_mixed_formulas_list("Li3ScCl6","Li3YCl6")
    features = df.drop(columns=['formula','composition','normal_formula','temperature'])
    predictions = model.predict(features[model.feature_names_in_])
    df['predicted_log_conductivity'] = predictions
    print(df[['formula','temperature','predicted_log_conductivity']])
    plt.figure(figsize=(10,6))
    plt.plot(df['mixing_ratio'], np.array(df['predicted_log_conductivity']), marker='o')
    plt.xlabel('Mixing Ratio')
    plt.ylabel('Predicted log10 Conductivity (S/cm)')
    plt.title('Predicted Ionic Conductivity vs Mixing Ratio at 25°C')
    plt.grid()
    plt.show()
