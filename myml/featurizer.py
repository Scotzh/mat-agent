import re
from collections import defaultdict
from fractions import Fraction
from math import gcd
from functools import reduce
import pandas as pd
import numpy as np
# 读取元素特征数据
element_features = pd.read_csv('./myml/element_features.csv')
element_features = element_features.set_index('element')


def get_element_features_columns() -> list:
    print("获取元素特征列名")
    print("元素特征列名:", element_features.columns.tolist())
    """
    获取元素特征的列名
    """
    return element_features.columns.tolist()


def normalize_formula(formula: str) -> dict:
    """
    支持括号和小数系数的化学式解析，改进点：
    - 使用 Fraction 以保留精确分数；
    - 识别像 0.33/0.333 这类单一数字重复的小数为循环小数（如 1/3）；
    返回元素:计数的字典（整数计数）。
    """
    def multiply_dict(d, factor):
        return {k: v * factor for k, v in d.items()}

    def merge_dict(d1, d2):
        for k, v in d2.items():
            d1[k] = d1.get(k, 0) + v
        return d1

    def smart_fraction_from_str(s: str) -> Fraction:
        # 仅对几个常见近似小数做特例识别（要求简单、稳定）
        s = s.strip()
        # 处理带符号或整数部分
        if '.' not in s:
            return Fraction(int(s), 1)
        whole, dec = s.split('.')
        # 特例：把 0.33 / 0.333 映射为 1/3；0.167 / 0.1667 映射为 1/6
        try:
            whole_int = int(whole)
        except Exception:
            whole_int = 0
        if dec in ('33', '333') and whole_int == 0:
            return Fraction(1, 3)
        if dec in ('33', '333') and whole_int != 0:
            return Fraction(whole_int, 1) + Fraction(1, 3)
        if dec in ('67', '667') and whole_int == 0:
            return Fraction(2, 3)
        if dec in ('67', '667') and whole_int != 0:
            return Fraction(whole_int, 1) + Fraction(2, 3)
        if dec in ('167', '1667') and whole_int == 0:
            return Fraction(1, 6)
        if dec in ('167', '1667') and whole_int != 0:
            return Fraction(whole_int, 1) + Fraction(1, 6)
        if dec in ('83', '833') and whole_int == 0:
            return Fraction(5, 6)
        if dec in ('83', '833') and whole_int != 0:
            return Fraction(whole_int, 1) + Fraction(5, 6)
        return Fraction(s).limit_denominator(100000)

    def parse(s):
        tokens = re.findall(r'([A-Z][a-z]?|\(|\)|\d*\.?\d+)', s)
        elem_dict = defaultdict(Fraction)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == '(':
                # 找到括号内容
                j = i + 1
                depth = 1
                sub_tokens = []
                while j < len(tokens):
                    if tokens[j] == '(':
                        depth += 1
                    elif tokens[j] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    sub_tokens.append(tokens[j])
                    j += 1
                sub_formula = ''.join(sub_tokens)
                sub_dict = parse(sub_formula)
                i = j + 1
                # 检查括号后是否有数字
                if i < len(tokens) and re.match(r'\d*\.?\d+', tokens[i]):
                    factor = smart_fraction_from_str(tokens[i])
                    i += 1
                else:
                    factor = Fraction(1, 1)
                sub_dict = multiply_dict(sub_dict, factor)
                elem_dict = merge_dict(elem_dict, sub_dict)
            elif re.match(r'[A-Z][a-z]?', token):
                elem = token
                i += 1
                if i < len(tokens) and re.match(r'\d*\.?\d+', tokens[i]):
                    count = smart_fraction_from_str(tokens[i])
                    i += 1
                else:
                    count = Fraction(1, 1)
                elem_dict[elem] += count
            else:
                i += 1
        return elem_dict

    # 去除空格
    formula = str(formula).replace(' ', '')
    elements = parse(formula)

    # 找所有分母，转为整数 lcm
    denominators = [coeff.denominator for coeff in elements.values()]
    # 若没有小数， denominators 可能全部为1
    lcm_val = reduce(lambda a, b: a * b // gcd(a, b), set(denominators))
    int_formula = {}
    for elem, coeff in elements.items():
        product = coeff * lcm_val
        if product.denominator != 1:
            raise ValueError(f"系数转换不精确: {coeff} * {lcm_val} = {float(product)} (非整数)")
        int_formula[elem] = int(product.numerator)
    return int_formula


def get_max_feature(normalize_formula: dict, feature_name: str) -> float:
    """
    获取化学式中元素的最大特征值
    """
    max_value = -np.inf
    for elem, coeff in normalize_formula.items():
        if elem in element_features.index:
            value = element_features.at[elem, feature_name]
            if value > max_value:
                max_value = value
    return max_value if max_value != -np.inf else None

def get_min_feature(normalize_formula: dict, feature_name: str) -> float:
    """
    获取化学式中元素的最小特征值
    """
    min_value = np.inf
    for elem, coeff in normalize_formula.items():
        if elem in element_features.index:
            value = element_features.at[elem, feature_name]
            if value < min_value:
                min_value = value
    return min_value if min_value != np.inf else None

def get_mean_feature(normalize_formula: dict, feature_name: str) -> float:
    """
    获取化学式中元素的平均特征值
    """
    total_value = 0.0
    count = 0
    for elem, coeff in normalize_formula.items():
        if elem in element_features.index:
            value = element_features.at[elem, feature_name]
            total_value += value * coeff
            count += coeff
    
    return total_value / count if count > 0 else None

def get_range_feature(normalize_formula: dict, feature_name: str) -> float:
    """
    获取化学式中元素的特征值范围
    """
    max_value = get_max_feature(normalize_formula, feature_name)
    min_value = get_min_feature(normalize_formula, feature_name)
    
    if max_value is None or min_value is None:
        return None
    
    return max_value - min_value

def get_std_feature(normalize_formula: dict, feature_name: str) -> float:
    """
    获取化学式中元素的特征值标准差
    """
    values = []
    for elem, coeff in normalize_formula.items():
        if elem in element_features.index:
            value = element_features.at[elem, feature_name]
            values.append(value * coeff)
    
    if not values:
        return None
    
    mean_value = np.mean(values)
    variance = np.mean([(x - mean_value) ** 2 for x in values])
    return np.sqrt(variance)

def get_skew_feature(normalize_formula: dict, feature_name: str) -> float:
    """
    获取化学式中元素的特征值偏度
    """
    values = []
    for elem, coeff in normalize_formula.items():
        if elem in element_features.index:
            value = element_features.at[elem, feature_name]
            values.append(value * coeff)
    
    if not values:
        return None
    
    mean_value = np.mean(values)
    std_dev = np.std(values)
    if std_dev == 0:
        return 0.0
    
    skewness = np.mean([(x - mean_value) ** 3 for x in values]) / (std_dev ** 3)
    return skewness


def get_feature(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    获取DataFrame中每个化学式的特征值
    """
    new_df = df.copy()
    new_df[f'max_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_max_feature(x, feature_name))
    new_df[f'min_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_min_feature(x, feature_name))
    new_df[f'mean_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_mean_feature(x, feature_name))
    new_df[f'range_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_range_feature(x, feature_name))
    new_df[f'std_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_std_feature(x, feature_name))
    new_df[f'skew_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_skew_feature(x, feature_name))
    return new_df

def get_all_features(df: pd.DataFrame, drops: str = ['element', 'FirstIonizationEnergy/affinity', 'is_s', 'is_p', 'is_d', 'is_ds',
       'is_f', 'Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5', 'Row_6', 'Row_7',
       'Column_1', 'Column_2', 'Column_3', 'Column_4', 'Column_5', 'Column_6',
       'Column_7', 'Column_8', 'Column_9', 'Column_10', 'Column_11',
       'Column_12', 'Column_13', 'Column_14', 'Column_15', 'Column_16',
       'Column_17', 'Column_18']) -> pd.DataFrame:
    newdf = df.copy()
    newdf['normalized_formula'] = newdf['formula'].apply(normalize_formula)
    for feature in element_features.columns:
        if feature in drops:
            continue
        # 仅处理数值类型的特征
        # print(f"Processing feature: {feature}")
        newdf = get_feature(newdf, feature)
    # 删除不需要的列
    newdf.drop(columns=['normalized_formula'], inplace=True, errors='ignore')
    return newdf


def calc_block_fractions(normalized_formula):
    total = sum(normalized_formula.values())
    print(f"Total elements in formula: {total}")
    s = p = d = ds = f = 0
    for elem, count in normalized_formula.items():
        if elem in element_features.index:
            if element_features.at[elem, 'is_s']:
                s += count
            if element_features.at[elem, 'is_p']:
                p += count
            if element_features.at[elem, 'is_d']:
                d += count
            if element_features.at[elem, 'is_ds']:
                ds += count
            if element_features.at[elem, 'is_f']:
                f += count
    if total == 0:
        return pd.Series([0,0,0,0,0], index=['frac_s','frac_p','frac_d','frac_ds','frac_f'])
    return pd.Series([s/total, p/total, d/total, ds/total, f/total], index=['frac_s','frac_p','frac_d','frac_ds','frac_f'])

def calc_block_fractions_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个化学式中 s, p, d, ds, f 区块的比例
    """
    df[['frac_s', 'frac_p', 'frac_d', 'frac_ds', 'frac_f']] = df['normalized_formula'].apply(calc_block_fractions)
    return df.drop(columns=['normalized_formula'], errors='ignore')



def calc_column_fractions_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个化学式中各族元素的比例
    """
    columns = [f'Column_{i}' for i in range(1, 19)]
    def calc_column_fractions(normalized_formula):
        print(f"Calculating column fractions for formula: {normalized_formula}")
        total = sum(normalized_formula.values())
        col_counts = dict.fromkeys(columns, 0)
        for elem, count in normalized_formula.items():
            if elem in element_features.index:
                for col in columns:
                    if col in element_features.columns and element_features.at[elem, col]:
                        col_counts[col] += count
                        break  # 每个元素只属于一个族
        if total == 0:
            return pd.Series([0]*18, index=[f'frac_{col}' for col in columns])
        return pd.Series([col_counts[col]/total for col in columns], index=[f'frac_{col}' for col in columns])
    df['normalized_formula'] = df['formula'].apply(normalize_formula)
    df[[f'frac_{col}' for col in columns]] = df['normalized_formula'].apply(calc_column_fractions)
    return df.drop(columns=['normalized_formula'], errors='ignore')



def calc_orbital(df: pd.DataFrame) -> pd.DataFrame:
    newdf = df.copy()
    newdf['normalized_formula'] = newdf['formula'].apply(normalize_formula)
    from myml import atomic_orbital_calc
    newdf['obidata'] = newdf['normalized_formula'].apply(lambda x: atomic_orbital_calc.ImprovedMolecularOrbitals(x).get_data()) 
    newdf['HOMO'] = newdf['obidata'].apply(lambda x: x['HOMO'])
    newdf['LUMO'] = newdf['obidata'].apply(lambda x: x['LUMO'])
    newdf['gap'] = newdf['obidata'].apply(lambda x: x['gap'])
    return newdf.drop(columns=['normalized_formula', 'obidata'], errors='ignore')


def my_featurizer(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    from pymatgen.core.composition import Composition
    from matminer.featurizers.composition.element import Stoichiometry
    from matminer.featurizers.composition.packing import AtomicPackingEfficiency
    import featurizer as ft
    def new_formula(formula:str)-> Composition: 
        # 把原来的化学式转为标准化学式
        nm_formula = ft.normalize_formula(formula)
        # print(f"原始化学式：{formula}，标准化：{nm_formula}",end='，')
        nm_formula = str(''.join([f"{k}{v if v>1 else ''}" for k,v in nm_formula.items()]))
        print(nm_formula)
        composition = Composition(nm_formula)
        return composition
    drops = ['Etot',
    'Etot/atomic_number^2',
    'Ekin',
    'Ekin/atomic_number^2',
    'Ecoul',
    'Ecoul/atomic_number^2',
    'Eenuc',
    'Eenuc/atomic_number^2',
    'Exc',
    'Exc/atomic_number^2','EATOM/ZVAL','is_s',
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
    'Row',]
    new_df["composition"] = new_df["formula"].apply(new_formula)
    new_df = Stoichiometry().featurize_dataframe(new_df, 'composition',ignore_errors=True)
    new_df = AtomicPackingEfficiency().featurize_dataframe(new_df, 'composition',ignore_errors=True)
    new_df = calc_orbital(new_df)
    new_df = get_all_features(new_df, drops=drops)
    new_df.rename(columns={'dist from 1 clusters |APE| < 0.010': 'APE_close_to_1'}, inplace = True)
    new_df.rename(columns={'dist from 3 clusters |APE| < 0.010': 'APE_close_to_3'}, inplace = True)
    new_df.rename(columns={'dist from 5 clusters |APE| < 0.010': 'APE_close_to_5'}, inplace = True)
    return new_df.drop(columns=['composition'], errors='ignore')
