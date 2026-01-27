import re
from collections import defaultdict
from fractions import Fraction
from math import gcd
from functools import reduce
import pandas as pd
import numpy as np
import xgboost
# 读取元素特征数据
element_features = pd.read_csv('./myml/element_features_bandgap.csv')
element_features = element_features.set_index('element')


def normalize_formula(formula: str) -> dict:
    """
    支持括号和小数系数的化学式解析，如 Ag(W3Br7)2 或 Ag0.5Ge1Pb1.75S4
    返回元素:计数的字典
    """
    def multiply_dict(d, factor):
        return {k: v * factor for k, v in d.items()}

    def merge_dict(d1, d2):
        for k, v in d2.items():
            d1[k] = d1.get(k, 0) + v
        return d1

    def parse(s):
        tokens = re.findall(r'([A-Z][a-z]?|\(|\)|\d*\.?\d+)', s)
        stack = []
        elem_dict = defaultdict(float)
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
                    factor = float(tokens[i])
                    i += 1
                else:
                    factor = 1.0
                sub_dict = multiply_dict(sub_dict, factor)
                elem_dict = merge_dict(elem_dict, sub_dict)
            elif re.match(r'[A-Z][a-z]?', token):
                elem = token
                i += 1
                if i < len(tokens) and re.match(r'\d*\.?\d+', tokens[i]):
                    count = float(tokens[i])
                    i += 1
                else:
                    count = 1.0
                elem_dict[elem] += count
            else:
                i += 1
        return elem_dict

    # 去除空格
    formula = formula.replace(' ', '')
    elements = parse(formula)

    # 找所有分母，转为整数
    denominators = []
    for coeff in elements.values():
        if float(coeff).is_integer():
            denominators.append(1)
        else:
            frac = Fraction(coeff).limit_denominator(100000)
            denominators.append(frac.denominator)
    lcm_val = reduce(lambda a, b: a * b // gcd(a, b), set(denominators))
    int_formula = {}
    for elem, coeff in elements.items():
        int_coeff = round(coeff * lcm_val)
        if abs(int_coeff - (coeff * lcm_val)) > 1e-6:
            raise ValueError(f"系数转换不精确: {coeff} * {lcm_val} = {coeff*lcm_val}")
        int_formula[elem] = int_coeff
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

def get_avg_feature(normalize_formula: dict, feature_name: str) -> float:
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

def get_feature(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    获取DataFrame中每个化学式的特征值
    """
    new_df = df.copy()
    new_df[f'max_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_max_feature(x, feature_name))
    new_df[f'min_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_min_feature(x, feature_name))
    new_df[f'avg_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_avg_feature(x, feature_name))
    new_df[f'range_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_range_feature(x, feature_name))
    new_df[f'std_{feature_name}'] = new_df['normalized_formula'].apply(lambda x: get_std_feature(x, feature_name))
    return new_df

def get_all_features(df: pd.DataFrame) -> pd.DataFrame:
    drops = ['normalized_formula', "is_s","is_p","is_d","is_ds","is_f",'Row_1',
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
    'Column_18',]
    for feature in element_features.columns:
        if feature in drops:
            continue
        # 仅处理数值类型的特征
        # print(f"Processing feature: {feature}")
        df = get_feature(df, feature)
    return df


def predict_bandgap(formula: str) -> float:
    """
    使用预训练的模型预测给定化学式的带隙值
    """
    model = xgboost.XGBRegressor()
    model.load_model('./myml/xgb_model.json')
    normalized_formula = normalize_formula(formula)
    features = get_all_features(pd.DataFrame({'normalized_formula': [normalized_formula]}))
    feature_vector = features[model.feature_names_in_]
    prediction = model.predict(feature_vector)
    return prediction[0]