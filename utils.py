import os
import time
import logging
from functools import wraps
from collections import defaultdict
import h5py
import numpy as np

def time_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用被装饰的函数
        end_time = time.time()  # 记录结束时间
        duration = end_time - start_time  # 计算持续时间
        print(f"Function '{func.__name__}' executed in {duration:.2f} seconds.")
        return result  # 返回函数的结果
    return wrapper
    
def save_soap_to_hdf5(soap_dict, hdf5_name):
    """
    Save SOAP descriptors to an HDF5 file.

    Example input structure:
    defaultdict(list, {
        'C2H3N3O': [
            array([[...], [...], ...]),  # SOAP descriptors for the first molecule
            array([[...], [...], ...])   # SOAP descriptors for the second molecule
        ],
        'C2H6': [
            array([[...], [...], ...])   # SOAP descriptors for another molecule
        ]
    })

    - Each value is an array representing the SOAP descriptors for a molecule, with shape (N, M),
    where N is the number of descriptors and M is the dimension of each descriptor.
    """
    with h5py.File(hdf5_name, "w") as hdf:
        for formula, soap_list in soap_dict.items():
            stacked_soap = np.array(soap_list)
            hdf.create_dataset(formula, data=stacked_soap)
            
def read_soap_from_hdf5(hdf5_name):
    """
    Read SOAP descriptors from an HDF5 file.

    Returns:
        defaultdict(list): A dictionary with molecule formulas as keys
        and lists of corresponding SOAP descriptors as values.
    """
    soap_dict = defaultdict(list)

    with h5py.File(hdf5_name, "r") as hdf:
        for formula in hdf.keys():
            soap_descriptors = hdf[formula][:]
            soap_dict[formula].append(soap_descriptors)

    return soap_dict

# 设置总的日志记录
def setup_total_logging(path):
    '''
    Setup the total logging for the algorithm.
    '''
    # 创建总的 Logger
    total_logger = logging.getLogger("total_logger")
    total_logger.setLevel(logging.INFO)

    # 清除之前的处理器，确保每次都干净
    for handler in total_logger.handlers[:]:
        total_logger.removeHandler(handler)
    
    os.makedirs(path, exist_ok=True)

    # 创建文件处理器
    file_handler = logging.FileHandler(os.path.join(path, "total_output.log"), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    total_logger.addHandler(file_handler)

    return total_logger
    
# 设置各化学式的日志记录
def setup_logging(reaction_formula, path):
    '''
    Setup the logging for each chemical formula.
    '''
    # 创建新的 Logger
    logger = logging.getLogger(reaction_formula)
    logger.setLevel(logging.INFO)
        
    # 清除之前的处理器，确保每次都干净
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建文件夹
    formula_path = os.path.join(path, reaction_formula)
    os.makedirs(formula_path, exist_ok=True)

    # 创建文件处理器
    file_handler = logging.FileHandler(os.path.join(formula_path, f"{reaction_formula}_output.log"), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 添加处理器到 Logger
    logger.addHandler(file_handler)
    
    return logger