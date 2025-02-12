import os
import time
import logging
from functools import wraps
from collections import defaultdict
import h5py
import numpy as np
from dscribe.descriptors import SOAP

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
    
def compute_soap_descriptors(structures, njobs, species, r_cut, n_max, l_max, logger):
    """
    Function: Compute SOAP descriptors for a list of structures
    Input:
        SOAP inputs
        logger: logger object
    Output:
        List of SOAP descriptors in numpy.ndarray format
    """
    start_time = time.time()
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max
    )

    try:
        soap_descriptors = soap.create(structures, n_jobs=njobs)
    except IndexError:
        # 说明 structures 为空
        soap_descriptors = []

    end_time = time.time()
    logger.info(f"SOAP descriptors computed in {end_time - start_time:.2f} seconds")

    # 由于此处调用本函数计算时，structures 中的结构默认是化学式相同的，因此返回的是 np.ndarray
    # 需要返回 list 类型，其中元素为 np.ndarray
    if type(soap_descriptors) == np.ndarray:
        return [i for i in soap_descriptors]
    
    # 原子数不相同时直接返回 list 即可
    return soap_descriptors
    
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
    
def defaultdict_profiler(soap_data):
    """
    Print the available formulas and the shape of their corresponding SOAP descriptors.
    """
    formulas = list(soap_data.keys())
    print("Available formulas:", formulas)

    # 遍历每个分子式并读取 SOAP 描述符
    for formula in formulas:
        soap_descriptors = soap_data[formula][:][0]
        print(f"Formula: {formula}, Shape of SOAP descriptors: {soap_descriptors.shape}")
    print("---------")

# 设置总的日志记录
def setup_total_logging():
    '''
    Setup the total logging for the algorithm.
    '''
    # 创建总的 Logger
    total_logger = logging.getLogger("total_logger")
    total_logger.setLevel(logging.INFO)

    # 清除之前的处理器，确保每次都干净
    for handler in total_logger.handlers[:]:
        total_logger.removeHandler(handler)

    # 创建文件处理器
    file_handler = logging.FileHandler("total_output.log", mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    total_logger.addHandler(file_handler)

    return total_logger
    
# 设置各化学式的日志记录
def setup_logging(reaction_formula):
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
    os.makedirs(reaction_formula, exist_ok=True)

    # 创建文件处理器
    file_handler = logging.FileHandler(os.path.join(reaction_formula, f"{reaction_formula}_output.log"), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 添加处理器到 Logger
    logger.addHandler(file_handler)
    
    return logger