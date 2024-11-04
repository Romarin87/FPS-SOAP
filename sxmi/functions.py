import time
from functools import wraps

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