import ctypes
import subprocess
import time


#
# for i in range(20):
    ##########################################################################################################
    # # 加载DLL  持续时间9.123719930648804秒
    # start = time.time()
    # my_library = ctypes.CDLL("./123/connect5-ai.dll")
    #
    # # 指定函数原型，告诉ctypes该函数的返回类型和参数类型
    # get_action_once = my_library.get_action_once
    # # get_action_once.restype = ctypes.POINTER(ctypes.c_int)
    # get_action_once.restype = None
    # get_action_once.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    #
    # # 示例调用
    # search_time = 5
    # return_nums = 2
    # nums = 4
    # history = [2, 2, 1, 4]
    # history_array = (ctypes.c_int * nums)(*history)
    # # res = get_action_once(5, return_nums, nums, history_array)
    # get_action_once(5, return_nums, nums, history_array)
    # long_time = time.time() - start
    #
    # # print(res)
    # print(f"dll持续时间{long_time}秒")
    #


    # ##########################################################################################################
    # 加载exe  # 13.328578233718872秒   但是c++指定的时间是5s
    # start = time.time()
    # arguments = ['5', '2', '2', '1', '4']
    # ret = subprocess.Popen(
    #     r"Release/connect5-ai.exe",
    #     stdout=subprocess.PIPE,
    #     stdin=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    # )
    #
    # for tmp in arguments:
    #     ret.stdin.write(bytes(tmp + "\n",'utf-8'))
    #     ret.stdin.flush()
    # result = []
    # output = ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
    # print(output)
    # output = ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
    # print(output)
    # output = ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
    # print(output)
    # for i in range(2):
    #     output = ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
    #     result.append(int(output))
    # long_time = time.time() - start
    # print(result)
    # print(f"exe所用时间:{long_time}秒")


