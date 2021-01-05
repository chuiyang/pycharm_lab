import pandas as pd
import numpy as np
import torch


def f_bonds_change(f_bonds, b_scope_0):  # 0
    _, b_num = b_scope_0  # b_num可以代表 n*147中的n
    dx = 0.00001
    k = int(f_bonds.shape[0] / 2)
    # print('f_bonds.shape[0]', f_bonds.shape[0])
    # print('k', k)
    # print('int(b_num)*147:', int(b_num)*147)

    for i in range(0, int(b_num)*147):  # int(b_num)*147
        t = int(b_num*i)
        m = i//147+1
        n = i % 147
        f_bonds[t+m, n] = f_bonds[t+m, n] + dx
    for i in range(0, int(b_num)*147):
        t = int(b_num*i)
        m = i//147+1
        n = i % 147
        f_bonds[m+k+t, n] = f_bonds[m+k+t, n] - dx
    # print(f_bonds[-12:, -5:])
    return f_bonds
