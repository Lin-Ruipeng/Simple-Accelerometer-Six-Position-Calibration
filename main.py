import os

import numpy as np
import pandas as pd


# 求取指定轴均值：因为采样率100Hz且只取30s中间的10s,所以开始行是1000,结束行是2000
def mean_with_pandas(csv_file, cols=[1, 2, 3], start_row=1000, end_row=2000):
    df = pd.read_csv(csv_file)
    df_subset = df.iloc[start_row - 1 : end_row, cols]
    return df_subset.mean(axis=0).tolist()


# 原始数据被存在data/下,并且名字是符号+轴+.csv的固定格式
def main():
    data_path = "data/"
    df_list = ["+X.csv", "+Y.csv", "+Z.csv", "-X.csv", "-Y.csv", "-Z.csv"]
    # 求取均值, 其中cols = [1, 2, 3] 分别对应X/Y/Z轴的数据
    posX_list = mean_with_pandas(os.path.join(data_path, df_list[0]))
    posY_list = mean_with_pandas(os.path.join(data_path, df_list[1]))
    posZ_list = mean_with_pandas(os.path.join(data_path, df_list[2]))
    negX_list = mean_with_pandas(os.path.join(data_path, df_list[3]))
    negY_list = mean_with_pandas(os.path.join(data_path, df_list[4]))
    negZ_list = mean_with_pandas(os.path.join(data_path, df_list[5]))

    print(f" posX: {posX_list}, \n posY: {posY_list}, \n posZ: {posZ_list}")
    print(f" negX: {negX_list}, \n negY: {negY_list}, \n negZ: {negZ_list}")

    # 取出重力轴
    posX, posY, posZ = posX_list[0], posY_list[1], posZ_list[2]
    negX, negY, negZ = negX_list[0], negY_list[1], negZ_list[2]

    # 计算重力参考值
    g = (posX + posY + posZ - negX - negY - negZ) / 6
    print(f"g: {g}")

    # 构建A+矩阵
    # A_pos = [posX_list, posY_list, posZ_list]
    A_pos = [[posX, 0, 0], [0, posY, 0], [0, 0, posZ]]
    # 构建A-矩阵
    # A_neg = [negX_list, negY_list, negZ_list]
    A_neg = [[negX, 0, 0], [0, negY, 0], [0, 0, negZ]]

    # 计算校准矩阵 Cs = 2 * g * (A+ - A-)^-1
    Cs = 2 * g * np.linalg.inv(np.array(A_pos) - np.array(A_neg))
    print(f"Cs: {Cs}")

    # 计算零偏矩阵 A0 = (A+ + A-) / 2
    A0 = (np.array(A_pos) + np.array(A_neg)) / 2
    print(f"A0: {A0}")

    # 对原始数据进行校正 q = Cs @ (q - q0)
    q0 = np.array([A0[0][0], A0[1][1], A0[2][2]])
    qPosX_list = Cs @ (posX_list - q0)
    qPosY_list = Cs @ (posY_list - q0)
    qPosZ_list = Cs @ (posZ_list - q0)
    qNegX_list = Cs @ (negX_list - q0)
    qNegY_list = Cs @ (negY_list - q0)
    qNegZ_list = Cs @ (negZ_list - q0)

    print(f" qPosX: {qPosX_list}, \n qPosY: {qPosY_list}, \n qPosZ: {qPosZ_list}")
    print(f" qNegX: {qNegX_list}, \n qNegY: {qNegY_list}, \n qNegZ: {qNegZ_list}")


if __name__ == "__main__":
    main()
