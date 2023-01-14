import numpy as np
import math
from numpy.linalg import matrix_rank


class MatrixFactorization():
    # 输入：A(矩阵)，fac_type(分解的类型)
    # fac_type：
    # ‘LU‘对应LU分解
    # ‘GS‘对应LU分解
    # ‘HR‘对应LU分解
    # ‘GR‘对应LU分解
    # ‘URV‘对应URV分解
    # 输出：分解完的各个矩阵
    def fac(self, A, fac_type):
        A = A.astype(float)
        m, n = A.shape[0], A.shape[1]
        if fac_type == 'LU':
            return self.LU(A, m, n)
        elif fac_type == 'GS':
            return self.QR(A, m, n)
        elif fac_type == 'HR':
            return self.householder_reduction(A, m, n)
        elif fac_type == 'GR':
            return self.givens_reduction(A, m, n)
        elif fac_type == 'URV':
            return self.URV(A, m, n)

    # 输入：A(矩阵)，b(值向量)，type(使用哪种矩阵分解)
    # 输出：方程的解
    def solve(self, A, b, type):
        A = A.astype(float)
        m, n = A.shape[0], A.shape[1]

        # 根据不同的矩阵分解方式求y
        y = np.zeros(m)
        U = None
        if type == 'LU':
            L, U = self.LU(A, m, n)
            for i in range(m):
                temp = 0
                for j in range(i):
                    temp += y[j] * L[i][j]
                y[i] = (b[i] - temp) / L[i][i]
        elif type == 'GS':
            Q, U = self.QR(A, m, n)
            y = np.matmul(Q.T, b)
        elif type == 'HR':
            Q, U = self.householder_reduction(A, m, n)
            y = np.matmul(Q.T, b)
        elif type == 'GR':
            Q, U = self.givens_reduction(A, m, n)
            y = np.matmul(Q.T, b)

        x = np.zeros(n)
        for i in (n - np.arange(1, n + 1)):
            temp = 0
            for j in range(i + 1, n):
                temp += x[j] * U[i][j]
            if U[i][i] == 0:
                x[i] = 1
            else:
                x[i] = (y[i] - temp) / U[i][i]
        return (x * 1e3).round() / 1e3

    # 输入：A(矩阵)，b(值向量)，type(使用哪种矩阵分解)
    # 输出：矩阵行列式值
    def determinant(self, A, type):
        A = A.astype(float)
        m, n = A.shape[0], A.shape[1]
        if m != n:
            print('该矩阵无法计算行列式')
            return None
        if type == 'LU':
            _, U = self.LU(A, m, n)
        elif type == 'GS':
            Q, U = self.QR(A, m, n)
        elif type == 'HR':
            Q, U = self.householder_reduction(A, m, n)
        else:
            Q, U = self.givens_reduction(A, m, n)
        res = 1
        for i in range(m):
            res *= U[i][i]
        # 保留三位小数
        return int(res * 1e3) / 1e3

    # 输入：A(矩阵)，m(行数), n(列数)
    # 输出：分解完的L、U
    def LU(self, A, m, n):
        A = A.astype(float)
        L = np.eye(m)
        U = A.copy()
        if n > m:
            print("矩阵不能进行LU分解")
            return None, None
        for i in range(n):
            if U[i][i] == 0:
                k = min(i + 1, m - 1)
                while (k < m and U[i][k] != 0):
                    k += 1
                if (k == m - 1):
                    continue
                print(i, k, m)
                print("矩阵不能进行LU分解")
                return None, None
            for j in range(i + 1, m):
                L[i] += L[j] / U[i][i] * U[j][i]
                U[j] -= U[i] / U[i][i] * U[j][i]
        return (L.T * 1e5).round() / 1e5, (U * 1e5).round() / 1e5

    # 输入：A(矩阵)，m(行数), n(列数)
    # 输出：分解完的Q、R
    def QR(self, A, m, n):
        A = A.astype(float)
        Q, R = np.zeros((m, m)), np.zeros((m, n))
        for i in range(m):
            if i == 0:
                R[0, 0] = np.sqrt(np.power(A[:, 0], 2).sum())
                Q[:, 0] = A[:, 0] / R[0, 0]
                continue
            temp = np.zeros(m)
            for j in range(i):
                R[j, i] = (Q[:, j] * A[:, i]).sum()
                temp += R[j, i] * Q[:, j]
            Q[:, i] = A[:, i] - temp
            R[i, i] = np.sqrt(np.power(Q[:, i], 2).sum())
            if R[i, i] == 0:
                continue
            Q[:, i] /= R[i, i]
        return (Q * 1e5).round() / 1e5, (R * 1e5).round() / 1e5

    # 输入：A(矩阵)，m(行数), n(列数)
    # 输出：分解完的Q、R
    def householder_reduction(self, A, m, n):
        A = A.astype(float)
        # 化简次数
        times = min(m, n)
        R = A.copy()
        Q = np.eye(m)
        for i in range(times - 1):
            ei = np.zeros(m)
            ei[i] = 1
            # 一定要写cpoy，不然R中的值也会跟着改
            temp = R[:, i].copy()
            temp[:i] = 0
            # print(temp, ei, times, A.shape)
            ui = temp - np.sqrt(np.power(temp, 2).sum()) * ei
            if np.count_nonzero(ui) == 0:
                continue
            ui = ui.reshape(m, -1)
            Ri = np.eye(m) - 2 * (ui * ui.T) / ((ui * ui).sum())
            R = np.matmul(Ri, R)
            Q = np.matmul(Q, Ri)
        # 结果保留五位小数
        return (Q * 1e5).round() / 1e5, (R * 1e5).round() / 1e5

    # 输入：A(矩阵)，m(行数), n(列数)
    # 输出：分解完的Q、R
    def givens_reduction(self, A, m, n):
        A = A.astype(float)
        # 化简次数
        times = min(m, n)
        Q = np.eye(m)
        R = A.copy()
        for i in range(times):
            for j in range(i + 1, times):
                temp = math.sqrt(R[i][i] * R[i][i] + R[j][i] * R[j][i])
                cos = R[i][i] / temp
                sin = R[j][i] / temp
                P = np.eye(m)
                P[i][i] = cos
                P[j][j] = cos
                P[i][j] = sin
                P[j][i] = -sin
                R = np.matmul(P, R)
                P_inv = P
                P_inv[i][j] = -sin
                P_inv[j][i] = sin
                Q = np.matmul(Q, P_inv)
        # 结果保留五位小数
        return (Q * 1e5).round() / 1e5, (R * 1e5).round() / 1e5

    # 输入：A(矩阵)，m(行数), n(列数)
    # 输出：分解完的U、R、V矩阵
    def URV(self, A, m, n):
        A = A.astype(float)
        rank = matrix_rank(A)
        Q1, R1 = self.householder_reduction(A, m, n)
        T = (R1[: rank]).T
        U = Q1
        Q2, R2 = self.householder_reduction(T, n, rank)
        # print(T, cut(np.matmul(Q2, R2)))
        V = Q2.T
        R = np.zeros((m, n))
        R[:rank, :rank] = (R2[:rank]).T
        # print(np.matmul(U, np.matmul(R, V)))
        # 结果保留五位小数
        return (U * 1e5).round() / 1e5, (R * 1e5).round() / 1e5, (V * 1e5).round() / 1e5


###############################################################################
# 1.程序说明：定义了一个MatrixFactorization类，内部实现关于矩阵分解的LU、QR(Gram-Schmidt)
# Orthogonal Reduction(Householder reduction和Givens reduction)、URV
# 类中的fac函数是用于进行矩阵分解的统一接口，solve函数是用于进行方差求解的统一接口，determinant为求矩阵行列式的接口
mf = MatrixFactorization()


###############################################################################
# 2.进行矩阵分解的例子
print("################矩阵分解#################")
# 2.1矩阵的LU分解
A = np.array([[1, 3, 4],
              [1, 4, 6],
              [2, 6, 10]])
L, U = mf.fac(A, 'LU')
print(f'LU分解\n'
      f'A矩阵:\n{A}\n'
      f'L矩阵:\n{L}\n'
      f'U矩阵:\n{U}\n')


A = np.array([[0, -20, -14],
              [3, 27, -4],
              [4, 11, -2]])
# 2.2使用Gram-Schmidt进行Q、R分解
Q, R = mf.fac(A, 'GS')
print(f'Gram-Schmidt进行Q、R分解\n'
      f'A矩阵:\n{A}\n'
      f'Q矩阵:\n{Q}\n'
      f'R矩阵:\n{R}\n')
# 2.3使用Householder reduction进行Q、R分解
Q, R = mf.fac(A, 'HR')
print(f'Householder reduction进行Q、R分解\n'
      f'A矩阵:\n{A}\n'
      f'Q矩阵:\n{Q}\n'
      f'R矩阵:\n{R}\n')

# 2.4使用Givens reduction进行Q、R分解
Q, R = mf.fac(A, 'GR')
print(f'Givens reduction进行Q、R分解\n'
      f'A矩阵:\n{A}\n'
      f'Q矩阵:\n{Q}\n'
      f'R矩阵:\n{R}\n')

# 2.4矩阵的URV分解
A = np.array([[1, 3, 4],
              [1, 4, 6],
              [2, 6, 10]])
U, R, V = mf.fac(A, 'URV')
print(f'LU分解\n'
      f'A矩阵:\n{A}\n'
      f'U矩阵:\n{U}\n'
      f'R矩阵:\n{R}\n'
      f'U矩阵:\n{V}\n')

###############################################################################
print("#############解方程###############")
# 3.在矩阵分解基础上解方程
A = np.array([[4, 2, 4],
              [1, 4, 2],
              [1, 2, 2]])
b = np.array([11, 12, 9])
# type可以换为
x_LU = mf.solve(A, b, type='LU')
x_GS = mf.solve(A, b, type='GS')
x_HR = mf.solve(A, b, type='HR')
x_GR = mf.solve(A, b, type='GR')
print(f'解方程：\n'
      f'A矩阵:\n{A}\n'
      f'b向量:\n{b}\n'
      f'采用LU分解方程的解:{x_LU}\n'
      f'采用Gram-Schmidt,方程的解:{x_GS}\n'
      f'采用Givens reduction, 方程的解:{x_GR}\n'
      f'采用Householder reduction,方程的解:{x_HR}\n')

###############################################################################
print("#############求行列式###############")
# 4.在矩阵分解基础上求行列式
A = np.array([[3, 2, 2],
              [3, 4, 6],
              [9, 6, 12]])
res_LU = mf.determinant(A, type='LU')
res_GS = mf.determinant(A, type='GS')
res_HR = mf.determinant(A, type='HR')
res_GR = mf.determinant(A, type='GR')
print(f'求行列：\n'
      f'A矩阵:\n{A}\n'
      f'采用LU分解，矩阵行列式:{res_LU}\n'
      f'采用Gram-Schmidt,矩阵行列式:{res_GS}\n'
      f'采用Givens reduction, 矩阵行列式:{res_GR}\n'
      f'采用Householder reduction,矩阵行列式:{res_HR}\n')
