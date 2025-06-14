import matplotlib.pyplot as plt
import numpy as np

# 載入數據
# gmem_data = np.loadtxt('hist_cpu.dat', skiprows=1)
# gmem_data = np.loadtxt('hist_gmem.dat', skiprows=1)
gmem_data = np.loadtxt('hist_shmem.dat', skiprows=1)

x = gmem_data[:, 0]  # 第一列是 bin 中心點
counts = gmem_data[:, 1]  # 第二列是計數

# 參數
N = 81919979  # 實際總計數（與 hist_cpu.dat 一致）
bins = 128
Rmin, Rmax = 0.0, 20.0
binsize = (Rmax - Rmin) / bins  # 0.15625

# 正規化密度
density = counts / (N * binsize)

# 理論分佈
x_theory = np.linspace(Rmin, Rmax, 1000)
f_theory = np.exp(-x_theory)

# 繪圖
plt.bar(x, density, width=binsize, label='Histogram', alpha=0.5)
plt.plot(x_theory, f_theory, 'r-', label='P(x) = exp(-x)')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Histogram vs. Theoretical Probability Distribution')
plt.legend()
# plt.savefig("hist_cpu.png")
# plt.savefig("hist_gmem.png")
plt.savefig("hist_shmem.png")
plt.show()