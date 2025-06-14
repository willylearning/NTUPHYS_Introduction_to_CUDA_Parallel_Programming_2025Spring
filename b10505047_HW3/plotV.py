import numpy as np
import matplotlib.pyplot as plt

# 讀入數據
data = np.loadtxt('vr_result_64.dat')
r = data[:, 0]
V_numerical = data[:, 1]
V_coulomb = data[:, 2]

# 畫圖
plt.plot(r, V_numerical, 'bo-', label='Numerical V(r)')
plt.plot(r, V_coulomb, 'r--', label='Coulomb V(r) = 1/(4πɛ₀r)')

plt.xlabel('r (distance from center)')
plt.ylabel('Potential V(r)')
plt.title('Numerical vs Coulomb Potential along body diagonal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('vr_result_64.png', dpi=300)
plt.show()