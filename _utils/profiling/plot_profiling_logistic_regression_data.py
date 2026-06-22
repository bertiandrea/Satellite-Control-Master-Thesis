import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x_envs = np.array([1, 2, 4, 8, 16, 32, 64, 128])  # Numero di ambienti
x_log = np.log10(x_envs)

y = np.array([6.81, 8.86, 12.82, 18.61, 27.57, 43.9, 61.59, 74.49])

def logistic4(x, A, K, B, M):
    return A + (K - A) / (1 + np.exp(-B * (x - M)))

# Parametri iniziali e bounds
p0_log = [0, 100, 1, np.median(x_log)]
bounds_log = ([0, 0, 0, x_log.min()], [100, 100, np.inf, x_log.max()])
# Fit della curva logistica
params_logX, _ = curve_fit(logistic4, x_log, y, p0=p0_log, bounds=bounds_log, maxfev=20000)
# Calcolo R²
y_fit_logX = logistic4(x_log, *params_logX)
R2_logX = 1 - np.sum((y - y_fit_logX)**2) / np.sum((y - np.mean(y))**2)

# Predizione per nuovi valori
x_new = np.array([258, 512])
y_pred_logX = logistic4(np.log10(x_new), *params_logX)

# Plot dei risultati
plt.figure(figsize=(10, 6))
plt.scatter(x_envs, y, c='black', marker='o', label='Dati reali')

x_plot = np.linspace(x_envs.min(), max(x_envs.max(), x_new.max()), 300)
plt.plot(x_plot, logistic4(np.log10(x_plot), *params_logX), ls='--', lw=2,
         label=f'Logistic (log x) R²={R2_logX:.3f}')

plt.scatter(x_new, y_pred_logX, marker='D', s=100, label='Pred Logistica')

for xi, yv in zip(x_new, y_pred_logX):
    plt.annotate(f'{yv:.1f}%', (xi, yv),
                 textcoords="offset points", xytext=(0, 10), ha='center')

plt.xlabel('Numero di ambienti [Migliaia]')
plt.ylabel('Utilizzo GPU (%)')
plt.title(f'Logistic Regression (log x) - GPU vs # Ambienti - R²={R2_logX:.3f}')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize='small')
plt.tight_layout()
plt.savefig('gpu%_vs_#env.png', dpi=300, bbox_inches='tight')
plt.show()
