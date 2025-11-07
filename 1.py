import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy import stats

# Налаштування стилю
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'

# === 1. ДАНІ ===
data = {
    "Активність": [
        1.6, 0.8, 1.2, 0.5, 0.9, 1.1, 1.1, 0.6,
        1.5, 0.8, 0.9, 1.2, 0.5, 1.3, 0.8, 1.2,
        1.7, 0.9, 1.3, 0.6, 1.0, 1.2, 1.2, 0.7,
        1.6, 0.9, 1.0, 1.3, 0.6, 1.4, 0.9, 1.3
    ]
}
df = pd.DataFrame(data)
df["День"] = np.arange(1, len(df) + 1)

# === 2. РОЗРАХУНКИ ===
mean_y = df["Активність"].mean()
std_y = df["Активність"].std()
min_y = df["Активність"].min()
max_y = df["Активність"].max()
cv = (std_y / mean_y) * 100
acf_vals = acf(df["Активність"], nlags=4, fft=False)

y_t = df["Активність"].iloc[:-1].values
y_t1 = df["Активність"].iloc[1:].values
m, b = np.polyfit(y_t, y_t1, 1)
r1 = np.corrcoef(y_t, y_t1)[0, 1]

# === 3. СТВОРЕННЯ ГРАФІКІВ ===
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Моніторинг активності користувачів електронної бібліотеки',
             fontsize=16, fontweight='bold', y=0.95)

# --- Графік 1: Часовий ряд ---
ax1 = plt.subplot(2, 2, 1)
ax1.plot(df["День"], df["Активність"], 'o-', color='#2E86AB',
         linewidth=2, markersize=5, markerfacecolor='#2E86AB',
         markeredgecolor='white', markeredgewidth=1)
ax1.axhline(y=mean_y, color='#A23B72', linestyle='--', linewidth=2,
           label=f'Середнє = {mean_y:.2f}')

# Підписи для екстремальних значень з правильними межами
max_idx = df["Активність"].idxmax()
min_idx = df["Активність"].idxmin()

# Встановлюємо межі графіка перед додаванням анотацій
ax1.set_ylim(0.4, 1.9)
ax1.set_xlim(0.5, 32.5)

# Анотації з перевіркою меж
ax1.annotate(f'MAX: {max_y:.1f}', xy=(max_idx+1, max_y),
            xytext=(max_idx+1, max_y+0.1),  # менший зсув
            arrowprops=dict(arrowstyle='->', color='red', lw=1),
            ha='center', fontweight='bold', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

ax1.annotate(f'MIN: {min_y:.1f}', xy=(min_idx+1, min_y),
            xytext=(min_idx+1, min_y-0.1),  # менший зсув
            arrowprops=dict(arrowstyle='->', color='blue', lw=1),
            ha='center', fontweight='bold', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

ax1.set_xlabel('День спостереження', fontsize=11, fontweight='bold')
ax1.set_ylabel('Рівень активності', fontsize=11, fontweight='bold')
ax1.set_title('(а) Графік активності користувачів', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, 33, 3))

# --- Графік 2: Діаграма розсіювання ---
ax2 = plt.subplot(2, 2, 2)

# Встановлюємо межі перед побудовою
ax2.set_xlim(0.4, 1.8)
ax2.set_ylim(0.4, 1.8)

scatter = ax2.scatter(y_t, y_t1, c=range(len(y_t)), cmap='viridis',
                     s=50, alpha=0.7, edgecolors='black', linewidth=0.3)

# Лінія регресії
x_line = np.linspace(0.4, 1.8, 100)
y_line = m * x_line + b
ax2.plot(x_line, y_line, 'r-', linewidth=2,
        label=f'y = {m:.3f}x + {b:.3f}\nR² = {r1**2:.3f}')

# Додаємо колірну шкалу
cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
cbar.set_label('Порядковий номер', fontsize=9)

ax2.set_xlabel('Активність y(t)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Активність y(t+1)', fontsize=11, fontweight='bold')
ax2.set_title('(б) Залежність y(t+1) від y(t)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# --- Графік 3: ACF ---
ax3 = plt.subplot(2, 2, 3)
lags = np.arange(1, 5)
colors = ['#A23B72' if val < 0 else '#2E86AB' for val in acf_vals[1:]]
bars = ax3.bar(lags, acf_vals[1:], color=colors, alpha=0.7,
               edgecolor='black', linewidth=0.5, width=0.6)

# Встановлюємо межі для ACF
ax3.set_ylim(-0.6, 0.3)

# Додаємо значення на стовпці
for i, (bar, val) in enumerate(zip(bars, acf_vals[1:])):
    height = bar.get_height()
    va = 'bottom' if height > 0 else 'top'
    y_text = height + 0.02 if height > 0 else height - 0.03
    ax3.text(bar.get_x() + bar.get_width()/2., y_text,
             f'{val:.3f}', ha='center', va=va, fontweight='bold', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.set_xlabel('Лаг (дні)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Коефіцієнт автокореляції', fontsize=11, fontweight='bold')
ax3.set_title('(в) Функція автокореляції (ACF)', fontsize=12, fontweight='bold')
ax3.set_xticks(lags)
ax3.grid(True, alpha=0.3, axis='y')

# --- Графік 4: Статистика ---
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

# Текст зі статистикою
stats_text = [
    "СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ",
    "─" * 25,
    f"Кількість спостережень: {len(df)}",
    f"Середнє значення: {mean_y:.3f}",
    f"Стандартне відхилення: {std_y:.3f}",
    f"Мінімальна активність: {min_y:.3f}",
    f"Максимальна активність: {max_y:.3f}",
    f"Коефіцієнт варіації: {cv:.1f}%",
    "",
    "АВТОКОРЕЛЯЦІЯ",
    "─" * 25,
    f"r₁ (лаг 1): {acf_vals[1]:.4f}",
    f"r₂ (лаг 2): {acf_vals[2]:.4f}",
    f"r₃ (лаг 3): {acf_vals[3]:.4f}",
    f"r₄ (лаг 4): {acf_vals[4]:.4f}",
    "",
    "РЕГРЕСІЙНИЙ АНАЛІЗ",
    "─" * 25,
    f"Коефіцієнт кореляції: {r1:.4f}",
    f"Коефіцієнт детермінації: {r1**2:.4f}"
]

ax4.text(0.02, 0.98, '\n'.join(stats_text), transform=ax4.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA",
                  edgecolor="#DEE2E6", linewidth=1),
         fontfamily='monospace', linespacing=1.4)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
