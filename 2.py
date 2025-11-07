import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Налаштування стилю
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'

# === ПАРАМЕТРИ МОДЕЛІ ===
np.random.seed(42)
n = 500
trend = 0.015  # Сталий тренд (щоденне зростання активності)
volatility = 0.8  # Волатильність
initial_value = 20  # Початкова активність

# === ГЕНЕРАЦІЯ ДАНИХ ===
dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
y = [initial_value]
for t in range(1, n):
    y_t = y[t-1] + trend + np.random.normal(0, volatility)
    y.append(y_t)

df = pd.DataFrame({
    'Дата': dates,
    'Активність': y,
    'Тренд': initial_value + trend * np.arange(n)
})

# === ПАРАМЕТРИ ПРОГНОЗУ ===
current_time = 450  # Поточний момент для прогнозу
forecast_horizon = 50  # Прогноз на 50 днів
current_value = df['Активність'].iloc[current_time]

# === ФУНКЦІЯ ПРОГНОЗУ ===
def random_walk_forecast(current_val, horizon, trend_param, vol, n_sim=1000):
    forecasts = []
    for _ in range(n_sim):
        forecast_vals = [current_val]
        for t in range(1, horizon + 1):
            next_val = forecast_vals[t-1] + trend_param + np.random.normal(0, vol)
            forecast_vals.append(next_val)
        forecasts.append(forecast_vals)
    return np.array(forecasts)

# Генерація прогнозів
forecasts = random_walk_forecast(current_value, forecast_horizon, trend, volatility)
mean_forecast = forecasts.mean(axis=0)
std_forecast = forecasts.std(axis=0)

# Теоретичні значення
horizons = np.arange(forecast_horizon + 1)
theoretical_mean = current_value + trend * horizons
theoretical_variance = volatility**2 * horizons
theoretical_std = np.sqrt(theoretical_variance)

# === ВІЗУАЛІЗАЦІЯ ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Модель випадкового блукання з трендом для прогнозування активності користувачів',
             fontsize=16, fontweight='bold', y=0.95)

# === ГРАФІК 1: ВИПАДКОВЕ БЛУКАННЯ З ТРЕНДОМ ===
ax1.plot(df['Дата'], df['Активність'], 'b-', alpha=0.7, linewidth=1, label='Активність користувачів')
ax1.plot(df['Дата'], df['Тренд'], 'r--', linewidth=2, label=f'Тренд (μ = {trend:.3f})')

# Позначка початку прогнозу
forecast_start = df['Дата'].iloc[current_time]
ax1.axvline(x=forecast_start, color='black', linestyle='-', linewidth=2, label='Початок прогнозу')
ax1.axhline(y=current_value, color='green', linestyle='--', alpha=0.7, label=f'Поточне значення: {current_value:.1f}')

ax1.set_title('(а) Випадкове блукання з трендом: історична активність', fontsize=14, fontweight='bold')
ax1.set_xlabel('Дата')
ax1.set_ylabel('Активність користувачів')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# === ГРАФІК 2: ПРОГНОЗ ТА ПОХИБКА ===
forecast_dates = pd.date_range(start=forecast_start, periods=forecast_horizon + 1, freq='D')

# Прогноз та довірчі інтервали
ax2.plot(forecast_dates, mean_forecast, 'g-', linewidth=3, label='Середній прогноз')
ax2.fill_between(forecast_dates,
                mean_forecast - 1.96 * std_forecast,
                mean_forecast + 1.96 * std_forecast,
                alpha=0.3, color='green', label='95% довірчий інтервал')

# Теоретичне сподівання
ax2.plot(forecast_dates, theoretical_mean, 'r--', linewidth=2,
         label=f'Теоретичне сподівання: E[y(t+τ)] = y(t) + τ·μ')

ax2.set_title('(б) Прогноз та похибка прогнозування', fontsize=14, fontweight='bold')
ax2.set_xlabel('Дата')
ax2.set_ylabel('Активність')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Додаткові графіки для аналізу похибки
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))


# === ГРАФІК 3: СЕРЕДНЯ КВАДРАТИЧНА ПОХИБКА ===
mse_empirical = std_forecast**2
mse_theoretical = theoretical_variance

ax3.plot(horizons, mse_empirical, 'bo-', markersize=4, label='Емпірична MSE')
ax3.plot(horizons, mse_theoretical, 'r-', linewidth=2, label='Теоретична MSE = τ·σ²')
ax3.set_xlabel('Горизонт прогнозу (τ, дні)')
ax3.set_ylabel('Середня квадратична похибка (MSE)')
ax3.set_title('Залежність MSE від горизонту прогнозу', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# === ГРАФІК 4: СТАНДАРТНЕ ВІДХИЛЕННЯ ПОХИБКИ ===
ax4.plot(horizons, std_forecast, 'bo-', markersize=4, label='Емпіричне стандартне відхилення')
ax4.plot(horizons, theoretical_std, 'r-', linewidth=2, label='Теоретичне σ√τ')
ax4.set_xlabel('Горизонт прогнозу (τ, дні)')
ax4.set_ylabel('Стандартне відхилення похибки')
ax4.set_title('Залежність похибки від горизонту прогнозу', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
