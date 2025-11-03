# Practical-work-No.-2

Аналіз часового ряду: Моделювання успішної бізнес-моделі у сфері цифрових технологій

Вступ

Цей репозиторій містить матеріали та звіт про виконання практичної роботи, метою якої був комплексний аналіз та прогнозування часового ряду, що відображає динаміку ключових показників (KPI) успішної бізнес-моделі у сфері цифрових технологій.

Для аналізу була використана мова програмування Python з бібліотекою statsmodels.

1. Дослідження та стабілізація часового ряду

Перевірка стаціонарності

На першому етапі було проведено Розширений тест Дікі-Фуллера (ADF) для перевірки стаціонарності часового ряду.

    Результат тесту ADF: p-значення = 1.0.

    Висновок: Оскільки p-значення > 0.05, ряд є нестаціонарним (містить тренд).

Стабілізація

Для усунення тренду та приведення ряду до стаціонарного вигляду було застосовано диференціювання першого порядку (d=1).

    Рис. 1. Графіки вихідного нестаціонарного ряду та його стабілізованого диференційованого вигляду.

2. Побудова моделі ARIMA(1,1,1)

Після стабілізації даних була обрана та навчена модель ARIMA(1,1,1):

    AR (p=1): Один авторегресійний компонент.

    I (d=1): Порядок диференціювання (для стабілізації).

    MA (q=1): Один компонент ковзного середнього.

Аналіз параметрів

    Авторегресійний коефіцієнт (ar.L1): Виявився статистично значущим.

    Коефіцієнт ковзного середнього (ma.L1): Виявився незначущим.

        Примітка: Хоча ma.L1 є незначущим, обрана модель ARIMA(1,1,1) була використана як прийнятна для поточного аналізу.

3. Оцінка адекватності моделі

Для підтвердження якості моделі було проаналізовано її залишки (residuals). Адекватна модель залишає після себе залишки, які є "білим шумом" (випадковий ряд без автокореляції).

Аналіз залишків

    Корелограма ACF: Піки на графіку автокореляційної функції (ACF) залишків знаходяться в межах довірчого інтервалу, що підтверджує відсутність залежностей.

    Тест Л'юнга-Бокса (Ljung-Box Test):

        Prob(Q) (p-значення) = 0.22.

        Висновок: Оскільки 0.22>0.05, статистично підтверджено, що залишки є білим шумом, і модель є адекватною.


4. Прогнозування

Використовуючи побудовану та перевірену модель, було зроблено прогноз на 10 періодів уперед.

    Графік прогнозу відображає передбачувані значення, а також довірчий інтервал, що показує ступінь невизначеності прогнозу.

    Рис. 2. Прогноз часового ряду на 10 періодів уперед з довірчим інтервалом.

Вихідні матеріали

Код
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# 1. Часовий ряд (тут приклад - дохід компанії за роками, підстав свій варіант)
y = np.array([25, 30, 35, 40, 48, 55, 63, 70, 82, 95, 110, 130, 150, 170])

# Крок 1. Перевірка стаціонарності (ADF-тест)
result = adfuller(y)
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] > 0.05:
    print("Ряд НЕ стаціонарний")
else:
    print("Ряд стаціонарний")

# Крок 2. Приведення до стаціонарного вигляду
if result[1] > 0.05:
    y_proc = np.diff(y)
else:
    y_proc = y

# Крок 3. Побудова моделі ARIMA
# (тут ми задаємо p=1, d=1, q=1 як приклад - параметри підбираються через ACF/PACF)
model = sm.tsa.ARIMA(y, order=(1,1,1))

# Крок 4. Оцінка параметрів
res = model.fit()
print(res.summary())

# Крок 5. Перевірка адекватності
residuals = res.resid
fig, ax = plt.subplots(1,2,figsize=(10,4))
ax[0].plot(residuals)
ax[0].set_title('Залишки моделі')
sm.graphics.tsa.plot_acf(residuals, ax=ax[1])
plt.show()

# Крок 6. Прогнозування на 10 періодів
forecast = res.get_forecast(steps=10)
mean_forecast = forecast.predicted_mean
conf_int = forecast.conf_int()

plt.figure(figsize=(8,5))
plt.plot(y, label="Фактичні дані")
plt.plot(range(len(y), len(y)+10), mean_forecast, 'r--', label="Прогноз")
plt.fill_between(range(len(y), len(y)+10),
                 conf_int.iloc[:,0],
                 conf_int.iloc[:,1], color='pink', alpha=0.3)

plt.legend()
plt.title("Прогноз на 10 періодів уперед")
plt.show()

Результати
PS E:\Проекти пайтон> python pract2.py
ADF Statistic: 11.69836931046645
p-value: 1.0
Ряд НЕ стаціонарний

C:\Users\User\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-moving average components (MA) of order >= 1 were found.
You should consider setting non-stationary starting autoregressive parameters.
  warn('Non-moving average components (MA) of order >= 1 were found.'
C:\Users\User\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-stationary starting autoregressive parameters.
  warn('Non-stationary starting autoregressive parameters'
SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                   14
Model:                     ARIMA(1, 1, 1)   Log Likelihood:                -31.348
Date:                 Tue, 23 Sep 2025   AIC:                            68.697
Time:                         21:08:08   BIC:                            70.391
Sample:                              0   HQIC:                           68.348
                            - 14                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err    z    P>|z|   [0.025   0.975]
------------------------------------------------------------------------------
ar.L1          0.9844      0.041   23.889   0.000    0.904    1.065
ma.L1          0.0663      0.468    0.142   0.887   -0.851    0.983
sigma2         5.5145      2.105    2.620   0.009    1.389    9.640
==============================================================================
Ljung-Box (L1) (Q):          1.48   Jarque-Bera (JB):                1.82
Prob(Q):                     0.22   Prob(JB):                        0.40
Heteroskedasticity (H):      3.02   Skew:                            0.90
Prob(H) (two-sided):         0.31   Kurtosis:                        2.65
==============================================================================
Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).


Висновок

Усі поставлені завдання з аналізу часового ряду було успішно виконано. Застосовані методи дозволили визначити властивості даних, побудувати та перевірити адекватність моделі ARIMA(1,1,1), а також виконати надійне прогнозування ключових показників бізнес-моделі.
