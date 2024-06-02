import pandas as pd
import numpy as np
import gc
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

train =pd.read_csv("./input/store-sales-time-series-forecasting/train.csv")
test=pd.read_csv("./input/store-sales-time-series-forecasting/test.csv")
oil=pd.read_csv("./input/store-sales-time-series-forecasting/oil.csv")
stores=pd.read_csv("./input/store-sales-time-series-forecasting/stores.csv")
transactions=pd.read_csv("./input/store-sales-time-series-forecasting/transactions.csv")
holidays=pd.read_csv("./input/store-sales-time-series-forecasting/holidays_events.csv")
all_data = [train, transactions, stores, test, oil, holidays]
data_names_in_order = ['train', 'transactions', 'stores', 'test', 'oil', 'holidays']

# 전처리

for temp_data in all_data:
    if 'date' in temp_data.columns:
        temp_data.date = pd.to_datetime(temp_data.date)


for c, temp_data in enumerate(all_data):
    if 'date' in temp_data.columns:
        print("Sampling duration starts on", min(temp_data.date), end='  ')
        print("and ends on", max(temp_data.date), "for ", end='')

        print(data_names_in_order[c])

new_oil = oil.set_index('date').dcoilwtico.resample("D").sum().reset_index()
new_oil["dcoilwtico"] = np.where(new_oil["dcoilwtico"] == 0, np.nan, new_oil["dcoilwtico"])
new_oil["dcoilwtico_interpolated"] = new_oil.dcoilwtico.interpolate()


# 데이터셋 결합

train['train'] = 1
test['test'] = 1
data = pd.concat([train,test], axis=0)
data[['train', 'test']] = data[['train', 'test']].fillna(0)
data['date'] = pd.to_datetime(data['date'])


data = data.merge(new_oil[["date", "dcoilwtico_interpolated"]], on='date', how='left')

# stores

stores.rename(columns={"type":"store_type", "cluster":"store_cluster"}, inplace=True)
data = data.merge(stores, on='store_nbr', how='left')
cols = ['city', 'state', 'store_type', 'store_cluster']
data[cols] = data[cols].astype("category")

# holidays

holidays.rename
holidays.rename(columns={"type":"holiday_type", "locale":"holiday_locale", "locale_name":"off_locale_name"}, inplace=True)
holidays['is_holiday'] = True

#Feriados normais de domingo
# Domingo se torna o 7° dia da semana
sundays_hol = data.date.dt.weekday == 6


#hoilday não deve ser transferido para outra data, nem ser dia útil para que incluamos esta condição em todas as condições abaixo
# condição = (holidays.holiday_type != "Work Day") & (holidays.transferred == False)
#Também devemos remover linhas duplicadas que surgiram por pd.merge

holidays["date"] = pd.to_datetime(holidays["date"])

# Feriados Nacionais
feriados_nacionais = holidays[(holidays.holiday_type != "Work Day") & (holidays.transferred == False) & (holidays.holiday_locale == "National")]
feriados_nacionais = data.merge(feriados_nacionais[["date", "is_holiday"]], how="left", on="date")
feriados_nacionais = feriados_nacionais[feriados_nacionais.id<len(data)].drop_duplicates(subset=['id'])
feriados_nacionais['is_holiday'] = feriados_nacionais['is_holiday'].fillna(False)


#Feriados Locais (cidade)
feriados_cidade = holidays[(holidays.holiday_type != "Work Day") & (holidays.transferred == False) & (holidays.holiday_locale == "Local")]
feriados_cidade = data.merge(feriados_cidade.rename(columns={"off_locale_name":"city"})[["date", "is_holiday", "city"]], how="left", on=["date", "city"])
feriados_cidade = feriados_cidade[feriados_cidade.id<len(data)].drop_duplicates(subset=['id'])
feriados_cidade['is_holiday'] = feriados_cidade['is_holiday'].fillna(False)


#Feriados Regionais (Estado)
feriados_estaduais = holidays[(holidays.holiday_type != "Work Day") & (holidays.transferred == False) & (holidays.holiday_locale == "Local")]
feriados_estaduais = data.merge(feriados_estaduais.rename(columns={"off_locale_name":"state"})[["date", "is_holiday", "state"]], how="left", on=["date", "state"])
feriados_estaduais = feriados_estaduais[feriados_estaduais.id<len(data)].drop_duplicates(subset=['id'])
feriados_estaduais['is_holiday'] = feriados_estaduais['is_holiday'].fillna(False)


todos_feriados = sundays_hol.values | feriados_nacionais['is_holiday'].values  | feriados_cidade['is_holiday'].values  | feriados_estaduais['is_holiday'].values
data['is_holiday'] = todos_feriados.astype("bool")


#Calculando média de vendas diárias
venda_media = data.groupby('date').sales.mean().reset_index()
venda_media = venda_media.set_index("date").resample("D").sum().reset_index()
venda_media["sales"] = np.where(venda_media["sales"] == 0, np.nan, venda_media["sales"])
venda_media["sales"] = venda_media["sales"].interpolate()

#Calculando média móvel
venda_media = data.groupby('date').sales.mean().reset_index()
venda_media = venda_media.set_index("date").resample("D").sum().reset_index()
venda_media["sales"] = np.where(venda_media["sales"] == 0, np.nan, venda_media["sales"])
venda_media["sales"] = venda_media["sales"].interpolate()
date_range_complete = venda_media.date.copy()
venda_media.date = np.arange(len(venda_media))
venda_media = venda_media.set_index('date')
window_size = 20
venda_media_movel = venda_media.sales.rolling(window=window_size, center = True).mean()
venda_media_movel.plot()
plt.title('Moving Average plot')

# Treinando o modelo
trend_model = LinearRegression()
date_range = np.array(range(len(venda_media_movel))).reshape(-1, 1)[window_size:-window_size]
venda_media_movel = venda_media_movel[window_size:-window_size]
trend_model.fit(date_range, venda_media_movel)
trend_predictions = pd.DataFrame({"sales":trend_model.predict(np.array(range(len(venda_media))).reshape(-1, 1))})
trend_predictions.index = pd.Index(range(len(venda_media)))
detrended = venda_media - trend_predictions
detrended.plot()

plt.title('detrended data')

time_range = pd.date_range(start=data.date.iloc[0], end=data.date.iloc[-1])
time_arange = np.arange(len(time_range))
predicted = trend_model.predict(time_arange.reshape(-1, 1))
trend_df = pd.DataFrame({"date":list(time_range), "trend":list(predicted)})
data = data.merge(trend_df, how='left', on='date')

# Calculando FFT
y = detrended
fft_result = np.fft.fft(y)

# Frequência
sampling_frequency = 1   # A frequência de amostragem é diária
freq = np.fft.fftfreq(len(y), d=1/sampling_frequency)

# Convertendo frequência
positive_freq_mask = freq > 0
time = 1/(freq+1e-6)[positive_freq_mask]
amplitudes = np.abs(fft_result)[positive_freq_mask]

#Incluindo apenas 1 ano de intervalo
time_mask = time < 370
time = time[time_mask]
amplitudes = amplitudes[time_mask]

# Plot FFT
plt.figure(figsize=(15, 5))
plt.plot(time, amplitudes)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Time Series Data FFT')
plt.grid(True)
plt.show()

#Incluindo apenas 1 ano de intervalo
time_mask = time < 32
time = time[time_mask]
amplitudes = amplitudes[time_mask]

# Plot FFT
plt.figure(figsize=(15, 5))
plt.plot(time, amplitudes)
plt.xlabel(' (Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT dos dados de Série Temporal')
plt.grid(True)
plt.show()

temp = data.copy()
temp['day'] = train.date.dt.day
temp['month'] = train.date.dt.month
x = temp.groupby(temp.month).day.max()



# 급여일인지 확인 중. 급여는 매월 말일과 15일에 지급되며, 급여일의 다음 날에도 지급됩니다.
# 급여일
a = (data.date.dt.day == 15) | (data.date.dt.day == 31)
b = data.date.dt.month.isin([4, 6, 9, 11]) & data.date.dt.day == 30
c = data.date.dt.month.isin([2]) & data.date.dt.day == 29
data['iswageday'] = a | b | c

#o dia seguinte ao dia do salário
a = (data.date.dt.day == 16) | (data.date.dt.day == 1)
data['day_after_wageday'] = a

data[['iswageday', 'day_after_wageday']] = data[['iswageday', 'day_after_wageday']]

data['dayofweek'] = data.date.dt.weekday + 1 # +1 para evitar os zerados
data['month'] = data.date.dt.month
data['dayofyear'] = data.date.dt.dayofyear
data['dayofmonth'] = data.date.dt.day
data['year'] = data.date.dt.year
data['weekofmonth'] = ((data.date.dt.day-1) // 7) + 1
data['season'] = np.where(data.date.dt.month.isin([1,2,12]), 0, 1)
data['season'] = np.where(data.date.dt.month.isin([6,7,8]), 2, data['season'])
data['season'] = np.where(data.date.dt.month.isin([9,10,11]), 3, data['season'])
date_features = ['year', 'season', 'month', 'dayofmonth', 'dayofyear', 'dayofweek', 'weekofmonth']
data[date_features] = data[date_features].astype("int")

# Convertendo Onpromotion como coluna booleana aos dados
data['is_onpromotion'] = data['onpromotion'] > 0

fourier = CalendarFourier(freq="A", order=500)
dates = pd.date_range(start='2013-01-01', end='2017-08-15', freq='D')
dp = DeterministicProcess(
    index=dates,
    constant=True,
    #order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
Z = dp.in_sample()


plot_pacf(detrended, lags=100, title='Partial Autocorrelation Function (PACF)')
plt.show()

def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax

def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig


plot_lags(detrended.sales, lags = 370, nrows = 10, ncols = 4)

def add_lag_features(inp_data, lags):
    out_data = data.copy()
    for lag_number in lags:
        out_data["lag_"+ str(lag_number)] = lagged = inp_data.groupby(["family", "store_nbr"]).shift(lag_number).reset_index().sales
    return out_data

lag_numbers = [21, 27, 28, 35, 41, 49, 56, 63, 70, 77, 80]
lag_numbers_2 = [91, 108, 114, 126, 136, 140, 158, 182, 196, 259, 287, 322, 336, 364]
lagged_data = add_lag_features(data, lag_numbers + lag_numbers_2)

lagged_data[["sales"]+lagged_data.columns[lagged_data.columns.str.startswith("lag_")].tolist()].corr()

final_data = lagged_data.copy()
final_data = final_data[final_data.train == 1]
#Eliminando a coluna de vendas e definindo como y
final_data = final_data.drop(columns=['sales'], axis=1)

#Features categóricas
categorical_features = ['family', 'dayofweek','month','year','weekofmonth','season',
                        'city','state','store_type','store_cluster']

# One-hot encode para as colunas de features categóricas
final_data = pd.get_dummies(final_data, columns=categorical_features)

label_encoder = LabelEncoder()
final_data['daterange'] = np.arange(len(final_data['date']))
final_data = final_data.drop(columns=['date'])
y = train.sales
data_train, data_test, y_train, y_test = train_test_split(final_data, y, test_size=0.2, random_state=42)


params = {'objective':'reg:squarederror',  'eval_metric':'rmse'}

# Convertendo dados de treino para DMatrix
dtrain = xgb.DMatrix(data_train, label=y_train)

# Treinando o modelo dataGBoost
num_rounds = 100  
bst = xgb.train(params, dtrain, num_rounds)

def rmsle(y_true, y_pred):
    epsilon = 1e-6
    assert len(y_true) == len(y_pred)
    a = np.log1p(y_pred + epsilon)
    b = np.log1p(y_true + epsilon)
    squared_errors = ( a - b ) ** 2
    mean_squared_log_error = np.mean(squared_errors)
    rmsle = np.sqrt(mean_squared_log_error)
    return rmsle

# Convertendo dados de teste para DMatrix
dtest = xgb.DMatrix(data_test)

# Predição nos dados de teste
predictions = bst.predict(dtest)

# Avalia performance do modelo
from sklearn.metrics import mean_squared_error

y_test_trended = y_test
predictions_trended = predictions
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
print("Root Mean Squared Error:", rmse)

rmsle_score = rmsle(y_test, predictions)
print("RMSLE:", rmsle_score)

test_data = lagged_data[lagged_data.test == 1].copy()
test_data = test_data.drop(columns=['sales'], axis=1)
test_data = pd.get_dummies(test_data, columns=categorical_features)
test_data['daterange'] = np.arange(len(test_data['date']))
test_data = test_data.drop(columns=['date'])

train_columns = final_data.columns.tolist()
test_columns = test_data.columns.tolist()
test_data[list(set(train_columns)-set(test_columns))] = False
train_columns = final_data.columns
test_data_reordered = test_data[train_columns]

# Convertendo dados de teste para DMatrix
dtest = xgb.DMatrix(test_data_reordered)
predictions = bst.predict(dtest)
# Não deve haver valores negativos. substituímos os negativos por zero porque provavelmente não mostram vendas
predictions[predictions < 0] = 0
submission = pd.DataFrame({"id":test_data_reordered.id.values, "sales":predictions})
submission.to_csv('submission.csv', index = False)