import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Veri setlerini yükleme
train = pd.read_csv("train.csv")
weather = pd.read_csv("weather.csv")
holidays = pd.read_csv("holidays.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")



# Bayram_Flag sütunundaki birden fazla bayramı içeren girişleri işleme
holidays["Tatil Adı"] = holidays["Tatil Adı"].apply(lambda x: x.split(';')[0] if ';' in x else x)

# 'Tatil Adı' sütunundaki birden fazla bayramı içeren girişleri işleme
holidays["Tatil Adı"] = holidays["Tatil Adı"].str.split(';').explode()

# 'Yılbaşı' değerini '1', diğer değerleri '0' olarak kodlayan bir fonksiyon
def encode_holiday_flag(value):
    if value == 'Yılbaşı':
        return 1
    else:
        return 0

# 'Tatil Adı' sütununu kodlayın
holidays['Tatil Adı'] = holidays['Tatil Adı'].apply(encode_holiday_flag)

# Yıl, Ay ve Gün sütunlarını tarih formatına dönüştürme
holidays["Gün"] = pd.to_datetime(holidays[["Yıl", "Ay", "Gün"]].astype(str).agg('-'.join, axis=1))

# Veri setlerini birleştirme
train["tarih"] = pd.to_datetime(train["tarih"])
weather["date"] = pd.to_datetime(weather["date"]) # Tarih sütununu datetime türüne dönüştürme
train = pd.merge(train, weather, left_on=["tarih", "ilce"], right_on=["date", "name"], how="left")
train = pd.merge(train, holidays, left_on=["tarih"], right_on=["Gün"], how="left")

test = pd.merge(test, weather, left_on=["tarih", "ilce"], right_on=["date", "name"], how="left")
test = pd.merge(test, holidays, left_on=["tarih"], right_on=["Gün"], how="left")
test["tarih"] = pd.to_datetime(test["tarih"])
# Özellikleri seçme
features = ["t_2m:C", "effective_cloud_cover:p", "global_rad:W", "relative_humidity_2m:p",
            "wind_dir_10m:d", "wind_speed_10m:ms", "prob_precip_1h:p", "t_apparent:C", "Tatil Adı"]




# Eğitim ve test setlerini oluşturma
X_train = train[features]
y_train = train["bildirimsiz_sum"]
X_test = test[features]

print(X_train)

# CatBoost modelini eğitme
catboost_model = CatBoostRegressor()
catboost_model.fit(X_train, y_train)

# XGBoost modelini eğitme
xgboost_model = XGBRegressor()
xgboost_model.fit(X_train, y_train)

# Tahminler yapma
catboost_preds = catboost_model.predict(X_test)
xgboost_preds = xgboost_model.predict(X_test)

# Ensemble learning için tahminlerin birleştirilmesi (ortalama alarak)
ensemble_preds = (catboost_preds + xgboost_preds) / 2

# Sample submission dosyasına tahminleri ekleyerek yeni bir dosya oluşturma
submission = sample_submission.copy()
submission["bildirimsiz_sum"] = ensemble_preds
submission.to_csv("ensemble_submission.csv", index=False)
