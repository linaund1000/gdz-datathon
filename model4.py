# standart importlar

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("all_Merged.csv")
holidays = pd.read_csv("holidays.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

#print(df.head())
#print(df.info())

# NaN olanları '0' stringi ile değiştirelim
df['Tatil Adı'].fillna('0', inplace=True)

# Diğer değerleri '1' stringi ile değiştirelim
df['Tatil Adı'].replace(to_replace=df['Tatil Adı'][df['Tatil Adı'] != '0'].tolist(), value='1', inplace=True)


def generate_value(date):
    # Yıl, ay ve gün bilgilerini ayırma
    year, month, day = map(str, date.split('-'))
    
    # Ay bilgisini 2 ile, gün bilgisini 1 ile çarpma
    value = month+day
    
    return value

# Fonksiyonu DataFrame'deki tarih sütununa uygulama
df['tarih'] = df['tarih'].apply(lambda x: generate_value(x))


# Label encoding için LabelEncoder kullanma
label_encoder = LabelEncoder()
df['ilce'] = label_encoder.fit_transform(df['ilce'])

# 'Tatil Adı' sütununu int veri türüne dönüştürelim
df['Tatil Adı'] = df['Tatil Adı'].astype(int)
print(df.info())
print(df.head())
X = df.drop("bildirimsiz_sum", axis=1)
y = df["bildirimsiz_sum"]

corr = df.corr()
target_corr = abs(corr["bildirimsiz_sum"])

#print(target_corr.sort_values(ascending=False))

# bu değeri değiştirip deneyler yapabilirsiniz
corr_threshold = 0.02
high_corr_features = target_corr[target_corr > corr_threshold]
# özellik isimlerini alalım ve bildirimsiz_sum özelliğini çıkaralım
hcf_names = [k for k, v in high_corr_features.items()]; hcf_names.remove("bildirimsiz_sum")
#print(hcf_names)

features=["tarih","ilce",'bildirimli_sum', 'lat', 'lon', 't_2m:C', 'effective_cloud_cover:p', 'wind_speed_10m:ms', 'prob_precip_1h:p', 't_apparent:C', 'Tatil Adı']
df['tarih'] = df['tarih'].astype(int)
#modelHighCorr = XGBRegressor()
#modelHighCorr.fit(df[features],y)

test["tarih"] = pd.to_datetime(test["tarih"])
#print(test.info())
test = test.reindex(features, axis=1)
X_test = test[features]
X_test['ilce'] = label_encoder.fit_transform(X_test['ilce'])
#print(X_test.head())
def convert_to_year_month_day(date):
    # Tarihi stringe dönüştürme
    date_str = str(date)
    
    # Tarihi parçalara ayırma
    parts = date_str.split()[0].split("-")
    
    # Yıl, ay ve gün bilgilerini alma ve birleştirme
    year_month_day = "-".join(parts)
    
    return year_month_day
X_test['tarih'] =X_test['tarih'].apply(convert_to_year_month_day)
X_test['tarih'] = X_test['tarih'].apply(lambda x: generate_value(x))
#print(X_test.head())

X_test['tarih'] = X_test['tarih'].astype(int)
#print(X_test.info())

# Parametre aralıklarının belirlenmesi
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}
# Model oluşturma
xgb = XGBRegressor()

# GridSearchCV ile en iyi parametre kombinasyonunun bulunması
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(df[features],y)

# En iyi parametrelerin bulunması
best_params = grid_search.best_params_
print("En iyi parametreler:", best_params)

# XGBoost modelini eğitme
xgboost_model = XGBRegressor(**best_params)
xgboost_model.fit(df[features],y)

# CatBoost modelini eğitme
catboost_model = CatBoostRegressor()
catboost_model.fit(df[features],y)

# Tahminler yapma
#catboost_preds = catboost_model.predict(X_test)
xgboost_preds = xgboost_model.predict(X_test)
catboost_preds = catboost_model.predict(X_test)
print(xgboost_preds)
print(catboost_preds)

ensemble_preds = (catboost_preds + xgboost_preds) / 2
ensemble_preds=np.round(catboost_preds).astype(np.int8)
# Sample submission dosyasına tahminleri ekleyerek yeni bir dosya oluşturma
submission = sample_submission.copy()
submission["bildirimsiz_sum"] = ensemble_preds
submission.to_csv("ensemble3_submission.csv", index=False)
#model2.py