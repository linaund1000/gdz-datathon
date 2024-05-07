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

df = pd.read_csv("ortalama_veriler.csv")
test = pd.read_csv("testVeri.csv")
sample_submission = pd.read_csv("sample_submission.csv")

df = df.drop(columns=['tarih1'])
# Label encoding için LabelEncoder kullanma
label_encoder = LabelEncoder()
df['ilce'] = label_encoder.fit_transform(df['ilce'])

def generate_value(date):
    # Yıl, ay ve gün bilgilerini ayırma
    year, month, day = map(str, date.split('-'))
    
    # Ay bilgisini 2 ile, gün bilgisini 1 ile çarpma
    value = month*2+day
    
    return value

# Fonksiyonu DataFrame'deki tarih sütununa uygulama
df['tarih'] = df['tarih'].apply(lambda x: generate_value(x))
df['tarih'] = df['tarih'].astype(int)


X = df.drop("bildirimsiz_sum", axis=1)
y = df["bildirimsiz_sum"]

test["tarih"] = pd.to_datetime(test["tarih"])

X_test = test
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
X_test['tarih'] = X_test['tarih'].astype(int)
X_test['ilce'] = label_encoder.fit_transform(X_test['ilce'])

# Parametre aralıklarının belirlenmesi
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [50, 100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}


# Model oluşturma
xgb = XGBRegressor()

# GridSearchCV ile en iyi parametre kombinasyonunun bulunması
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X,y)

# En iyi parametrelerin bulunması
best_params = grid_search.best_params_
print("En iyi parametreler:", best_params)

# XGBoost modelini eğitme
xgboost_model = XGBRegressor(**best_params)
xgboost_model.fit(X,y)

# CatBoost modelini eğitme
catboost_model = CatBoostRegressor()
catboost_model.fit(X,y)

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
submission.to_csv("ensemble7_submission.csv", index=False)