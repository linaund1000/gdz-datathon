# Gerekli kütüphanelerin kurulması
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

allMerged=pd.read_csv("all_merged.csv")
weather = pd.read_csv("weather.csv")
holidays = pd.read_csv("holidays.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")

def generate_value(date):
    # Yıl, ay ve gün bilgilerini ayırma
    year, month, day = map(int, date.split('-'))
    
    # Ay bilgisini 1 ile, gün bilgisini 2 ile çarpma
    value = month * 1 + day * 2
    
    return value

# Fonksiyonu DataFrame'deki tarih sütununa uygulama
allMerged['tarih'] = allMerged['tarih'].apply(lambda x: generate_value(x))


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

def convert_to_year_month_day(date):
    # datetime nesnesini oluşturma
    dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    
    # Yıl, ay ve gün bilgilerini alma ve birleştirme
    year_month_day = dt.strftime("%Y-%m-%d")
    
    return year_month_day
weather['date'] = weather['date'].apply(convert_to_year_month_day)
# Yıl, Ay ve Gün sütunlarını tarih formatına dönüştürme
holidays["Gün"] = pd.to_datetime(holidays[["Yıl", "Ay", "Gün"]].astype(str).agg('-'.join, axis=1))
weather["date"] = pd.to_datetime(weather["date"]) # Tarih sütununu datetime türüne dönüştürme
test["tarih"] = pd.to_datetime(test["tarih"])
test = pd.merge(test, weather, left_on=["tarih", "ilce"], right_on=["date", "name"], how="left")
test = pd.merge(test, holidays, left_on=["tarih"], right_on=["Gün"], how="left")

# Özellikleri seçme
features = ["tarih","ilce","bildirimli_sum","t_2m:C", "effective_cloud_cover:p", "global_rad:W", "relative_humidity_2m:p",
            "wind_dir_10m:d", "wind_speed_10m:ms", "prob_precip_1h:p", "t_apparent:C"]
# Label encoding için LabelEncoder kullanma
label_encoder = LabelEncoder()

# Eğitim ve test setlerini oluşturma
X_train = allMerged[features]
X_train['ilce'] = label_encoder.fit_transform(X_train['ilce'])
y_train = allMerged["bildirimsiz_sum"]
print(X_train.info())
# Kalmasını istediğiniz değişken sayısı
select_k = 5

# Değişkenlerin seçim stratejisi belirlenir
# f_classif = ANOVA yönteminin kullanılmasıdır
selection = SelectKBest(f_classif, k=select_k).fit(X_train, y_train)

# İlişkili olan değişkenler gösterilir.
features = X_train.columns[selection.get_support()]
print(features)