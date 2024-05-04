
import pandas as pd

# Veri setlerini yükle
train_df = pd.read_csv("train.csv")
weather_df = pd.read_csv("weather.csv")

# "date" sütununu "tarih" olarak değiştirme
weather_df.rename(columns={"date": "tarih"}, inplace=True)
weather_df.rename(columns={"name": "ilce"}, inplace=True)

# Tüm ilçe isimlerini küçük harfe dönüştür
train_df["ilce"] = train_df["ilce"].str.lower()
weather_df["ilce"] = weather_df["ilce"].str.lower()

# Tarih sütunlarını birleştirme için ortak bir isimdeğişken olarak ayarla
train_df.rename(columns={"date": "tarih"}, inplace=True)
weather_df.rename(columns={"date": "tarih"}, inplace=True)


# Tarih sütunlarının formatlarını uygun hale getir
train_df["tarih"] = pd.to_datetime(train_df["tarih"])
weather_df["tarih"] = pd.to_datetime(weather_df["tarih"])


# İlçe sütunlarına göre birleştir
merged_df = pd.merge(train_df, weather_df, on=["tarih", "ilce"])

# Birleştirilmiş veri çerçevesini göster
# print(merged_df)

merged_df.to_csv("merged_weather_train.csv", index=False)

########################################

holidays_df = pd.read_csv("holidays.csv")
test_df = pd.read_csv("test.csv")
# 'Gün', 'Ay' ve 'Yıl' sütunlarını birleştirerek 'Tarih' sütununu oluştur
holidays_df['tarih'] = pd.to_datetime(holidays_df[['Yıl', 'Ay', 'Gün']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')

# İstenen sıraya uygun sütunları seç
holidays_df = holidays_df[['tarih', 'Tatil Adı']]
test_df.rename(columns={"date": "tarih"}, inplace=True)
test_df["ilce"] = test_df["ilce"].str.lower()

# Tarih sütunlarını birleştirerek veri çerçevesini birleştir
merged_df['tarih'] = pd.to_datetime(merged_df['tarih'])
holidays_df['tarih'] = pd.to_datetime(holidays_df['tarih'])
test_df['tarih'] = pd.to_datetime(test_df['tarih'])

all_merged_df = pd.merge(merged_df, holidays_df, on='tarih', how='left')

all_merged_df.to_csv("last.csv", index=False)
######################

# Gereksiz sütunları çıkarma (örneğin, "ilce" sütunu iki kez var gibi görünüyor)
all_merged_df.drop_duplicates(inplace=True)

print(all_merged_df)
