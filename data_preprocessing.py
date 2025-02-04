import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

# CSV faylini o'qish
data = pd.read_csv('sales_data.csv')

# Ma'lumotlarni tekshirish
print("Ma'lumotlar:")
print(data.head())

# Sana ustunini datetime formatga o'tkazish
data['Date'] = pd.to_datetime(data['Date'])

# Sana va daromadlar ustunlari bilan ishlash
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month

# Linear Regression modelini yaratish
X = data[['Day', 'Month']]  # Kirish xususiyatlari (kun va oy)
y = data['Revenue']  # Maqsad o'zgaruvchisi (daromad)

# Ma'lumotlarni trening va test uchun bo'lish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelni yaratish
model = LinearRegression()
model.fit(X_train, y_train)

# Modelni baholash
y_pred = model.predict(X_test)
print(f"Modelning baholash natijasi: {model.score(X_test, y_test)}")

# Foydalanuvchidan kelajakdagi sana uchun prognoz olish
user_day = int(input("Bashorat qilish uchun kun raqamini kiriting: "))
user_month = int(input("Bashorat qilish uchun oy raqamini kiriting: "))

# Bashorat qilish
future_revenue = model.predict([[user_day, user_month]])
print(f"Kelajakdagi daromad (sana: {user_day}-{user_month}): {future_revenue[0]}")

# Ma'lumotlarni vizualizatsiya qilish
plt.figure(figsize=(10,6))
sns.lineplot(x=data['Date'], y=data['Revenue'], label='Sotuvlar', color='b')
plt.title('Sotuvlar va Daromadlar')
plt.xlabel('Sana')
plt.ylabel('Daromad')
plt.xticks(rotation=45)
plt.tight_layout()

# Grafikni ko'rsatish
plt.show()
