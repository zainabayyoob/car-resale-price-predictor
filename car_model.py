import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Visual layers configure karna
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})
# Output folder banana
os.makedirs('car_project_outputs', exist_ok=True)

print("--- Step 1: Cleaning Messy Car Dataset ---")

FILE_NAME = 'quikr_car.csv'
if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(f"Please ensure '{FILE_NAME}' is in your active working directory.")

car = pd.read_csv(FILE_NAME)

# Cleaning Data
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '').astype(int)
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split(' ').str.slice(0, 3).str.join(' ')
car = car.reset_index(drop=True)

print(f"Cleaned Dataset Dimensions: {car.shape}")

print("\n--- Step 2: Generating Explanatory Visualizations ---")
plt.figure(figsize=(12, 6))
top_companies = car['company'].value_counts().head(10).index
sns.boxplot(data=car[car['company'].isin(top_companies)], x='company', y='Price', palette='Set2')
plt.title('Price Distribution Across Top 10 Car Companies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('car_project_outputs/01_price_distribution.png', dpi=300)
plt.close()

print("\n--- Step 3: Structuring Machine Learning Pipeline ---")
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=433)

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(f"Model R² Score: {r2_score(y_test, y_pred):.4f}")

print("\n--- Step 4: Live Dynamic Pricing Engine ---")

def predict_car_price(name, company, year, kms_driven, fuel_type):
    input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                             columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    prediction = pipe.predict(input_df)
    return max(0, int(prediction[0]))

print("\n🚗 ENTER CAR DETAILS IN THE TERMINAL BELOW TO PREDICT PRICE: 🚗")

try:
    car_name = input("1. Enter Car Model Name (e.g., Maruti Suzuki Swift): ").strip()
    brand = input("2. Enter Company Brand (e.g., Maruti): ").strip()
    model_year = int(input("3. Enter Manufacturing Year (e.g., 2017): "))
    kilometers = int(input("4. Enter Kilometers Driven (e.g., 45000): "))
    fuel = input("5. Enter Fuel Type (Petrol/Diesel/LPG): ").strip()

    predicted_price = predict_car_price(car_name, brand, model_year, kilometers, fuel)
    
    print("\n" + "="*50)
    print("🎯 SYSTEM PREDICTION RESULT")
    print("="*50)
    print(f"🚘 Vehicle  : {car_name} ({model_year})")
    print(f"📊 Metrics  : {kilometers:,} kms | Fuel: {fuel}")
    print(f"💰 Estimated Market Value: ₹{predicted_price:,} INR")
    print("="*50 + "\n")
except Exception as e:
    print(f"Error: {e}. Please ensure all inputs are valid and try again.") 