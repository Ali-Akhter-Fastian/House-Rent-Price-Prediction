import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# House Rent Price Prediction - Linear Regression Model

df = pd.read_csv('house_price.csv')
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = X_train.copy()
X_test = X_test.copy()

num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(include='object').columns

#  OUTLIER HANDLING (train set only)

Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

price_mask = (y_train >= lower_bound) & (y_train <= upper_bound)

print(f"Removed {(~price_mask).sum()} outlier rows out of {len(y_train)} training rows")

X_train = X_train[price_mask]
y_train = y_train[price_mask]


imp_num = SimpleImputer(strategy='mean')
imp_cat = SimpleImputer(strategy='most_frequent')

X_train[num_cols] = imp_num.fit_transform(X_train[num_cols])
X_test[num_cols] = imp_num.transform(X_test[num_cols])

X_train[cat_cols] = imp_cat.fit_transform(X_train[cat_cols])
X_test[cat_cols] = imp_cat.transform(X_test[cat_cols])

# FEATURE SELECTION

mi = mutual_info_regression(X_train[num_cols], y_train, random_state=42)
mi_series = pd.Series(mi, index=num_cols)

print("Mutual information scores:\n", mi_series.sort_values(ascending=False))

selected_num = mi_series[mi_series > 0.1].index

# Keep selected numerical + ALL categorical
selected_features = list(selected_num) + list(cat_cols)

X_train = X_train[selected_features]
X_test = X_test[selected_features]

# Update categorical columns after selection
cat_cols = X_train.select_dtypes(include='object').columns

# ENCODING


ct = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
# SCALING
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
    X_test = X_test.toarray()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Predicted (first 5):", y_pred[:5])


# EVALUATION

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R2 Score:", r2 * 100)

encoded_cat_names = ct.named_transformers_['cat'].get_feature_names_out(cat_cols)
final_feature_names = list(encoded_cat_names) + [c for c in selected_features if c not in cat_cols]

coef_series = pd.Series(model.coef_, index=final_feature_names).sort_values(key=abs, ascending=False)
print("Feature Coefficients:\n", coef_series)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot.png")
plt.close()


# NEW DATA PREDICTION

new_record = pd.DataFrame({
    'number_of_bedrooms': [3],
    'square_footage': [1200],
    'locations': ['street no 04'],
    'age': [10]
})


new_record = new_record[selected_features]
new_record = ct.transform(new_record)

if hasattr(new_record, "toarray"):
    new_record = new_record.toarray()

new_record = scaler.transform(new_record)

prediction = model.predict(new_record)

print("New House Predicted Price:", prediction[0])