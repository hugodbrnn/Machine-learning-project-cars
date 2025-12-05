import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

df = pd.read_csv('autos.csv')
# ... (rest of the data cleaning and preparation code is the same)
def impute_by_similarity(df, target_col, keys, min_matches=3):
    df = df.copy()
    missing_mask = df[target_col].isna()
    n_missing = missing_mask.sum()
    if n_missing == 0:
        return df, 0
    df_missing = df.loc[missing_mask, keys].copy()
    df_missing["_row_id"] = df_missing.index
    df_known = df.loc[~missing_mask, keys + [target_col]].copy()
    merged = df_missing.merge(df_known, on=keys, how="left", suffixes=("", "_y"))
    merged = merged.dropna(subset=[target_col])
    if merged.empty:
        return df, 0
    grouped = merged.groupby("_row_id")[target_col]
    counts = grouped.size()
    modes = grouped.agg(lambda s: s.mode().iloc[0])
    valid_idx = counts[counts >= min_matches].index
    if len(valid_idx) == 0:
        return df, 0
    mode_per_row = modes.loc[valid_idx]
    df.loc[mode_per_row.index, target_col] = mode_per_row
    n_imputed = len(mode_per_row)
    return df, n_imputed

def process_data(df):
    df = df.copy()
    cat_cols_to_impute = ["vehicleType", "gearbox", "model", "fuelType"]
    repair_col = "notRepairedDamage"
    for col in cat_cols_to_impute + [repair_col]:
        df[f"{col}_was_missing"] = df[col].isna()
    similarity_keys = {
        "model":       ["brand", "vehicleType", "fuelType", "gearbox", "powerPS", "yearOfRegistration"],
        "fuelType":    ["brand", "vehicleType", "gearbox", "model", "powerPS", "yearOfRegistration"],
        "gearbox":     ["brand", "vehicleType", "fuelType", "model", "powerPS", "yearOfRegistration"],
        "vehicleType": ["brand", "fuelType", "gearbox", "model", "powerPS", "yearOfRegistration"],
    }
    initial_missing = df[cat_cols_to_impute].isna().sum()
    similarity_counts = {}
    remaining_after_similarity = {}
    for col in cat_cols_to_impute:
        df, n_imp = impute_by_similarity(df, col, similarity_keys[col], min_matches=3)
        similarity_counts[col] = n_imp
        remaining_after_similarity[col] = df[col].isna().sum()
    for col in cat_cols_to_impute:
        remaining = df[col].isna().sum()
        if remaining > 0:
            unknown_label = f"Unknown_{col}"
            df[col] = df[col].fillna(unknown_label)
    if repair_col in df.columns:
        df[repair_col] = df[repair_col].fillna("unknown")
    return df

df = process_data(df)

def clean_impossible_entries(df):
    df_clean = df.copy()
    mask_year = df_clean["yearOfRegistration"].between(1950, 2016)
    mask_month = df_clean["monthOfRegistration"].between(1, 12)
    mask_price = df_clean["price"] > 0
    mask_power = df_clean["powerPS"] > 0
    mask_offer = df_clean["offerType"] == "Angebot"
    mask = mask_year & mask_month & mask_price & mask_power & mask_offer
    df_clean = df_clean[mask].reset_index(drop=True)
    return df_clean

df = clean_impossible_entries(df)

def treat_extreme_outliers(df):
    df_clean = df.copy()
    q1_price = df_clean["price"].quantile(0.25)
    q3_price = df_clean["price"].quantile(0.75)
    iqr_price = q3_price - q1_price
    lower_price = max(500, q1_price - 1.5 * iqr_price)
    upper_price = min(150000, q3_price + 1.5 * iqr_price)
    df_clean = df_clean[df_clean["price"].between(lower_price, upper_price)]
    q1_power = df_clean["powerPS"].quantile(0.25)
    q3_power = df_clean["powerPS"].quantile(0.75)
    iqr_power = q3_power - q1_power
    lower_power = max(40, q1_power - 1.5 * iqr_power)
    upper_power = min(300, q3_power + 1.5 * iqr_power)
    df_clean = df_clean[df_clean["powerPS"].between(lower_power, upper_power)]
    return df_clean

df = treat_extreme_outliers(df)

def fix_dtypes(df):
    df = df.copy()
    date_cols = ["dateCrawled", "dateCreated", "lastSeen"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    cat_cols = ["name", "seller", "offerType", "abtest", "vehicleType", "gearbox", "model", "fuelType", "brand", "notRepairedDamage"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    int_cols = ["yearOfRegistration", "monthOfRegistration", "powerPS", "kilometer", "price", "postalCode", "nrOfPictures"]
    for col in int_cols:
         if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    return df
    
df = fix_dtypes(df)

def remove_duplicates(df, remove_near_duplicates=True):
    df_clean = df.copy()
    if "index" in df_clean.columns:
        id_col = "index"
    else:
        id_col = None
    if id_col is not None:
        strict_cols = [c for c in df_clean.columns if c != id_col]
        df_clean = df_clean.drop_duplicates(subset=strict_cols, keep="first")
    else:
        df_clean = df_clean.drop_duplicates(keep="first")
    if remove_near_duplicates:
        sig_cols = ["name", "seller", "brand", "model", "yearOfRegistration", "monthOfRegistration", "gearbox", "fuelType", "vehicleType", "powerPS", "kilometer", "postalCode"]
        sig_cols = [c for c in sig_cols if c in df_clean.columns]
        if "dateCreated" in df_clean.columns:
            df_clean = df_clean.sort_values("dateCreated")
        df_clean = df_clean.drop_duplicates(subset=sig_cols, keep="first")
    return df_clean
    
df = remove_duplicates(df, remove_near_duplicates=True)

cols_to_drop = ['index', 'name', 'dateCrawled', 'dateCreated', 'lastSeen', 'seller', 'offerType', 'nrOfPictures']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

def features(df):
    df = df.copy()
    # Create a mapping from category codes back to original brand names
    brand_map = dict(enumerate(df['brand'].cat.categories))
    
    # Feature Engineering
    df['car_age'] = 2016 - df['yearOfRegistration']
    df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)
    df['ps_per_year'] = df['powerPS'] / (df['car_age'] + 1)
    
    # Use the map to identify luxury brands
    luxury_brands = ['porsche', 'audi', 'bmw', 'mercedes_benz']
    df['is_luxury_brand'] = df['brand'].map(brand_map).isin(luxury_brands).astype(int)

    cat_cols = ['abtest', 'vehicleType', 'gearbox', 'model', 'fuelType', 'brand', 'notRepairedDamage']
    for col in cat_cols:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.codes

    df["brand_model_interaction"] = df['brand'].astype(str) + '_' + df['model'].astype(str)
    df['brand_model_interaction'] = LabelEncoder().fit_transform(df['brand_model_interaction'])
    
    return df

df = features(df)

from sklearn.cluster import KMeans
features_for_clustering = ['car_age', 'powerPS', 'kilometer', 'vehicleType', 'brand']
df_cluster = df[features_for_clustering]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df_scaled)

X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# --- Model Training ---
# GridSearchCV for Random Forest
param_grid_rf = {'n_estimators': [100], 'max_depth': [None, 20]}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
rf_best_pred = grid_rf.best_estimator_.predict(X_test)


scaler_pca = StandardScaler()
X_train_scaled_pca = scaler_pca.fit_transform(X_train)
X_test_scaled_pca = scaler_pca.transform(X_test)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled_pca)
X_test_pca = pca.transform(X_test_scaled_pca)
lr_pca_model = LinearRegression()
lr_pca_model.fit(X_train_pca, y_train)
lr_pca_pred = lr_pca_model.predict(X_test_pca)

# Other models
baseline_pred = np.full_like(y_test, y_train.mean())
lr_model = LinearRegression().fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
tree_model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
gb_model = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
xgb_model = XGBRegressor(random_state=42).fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
cat_model = CatBoostRegressor(verbose=0, random_state=42).fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)

# Neural Network
scaler_nn = StandardScaler()
X_train_scaled_nn = scaler_nn.fit_transform(X_train)
X_test_scaled_nn = scaler_nn.transform(X_test)
nn_model = Sequential([
    Dense(256, activation='relu', input_shape=[X_train.shape[1]]),
    BatchNormalization(), Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(1)
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=0.00001)
nn_model.fit(X_train_scaled_nn, y_train, validation_split=0.2, epochs=100, batch_size=128, callbacks=[early_stopping, reduce_lr], verbose=0)
nn_pred = nn_model.predict(X_test_scaled_nn).flatten()

# --- Evaluation and Plotting ---
models = {
    "Baseline": baseline_pred,
    "Linear Regression": lr_pred,
    "Linear Regression (PCA)": lr_pca_pred,
    "Decision Tree": tree_pred,
    "Random Forest (Default)": rf_pred,
    "Random Forest (Gridsearch)": rf_best_pred,
    "Gradient Boosting": gb_pred,
    "XGBoost": xgb_pred,
    "CatBoost": cat_pred,
    "Neural Network": nn_pred
}

# Print metrics table
r2_scores = {name: r2_score(y_test, pred) for name, pred in models.items()}
mae_scores = {name: mean_absolute_error(y_test, pred) for name, pred in models.items()}
mse_scores = {name: mean_squared_error(y_test, pred) for name, pred in models.items()}
rmse_scores = {name: np.sqrt(mse) for name, mse in mse_scores.items()}
metrics_df = pd.DataFrame({
    "R2 Score": r2_scores,
    "Mean Absolute Error": mae_scores,
    "Mean Squared Error": mse_scores,
    "Root Mean Squared Error": rmse_scores
})
print("--- Model Performance Comparison ---")
print(metrics_df.to_string())

# Generate and save Predicted vs. Actual plots for each model
for name, predictions in models.items():
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
    plt.title(f'Predicted vs. Actual Values - {name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plot_filename = f'predicted_vs_actual_{name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(plot_filename)
    plt.close() # Close the figure to free up memory
    print(f"Saved plot: {plot_filename}")

print("All plots generated and saved.")