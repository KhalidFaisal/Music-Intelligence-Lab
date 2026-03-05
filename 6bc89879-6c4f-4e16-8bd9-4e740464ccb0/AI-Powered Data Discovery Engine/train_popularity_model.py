
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Zerve design system ────────────────────────────────────────────────────────
BG_M     = "#1D1D20"
TEXT_M   = "#fbfbff"
SUBTXT_M = "#909094"
PALETTE_M = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF",
             "#1F77B4","#9467BD","#8C564B","#C49C94","#E377C2",
             "#F7B6D2","#DBDB8D"]

# ── 1. Feature engineering ─────────────────────────────────────────────────────
model_df = songs_df.copy()

le_genre = LabelEncoder()
le_mode  = LabelEncoder()
model_df["genre_enc"] = le_genre.fit_transform(model_df["genre"])
model_df["mode_enc"]  = le_mode.fit_transform(model_df["mode"])

MODEL_FEATURES = [
    "tempo", "energy", "danceability", "valence",
    "acousticness", "loudness_db", "speechiness", "instrumentalness",
    "duration_sec", "year", "genre_enc", "mode_enc",
]
TARGET = "popularity"

X = model_df[MODEL_FEATURES].values
y = model_df[TARGET].values.astype(float)

# ── 2. Train / test split — stratified on popularity quintile ─────────────────
pop_quintile = pd.qcut(y, 5, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=pop_quintile
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── 3. GradientBoostingRegressor (sklearn) + 5-fold CV ────────────────────────
gbm = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    min_samples_leaf=20,
    max_features=0.8,
    random_state=42,
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = np.sqrt(-cross_val_score(
    gbm, X_train, y_train,
    scoring="neg_mean_squared_error", cv=kf, n_jobs=-1
))
print(f"\n5-Fold CV RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")

# ── 4. Final fit ───────────────────────────────────────────────────────────────
gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)
gbm_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
gbm_r2   = float(r2_score(y_test, y_pred))
print(f"\nTest RMSE : {gbm_rmse:.3f}")
print(f"Test R²   : {gbm_r2:.4f}")

# ── 5. Feature importance chart ───────────────────────────────────────────────
importances  = gbm.feature_importances_
feat_order   = np.argsort(importances)
sorted_feats = [MODEL_FEATURES[i] for i in feat_order]
sorted_imps  = importances[feat_order] * 100

fig_importance, ax_imp = plt.subplots(figsize=(10, 6))
fig_importance.patch.set_facecolor(BG_M)
ax_imp.set_facecolor(BG_M)

bars = ax_imp.barh(
    sorted_feats, sorted_imps,
    color=PALETTE_M[:len(sorted_feats)], edgecolor="none", height=0.65
)
for bar, val in zip(bars, sorted_imps):
    ax_imp.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color=TEXT_M, fontsize=9)

ax_imp.set_xlabel("Feature Importance (%)", color=SUBTXT_M, fontsize=11)
ax_imp.set_title("Gradient Boosting — Feature Importance for Song Popularity",
                 color=TEXT_M, fontsize=13, fontweight="bold", pad=14)
ax_imp.tick_params(colors=TEXT_M, labelsize=10)
ax_imp.xaxis.label.set_color(SUBTXT_M)
for spine in ax_imp.spines.values():
    spine.set_edgecolor("#333337")
ax_imp.set_xlim(0, sorted_imps.max() * 1.22)
ax_imp.grid(axis="x", color="#333337", linewidth=0.5, alpha=0.7)
plt.tight_layout()
print("\n✅ Feature importance chart rendered.")

# ── 6. Predicted vs Actual scatter ────────────────────────────────────────────
fig_scatter, ax_sc = plt.subplots(figsize=(7, 6))
fig_scatter.patch.set_facecolor(BG_M)
ax_sc.set_facecolor(BG_M)

ax_sc.scatter(y_test, y_pred, alpha=0.22, s=12,
              color="#A1C9F4", edgecolors="none", label="Songs")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax_sc.plot(lims, lims, color="#ffd400", linewidth=1.5, linestyle="--", label="Perfect fit")
ax_sc.set_xlabel("Actual Popularity", color=SUBTXT_M, fontsize=11)
ax_sc.set_ylabel("Predicted Popularity", color=SUBTXT_M, fontsize=11)
ax_sc.set_title(
    f"Actual vs Predicted  |  R² = {gbm_r2:.3f}  |  RMSE = {gbm_rmse:.2f}",
    color=TEXT_M, fontsize=12, fontweight="bold", pad=12)
ax_sc.tick_params(colors=TEXT_M, labelsize=9)
for spine in ax_sc.spines.values():
    spine.set_edgecolor("#333337")
ax_sc.legend(facecolor="#29292e", edgecolor="#333337", labelcolor=TEXT_M, fontsize=9)
plt.tight_layout()
print("✅ Predicted vs Actual scatter rendered.")

# ── 7. What-if analysis ────────────────────────────────────────────────────────
CONTINUOUS_FEATS = [
    "tempo", "energy", "danceability", "valence",
    "acousticness", "loudness_db", "speechiness", "instrumentalness", "duration_sec"
]

median_row = {f: float(np.median(model_df[f])) for f in MODEL_FEATURES}
baseline_val = float(gbm.predict(np.array([[median_row[f] for f in MODEL_FEATURES]]))[0])

whatif_rows = []
for feat in CONTINUOUS_FEATS:
    std_val = float(model_df[feat].std())
    # +1 std
    row_plus = dict(median_row)
    row_plus[feat] += std_val
    lift_plus = float(gbm.predict(np.array([[row_plus[f] for f in MODEL_FEATURES]]))[0]) - baseline_val
    # -1 std
    row_minus = dict(median_row)
    row_minus[feat] -= std_val
    lift_minus = float(gbm.predict(np.array([[row_minus[f] for f in MODEL_FEATURES]]))[0]) - baseline_val

    if lift_plus >= lift_minus:
        best_dir, best_lift, best_val = "+1 std", lift_plus, median_row[feat] + std_val
    else:
        best_dir, best_lift, best_val = "-1 std", lift_minus, median_row[feat] - std_val

    whatif_rows.append({
        "feature": feat, "direction": best_dir,
        "std_change": std_val, "new_value": best_val, "lift": best_lift
    })

whatif_results_df = (pd.DataFrame(whatif_rows)
                       .sort_values("lift", ascending=False)
                       .reset_index(drop=True))

print("\n" + "="*64)
print("  WHAT-IF ANALYSIS — Top 3 Actionable Levers")
print(f"  Baseline popularity (median song): {baseline_val:.1f}")
print("="*64)
for _, row in whatif_results_df.head(3).iterrows():
    action = "Increase" if row["direction"] == "+1 std" else "Decrease"
    print(f"\n  🎯  {row['feature'].upper()}")
    print(f"     Action   : {action} by ~1 std dev  (±{row['std_change']:.3f})")
    print(f"     New value: {row['new_value']:.3f}")
    print(f"     Expected : +{row['lift']:.2f} popularity points")
print("="*64)

# Store key outputs
gbm_model        = gbm
gbm_features     = MODEL_FEATURES
gbm_test_rmse    = gbm_rmse
gbm_test_r2      = gbm_r2
gbm_whatif_df    = whatif_results_df
gbm_le_genre     = le_genre
gbm_le_mode      = le_mode
