# =========================================================
# 0) Libs
# =========================================================
import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from neuralforecast.core import NeuralForecast
from neuralforecast.models import LSTM, TimesNet, NHITS
import pytorch_lightning as pl, torch

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# Seed
SEED = 1
pl.seed_everything(SEED, workers=True)
torch.manual_seed(SEED); np.random.seed(SEED)

# Détection GPU pour Lightning
ACCEL = "gpu" if torch.cuda.is_available() else "cpu"
DEVICES = 1

TRAINER_KW = dict(
    accelerator=ACCEL,
    devices=DEVICES,
    enable_progress_bar=True,
    inference_mode=True,
    logger=False
)

# =========================================================
# 1) Données
#    - on resample à 10 min pour coller à ton pipeline
#    - on utilise T (degC) comme cible y
#    - toutes les autres colonnes numériques = exogènes historiques
# =========================================================
raw = pd.read_csv("weather.csv")
raw["date"] = pd.to_datetime(raw["date"])
raw = raw.set_index("date").sort_index()

# resample 10 minutes + interpolation temporelle
df10 = raw.resample("10min").mean().interpolate("time")

# features numériques
num_cols = df10.select_dtypes(include=["number"]).columns.tolist()
assert "T (degC)" in num_cols, "Colonne cible 'T (degC)' introuvable."
y_col = "T (degC)"

# exogènes historiques = toutes les numériques sauf la cible
hist_cols = [c for c in num_cols if c != y_col]

# cadre NeuralForecast
data = pd.DataFrame({
    "unique_id": "weather",
    "ds": df10.index,
    "y": df10[y_col]
}).reset_index(drop=True)

# on rajoute les exogènes au même DataFrame (obligatoire pour NF)
data = pd.concat([data, df10[hist_cols].reset_index(drop=True)], axis=1)

# Split
H = 24 * 6             # 24 h à 10 min
TEST_SIZE = H * 7      # 1 semaine pour test
train = data.iloc[:-TEST_SIZE].copy()
test  = data.iloc[-TEST_SIZE:].copy()

# =========================================================
# 2) Modèles
#    - LSTM & NHITS : avec exogènes historiques
#    - TimesNet : sans hist_exog (limitation du modèle)
# =========================================================
# =========================================================
# 2) Modèles
# =========================================================
common = dict(
    h=H,
    input_size=7*24*6,          # ~1 semaine d'historique
    step_size=H,
    learning_rate=1e-3,
    batch_size=32,
    valid_batch_size=64,
    windows_batch_size=128,
    val_check_steps=50,
    early_stop_patience_steps=400,
    scaler_type="robust",
    random_seed=SEED,
    **TRAINER_KW                # ✅ on déplie le dict
)

mdl_lstm = LSTM(**common, hist_exog_list=hist_cols, max_steps=600)
mdl_timesnet = TimesNet(**common, max_steps=1000)  # ⚠️ sans hist_exog
mdl_nhits = NHITS(**common, hist_exog_list=hist_cols, max_steps=1000)

models = [mdl_lstm, mdl_timesnet, mdl_nhits]
nf = NeuralForecast(models=models, freq="10min")

nf.fit(df=train, val_size=3*H)  # ✅ fonctionne

# =========================================================
# 3) Entraînement
#    - val_size = 3 horizons pour early stopping
# =========================================================
nf = NeuralForecast(models=models, freq="10min")
nf.fit(df=train, val_size=3*H)

# =========================================================
# 4) Prédictions H pas (24h) et alignement
# =========================================================
preds = nf.predict(df=data).reset_index()

cols_models = [c for c in preds.columns if c in ["LSTM","TimesNet","NHITS"]]

# Si 'ds' est présent dans les colonnes → on aligne sur les vraies dates
if "ds" in preds.columns:
    y_true = test.tail(H)[["ds", "y"]].reset_index(drop=True)
    y_pred = preds.tail(H)[["ds"] + cols_models].reset_index(drop=True)
    plot_df = y_true.merge(y_pred, on="ds", how="inner")
else:
    # sinon fallback par position
    print("⚠️ Pas de colonne 'ds' dans preds → alignement par index")
    y_true = test.tail(H).reset_index(drop=True)
    y_pred = preds.tail(H).reset_index(drop=True)
    plot_df = pd.concat([y_true["y"], y_pred[cols_models]], axis=1)
    plot_df.columns = ["y"] + cols_models

# Vérif de sécurité
if len(plot_df) != H:
    print(f"⚠️ Alignement: attendu {H}, obtenu {len(plot_df)}")


# =========================================================
# 5) Métriques (RMSE de façon compatible)
# =========================================================
metrics = []
for m in ["LSTM","TimesNet","NHITS"]:
    if m in plot_df.columns:
        mae  = mean_absolute_error(plot_df["y"], plot_df[m])
        rmse = np.sqrt(mean_squared_error(plot_df["y"], plot_df[m]))  # robuste
        metrics.append({"model": m, "MAE": mae, "RMSE": rmse})

metrics_df = pd.DataFrame(metrics).sort_values("RMSE")
metrics_df.to_csv("results/metrics1.csv", index=False)
print("\n📊 Metrics (24h):\n", metrics_df)

# =========================================================
# 6) Visualisation
# =========================================================
plt.figure(figsize=(14,4))
plt.plot(plot_df["ds"], plot_df["y"], label="True", color="black", lw=2)
for m in ["LSTM","TimesNet","NHITS"]:
    if m in plot_df.columns:
        plt.plot(plot_df["ds"], plot_df[m], label=m)
plt.xticks(rotation=45); plt.grid(alpha=.3)
plt.title("Prévision 24 h – pas = 10 min")
plt.ylabel("Température (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("results/forecast1.png", dpi=300)
plt.show()

print("\n✅ Fichiers créés : results/metrics1.csv  &  results/forecast1.png")

# (Optionnel) Sauvegarde des modèles pour réutilisation ultérieure
# nf.save(path="./checkpoints/weather_run/", overwrite=True, save_dataset=True)
