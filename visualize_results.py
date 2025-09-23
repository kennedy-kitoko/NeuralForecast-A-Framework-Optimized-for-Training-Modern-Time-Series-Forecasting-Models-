# =========================================================
# 4) Prédictions
# =========================================================
preds = nf.predict(df=data).reset_index()

# On garde les dernières H (24h) vraies valeurs et prévisions
y_true = test.tail(H).reset_index(drop=True)
y_pred = preds.tail(H).reset_index(drop=True)

# Fusion simple par position
plot_df = pd.concat([y_true["y"], y_pred[["LSTM","TimesNet","NHITS"]]], axis=1)
plot_df.columns = ["True","LSTM","TimesNet","NHITS"]

# =========================================================
# 5) Visualisation
# =========================================================
plt.figure(figsize=(16,5))
plt.plot(plot_df.index, plot_df["True"], label="True", color="black", lw=2)
plt.plot(plot_df.index, plot_df["LSTM"], label="LSTM", color="dodgerblue")
plt.plot(plot_df.index, plot_df["TimesNet"], label="TimesNet", color="darkorange")
plt.plot(plot_df.index, plot_df["NHITS"], label="NHITS", color="green")

plt.title("Prévision 24 h – pas = 1h")
plt.ylabel("Température (°C)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/forecast_vs_actual.png", dpi=300)
plt.show()
