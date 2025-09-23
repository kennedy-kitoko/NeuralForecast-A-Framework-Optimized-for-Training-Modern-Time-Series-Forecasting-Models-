# Sauvegarde complète du run
nf.save(
    path="./checkpoints/weather_run/",
    overwrite=True,        # écrase si dossier déjà existant
    save_dataset=True      # sauve aussi les data prétraitées
)

print("✅ Modèle exporté dans ./checkpoints/weather_run/")
