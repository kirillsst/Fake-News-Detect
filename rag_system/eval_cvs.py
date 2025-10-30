import pandas as pd

# Charge ton CSV complet
df = pd.read_csv("data/processed/chunks.csv")

# Sélectionne 10 vrais + 10 faux (si possible)
true_samples = df[df['label'].str.upper() == 'TRUE'].sample(2, random_state=42)
fake_samples = df[df['label'].str.upper() == 'FAKE'].sample(2, random_state=42)

# Combine et mélange
sample_df = pd.concat([true_samples, fake_samples]).sample(frac=1, random_state=42)

# Sauvegarde
sample_df.to_csv("data/sample_texts.csv", index=False)
print(" Fichier d’échantillon créé : data/sample_texts.csv")