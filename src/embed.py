
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_PATH = "data/processed/arxiv_filtered.parquet"

df = pd.read_parquet(DATA_PATH)
print(df.shape)

model = SentenceTransformer("all-MiniLM-L6-v2")

abstracts = df["abstract"].tolist()

embeddings = model.encode(abstracts, show_progress_bar=True)

print("Embeddings shape:", embeddings.shape)

np.save("data/processed/embeddings.npy", embeddings)
print("Saved embeddings to data/processed/embeddings.npy") 