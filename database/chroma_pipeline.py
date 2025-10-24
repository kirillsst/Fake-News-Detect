# import pandas as pd
# from .chroma_setup import get_chroma_collection
# from .chroma_insert import insert_chunk
# from  .chroma_utils import preprocess, chunk_text

# # CONFIG
# CSV_PATH = "data/Fake.csv"
# COLLECTION_NAME = "fake_news_collection"

# # Récupération de la collection
# collection = get_chroma_collection(collection_name=COLLECTION_NAME)

# # Chargement CSV
# df = pd.read_csv(CSV_PATH)
# print(f" {len(df)} articles chargés")

# # Traitement articles
# for idx, row in df.iterrows():
#     text = preprocess(row["text"])
#     chunks = chunk_text(text)

#     for i, chunk in enumerate(chunks):
#         chunk_id = f"{idx}_{i}"
#         metadata = {
#             "title": row.get("title", "unknown"),
#             "source": row.get("source", "unknown"),
#             "date": row.get("date", "unknown"),
#             "label": row.get("label", "unknown"),
#             "chunk_index": i
#         }
#         insert_chunk(chunk_id, chunk, metadata, collection)

# print(" Pipeline terminée !")
