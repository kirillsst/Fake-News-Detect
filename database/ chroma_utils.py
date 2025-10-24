# import re
# import numpy as np

# def preprocess(text):
#     text = re.sub(r'<[^>]+>', '', str(text))
#     text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s]', '', text)
#     return text.lower().strip()

# def chunk_text(text, chunk_size=300):
#     words = text.split()
#     return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# def normalize_vector(v):
#     v = np.array(v)
#     norm = np.linalg.norm(v)
#     return (v / norm).tolist() if norm != 0 else v.tolist()
