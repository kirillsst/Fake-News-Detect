from .rag_core import rag_analyze

if __name__ == "__main__":
    user_text = input("Entrez le texte à analyser :\n")
    result = rag_analyze(user_text)
    print("\n=== Résultat RAG ===")
    print(result)
