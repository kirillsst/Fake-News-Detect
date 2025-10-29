import pandas as pd

data = [
    {
        "article_id": 0,
        "chunk_id": 0,
        "text": "donald trump just couldn t wish all americans a happy new year and leave it at that ... only trump",
        "label": "Fake"
    },
    {
        "article_id": 0,
        "chunk_id": 1,
        "text": "year . 2018 will be a great year for america ! ... who use the word hater in a new years wish ?",
        "label": "Fake"
    },
    {
        "article_id": 0,
        "chunk_id": 2,
        "text": "alan sandoval alansandoval13 december 31 , 2017 ... his filter have be break down",
        "label": "Fake"
    },
    {
        "article_id": 98,
        "chunk_id": 1,
        "text": "which be grapple with wild fire ... fema have so far approve more than 660 million in aid",
        "label": "True"
    },
    {
        "article_id": 98,
        "chunk_id": 2,
        "text": "say they fear will hurt the commonwealth ... rossello on monday order an official review of the death toll",
        "label": "True"
    }
]

df = pd.DataFrame(data)
df.to_csv("rag_system_chaima/eval_examples.csv", index=False, encoding="utf-8")
print("CSV créé avec succès !")
