import nltk
for p in ["punkt", "wordnet", "omw-1.4", "stopwords"]:
    nltk.download(p)
print("NLTK data OK")
