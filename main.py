import math
from collections import Counter

# Step 1: Read documents from file
with open("documents.txt", "r") as file:
    documents = file.readlines()

# Clean documents
documents = [doc.strip().lower() for doc in documents if doc.strip() != ""]

# Safety check
if not documents:
    print("Error: documents.txt is empty!")
    exit()

print("\nLoaded Documents:\n", documents)

# Tokenize documents
tokenized_docs = [doc.split() for doc in documents]

N = len(tokenized_docs)  # Total number of documents

# -----------------------------------
# Step 2: Calculate Term Frequency (TF)
# -----------------------------------
tf = []

for doc in tokenized_docs:
    word_count = Counter(doc)
    total_words = len(doc)
    tf_doc = {}

    for word in word_count:
        tf_doc[word] = word_count[word] / total_words  # normalized TF

    tf.append(tf_doc)

# -----------------------------------
# Step 3: Calculate Document Frequency (DF)
# -----------------------------------
df = {}

for doc in tokenized_docs:
    unique_words = set(doc)
    for word in unique_words:
        df[word] = df.get(word, 0) + 1

# -----------------------------------
# Step 4: Calculate Inverse Document Frequency (IDF)
# -----------------------------------
idf = {}

for word in df:
    idf[word] = math.log10(N / df[word])  # log base 10

# -----------------------------------
# Step 5: Calculate TF-IDF
# -----------------------------------
tfidf = []

for doc in tf:
    tfidf_doc = {}
    for word in doc:
        tfidf_doc[word] = doc[word] * idf[word]
    tfidf.append(tfidf_doc)

# -----------------------------------
# Step 6: Display TF-IDF Results
# -----------------------------------
print("\nTF-IDF Values:\n")

for i, doc in enumerate(tfidf):
    print(f"Document {i+1}:")
    for word, value in doc.items():
        print(f"{word}: {value:.4f}")
    print()

# -----------------------------------
# Step 7: Show Top Words per Document
# -----------------------------------
print("\nTop Words per Document:\n")

for i, doc in enumerate(tfidf):
    print(f"Document {i+1}:")
    sorted_words = sorted(doc.items(), key=lambda x: x[1], reverse=True)
    
    for word, value in sorted_words[:3]:
        print(f"{word}: {value:.4f}")
    print()