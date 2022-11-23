from sklearn.feature_extraction.text import CountVectorizer

document = ["The ones who are crazy enough",
			"to think that they can change",
			"the world are the ones who do"]

# Creating a Vectorizer Object
vectorizer = CountVectorizer()

vectorizer.fit(document)

# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)

# Encoding the Document
vector = vectorizer.transform(document)

# Summarizing the Encoded Texts

print("Encoded Document is: ")
print(vector.toarray())