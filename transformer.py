import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
import string
import gensim
from gensim.models import Word2Vec
import numpy as np

def getPositionalEncoding(seq_len,d,n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P



dataset = 'I drink and I know things. When you play the game of thrones, you win or you die. The true enemy wont wait out the storm, He brings the storm.'

#lemmatize/lowercase/no punctuation
dataset = dataset.lower()
dataset = dataset.translate(str.maketrans('','',string.punctuation))
word_tokenizer = nltk.word_tokenize(dataset)
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokenizer]
lemmatized_data = ' '.join(lemmatized_words)

#split dataset into individual words
words = dataset.split()

#remove duplicates
unique = list(set(words))

print(len(unique))
print(unique)

#encoding -- giving unique numbers to each unique word in dataset

encoded_unique = [unique.index(word) for word in unique]

print(encoded_unique)

#create embeddings
sentence_to_use = 'when you play game of thrones' #--> [13,21,5,7,1,3]

model1 = gensim.models.Word2Vec(sentence_to_use, min_count=1, vector_size=512, window=5)

#hard coding it
embedding = []

#create positional encoding
positional = getPositionalEncoding(seq_len = 6, d=512)
print(positional)


#add positional and word embeddings together




#multi-head attention - only doing single for this
#Query, key, value matrices

