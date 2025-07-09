import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() # lemmatizer is used to reduce the words to their base form (like running -> run)

intents = json.loads(open('intents.json').read())

words = [] # is for building the vocabulary.
classes = [] # stores the intents
documents = [] #  is for storing each pattern (as a list of words) and its associated intent.
ignoreLetters = ['?', '!', '.', ',']

# intents is the whole dictionary.
# intents['intents'] is the list.
#   [
#    {"tag": "greeting", "patterns": ["Hi", "Hello"], "responses": ["Hello!"]},
#    {"tag": "goodbye", "patterns": ["Bye"], "responses": ["Goodbye!"]}
#  ]

for intent in intents['intents']: # for each intent in the intents.json file
    for pattern in intent['patterns']: 
        wordList = nltk.word_tokenize(pattern) #tokenize the pattern (split the pattern into words like "Hello World!" -> ["Hello", "World", "!"])
        words.extend(wordList)
        documents.append((wordList, intent['tag'])) # append the wordList and the tag to the documents list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
# words has the tokenized words from the patterns (or wordList)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters] # lemmatize the words and remove the ignoreLetters
words = sorted(set(words)) # set(classes) removes any duplicate items from the classes list.
                             # sorted(...) sorts the resulting set in ascending (alphabetical) order and returns it as a list.

classes = sorted(set(classes)) # sorted and repeatations are removed 

pickle.dump(words, open('words.pkl', 'wb')) # wb stands for Write Binary mode
pickle.dump(classes, open('classes.pkl', 'wb'))

# - words now has the lemmatized form of the tokenized words (without ignoreLetters)
# - classes has the tags 
# - documents has the wordList and the tag (word list is the tokenized words from the patterns) it looks like this:
#     [
#         (['Hi'], 'greeting'),
#         (['Hello'], 'greeting'),
#         (['How', 'are', 'you', '?'], 'greeting'),
#         (['Bye'], 'goodbye'),
#         (['See', 'you', 'later'], 'goodbye')
#         # ...and so on for all patterns in all intents
#     ]

training = []
outputEmpty = [0] * len(classes) # For example, if you have 3 tags: ['greeting', 'goodbye', 'thanks'], then outputEmpty is [0, 0, 0].

for document in documents:
    bag = []
    wordPatterns = document[0] #  document[0] is the list of words (tokens) from a pattern.
                               #  document[1] is the tag (intent) for that pattern.
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns] # lemmatize the key fields (ie key of the dictionary documents)
    # wordPatterns has lemmatized form of words of the key fields of the dictionary documents
    # wordPatterns looks like ['how', 'are', 'you', '?'] (from (['How', 'are', 'you', '?'], 'greeting'))
    # Example:
    # Suppose words = ['are', 'bye', 'hello', 'hi', 'how', 'later', 'see', 'you']
    # and wordPatterns = ['how', 'are', 'you', '?'] (from the pattern "How are you?")

    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0) # one-hot encoding
    
    # The bag will be constructed as follows:
    # For example, suppose after processing, words is: ['are', 'bye', 'hello', 'hi', 'how', 'later', 'see', 'you']
    # For each word in words, if it appears in wordPatterns, append 1, else append 0.
    # So, the bag will be:
    #   'are'   -> 1 (in wordPatterns)
    #   'bye'   -> 0 (not in wordPatterns)
    #   'hello' -> 0 (not in wordPatterns)
    #   'hi'    -> 0 (not in wordPatterns)
    #   'how'   -> 1 (in wordPatterns)
    #   'later' -> 0 (not in wordPatterns)
    #   'see'   -> 0 (not in wordPatterns)
    #   'you'   -> 1 (in wordPatterns)
    # Resulting bag: [1, 0, 0, 0, 1, 0, 0, 1]   (ie, one-hot encoded)
    
    outputRow = list(outputEmpty) # This makes a copy of outputEmpty for each document (pattern).
    outputRow[classes.index(document[1])] = 1 #  sets the position corresponding to the tag (document[1] is the tag part) of the current document to 1.(one-hot encoding)
    # For example, if document[1] is 'goodbye' and classes is ['greeting', 'goodbye'], then classes.index('goodbye') is 1, so outputRow becomes [0, 1].
    training.append(bag + outputRow)
    # Combine the bag-of-words vector and the one-hot encoded outputRow into a single training example.
    # For example, if bag = [1, 0, 0, 0, 1, 0, 0, 1] (for the pattern "How are you?")
    # and outputRow = [1, 0] (for the tag 'greeting'),
    # then training.append(bag + outputRow) will add:
    #   [1, 0, 0, 0, 1, 0, 0, 1, 1, 0]
    # to the training list. This list contains both the input features (bag) and the output label (outputRow). 

random.shuffle(training) # to ensure the neural network doesn't learn any order dependancy
training = np.array(training)

trainX = training[:, :len(words)] # selects all rows and only the colums upto len(words) (ie, only the bag and not the outputRow)
trainY = training[:, len(words):] # selects all rows and only the colums from len(words) to end (ie, only outputRow excluding bag)

# Designing Neural Network Model (consist of 3 fully connected layers)
# --------------------------------------------------------------------
model = tf.keras.Sequential() # allow the layers to be added in sequence

# Layer 1 (128 nodes) 
model.add(tf.keras.layers.Dense(128, input_shape = (len(trainX[0]), ), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5)) # (dropout regularization) It will randomly set 50% of inputs to 0 at each update during training to reduce overfitting
# Layer 2 (64 nodes)
model.add(tf.keras.layers.Dense(64, activation='relu'))
# Layer 3 (no. of nodes equal to the no. of classes)
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True) # (Stochastic Gradient Descent optimizer: sgd is the optimizer that controls how your neural network learns during training, using the stochastic gradient descent algorithm with momentum and Nesterov acceleration.)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(trainX),np.array(trainY),epochs=200,batch_size=5,verbose=1) # training history
model.save('chatbot_model.h5',hist) # model is saved along with the training history
print('Executed')