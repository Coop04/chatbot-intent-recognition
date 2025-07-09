import random
import json
import pickle
import numpy as np
import nltk

from textblob import TextBlob

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence): # to tokenize and lemmatize an input sentence
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence): # to one-hot encode the sentence words into a numpy array, bag
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0] 
    # model.predict gives a probability distribution over all classes (not a one-hot vector).
    # During training, outputRow is a one-hot encoded vector for the correct class.
    # During prediction, model.predict outputs probabilities for each class.
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key= lambda x: x[1], reverse = True) # sort the list based on the second item ie, r (in descending order)
    return_list = []
    for r in results: # results is of the form [index, probablity]
        return_list.append({'intent':classes[r[0]],  # r[0] represents the indices of all the intends (or in other words tags)
                            'probability':str(r[1])}) # r[1] is the predictes probability that satisfy the ERROR_THRESHOLD condetion
    
    # Example:
    # Suppose you have 3 intents: classes = ['greeting', 'goodbye', 'thanks']
    # After model prediction, you get:
    #   res = [0.1, 0.7, 0.2]  # Probabilities for each intent
    # Applying ERROR_THRESHOLD = 0.25:
    #   results = [(1, 0.7)]   # Only the 'goodbye' intent passes the threshold
    # When building the return_list:
    #   return_list = [{'intent': 'goodbye', 'probablity': '0.7'}]

    return return_list

def get_response(intents_list,intents_json): # intents_list is predicted intent(from model) intends_json is from the json file
    if not intents_list:
        return "I'm sorry, I didn't understand that."
    list_of_intents = intents_json['intents']
    # intents_list = [
    #     {'intent': 'greeting', 'probability': '0.85'},
    #     {'intent': 'goodbye', 'probability': '0.10'}
    # ]

    # intents_json = {
    #     "intents": [
    #         {
    #             "tag": "greeting",
    #             "responses": ["Hello!", "Hi there!", "Greetings!"]
    #         },
    #         {
    #             "tag": "goodbye",
    #             "responses": ["Goodbye!", "See you later!", "Bye!"]
    #         }
    #     ]
    # }
    tag = intents_list[0]['intent'] # since after sorting the predicted probablities in the descending order the heighest probablity will be at the 0th index
                                     # if it is greetings
    for i in list_of_intents: # The function will find the intent with tag 'greeting' and randomly return one of:
                              # "Hello!", "Hi there!", or "Greetings!"
        if i['tag']==tag:
            result = random.choice(i['responses'])
            break
    return result

print('Great! Bot is Running!')

def correct_spelling(text):
    return str(TextBlob(text).correct())

while True:
    message = input('').lower()
    message = correct_spelling(message)
    ints = predict_class(message)
    res = get_response(ints,intents)
    print(res)