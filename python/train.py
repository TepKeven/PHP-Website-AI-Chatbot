import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os

import nltk
from nltk.stem.lancaster import LancasterStemmer

# nltk.download('punkt')
stemmer = LancasterStemmer()

data = None
train_dir = "../train"

for root, directories, files in os.walk(train_dir):
    for file in files:
        with open(f"../train/{file}", encoding="utf-8") as fileObject:
            if data is None:
                data = json.load(fileObject)
            else:
                obj = json.load(fileObject)
                data["intents"].extend(obj["intents"])
        print(data)


try:
    with open("../model/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds) # [[],[]]
            print(docs_x)
            docs_y.append(intent["tag"])
            print(docs_y)

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] # Remove affix from words. Ex: Eats -> Eat or goes > go
    print(words)
    words = sorted(list(set(words)))
    print(words)

    labels = sorted(labels)
    print(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    print(out_empty)

    for x, doc in enumerate(docs_x): # doc_x = [[x,y,z], [a,b,c]]
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc] # wrds = [x,y,z]
        print("wrds")
        print(wrds)

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        print(docs_y[x])

        training.append(bag) # list of index keyword in words list represent by 0 or 1
        output.append(output_row) # list of 0 or 1 to represent if word belong to this tag
        print(training)
        print(output)


    training = numpy.array(training)
    output = numpy.array(output)

    print(training)
    print(output)

    with open("../model/data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


try:
    tensorflow.compat.v1.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])]) #length of total words in training questions
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.load("../model.tflearn")
except:

    tensorflow.compat.v1.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("../model/model.tflearn")


def bag_of_words(s, words):

    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    print(bag)
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        print([bag_of_words(inp, words)])
        results = model.predict([bag_of_words(inp, words)])
        print(results)
        results_index = numpy.argmax(results)
        print(results_index)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
        print(model.evaluate(training,output))


chat()