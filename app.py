from flask import Flask, render_template,url_for, request, jsonify
import nltk
# from nltk.stem.lancaster import LancasterStemmer
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.framework import ops
from nltk.stem.lancaster import LancasterStemmer
from sklearn import *
stemmer = LancasterStemmer()



import numpy
import tflearn
import tensorflow
import random
import json

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle




with open("intents.json") as file:
	data = json.load(file)
with open("data.pickle","rb") as f:
	words, labels, training, output = pickle.load(f)
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)


ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
	
	bag = [0 for _ in range(len(words))]
	
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1

	return np.array(bag)

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')
@app.route("/index/cultibot")
def cultibot():
     return render_template("cultibot.html") #to send context to html

@app.route('/get')
def get_bot_response():
	# global seat_count
	message = request.args.get('msg')
	if message:
		message = message.lower()
		results = model.predict([bag_of_words(message,words)])[0]
		result_index = np.argmax(results)
		tag = labels[result_index]
		if results[result_index] > 0.7:
				for tg in data['intents']:
					if tg['tag'] == tag:
						responses = tg['responses']
				response = random.choice(responses)
		else:
			response = "I didn't quite get that, please try again."
		return str(response)
	return "Missing Data!"



@app.route("/index/plants")
def plants():
     return render_template("plants.html") #to send context to html

@app.route("/index/crops")
def crops():
     return render_template("crops.html") #to send context to html

@app.route("/index/fertilizer")
def fertilizer():
     return render_template("fertilizer.html") #to send context to html

@app.route("/index/about")
def about():
     return render_template("about.html") #to send context to html

#machine learning
	 
@app.route('/index/ml')
def ml():
    return render_template("ml.html")

model1=pickle.load(open('model.pkl','rb'))

@app.route('/index/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model1.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('ml.html',pred='Your Plants are in Danger.\nProbability of to be barren is {}'.format(output))
    else:
        return render_template('ml.html',pred='Your Plants are safe.\n Probability of to be fertilr is {}'.format(output))


if __name__ == "__main__":
	app.run()