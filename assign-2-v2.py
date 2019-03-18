###
#   Author:         Jesse Wheeler
#   Student ID:     101075970
###

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import time
import re

###
#   If changing the number of ingredients to use, be sure to set new_csv to
#   True on the first re-run
#
config = {
            'ingredients': 'all',
            'epochs': 15,
            'batch_size': 150,
            'layer_activation': 'relu',
            'output_activation': 'softmax',
            'optimizer': 'adam',
            'new_csv': False,
            'results': True
        }

###
#   Initial data file reads.
#
print("Reading JSON data files.")
df = pd.read_json('./train.json')
test = pd.read_json('./test.json')
ids = test['id']

###
#   LabelEncode cuisine in DataFrame.
#
le = LabelEncoder()
df['cuisine'] = le.fit_transform(df['cuisine'])

###
#   Transform all used ingredients into a single dimensional array
#   and sort them by number of occurrences.
#
print("Sorting and selecting most used ingredients (using top " + str(config['ingredients']) + ").")
ingr = np.concatenate(df['ingredients']).ravel()
newingr = []
def fix_ingr(ing):
    ing = re.findall('[0-9a-zA-Z%-]', ing)
    ing = ''.join(wrd for wrd in ing)
    return ing.lower()

for x in ingr:
    y = x.split()
    for z in y:
        z = fix_ingr(z)
        if not z in newingr:
            newingr = newingr + [z]
            
if config['ingredients'] == 'all':
    config['ingredients'] = len(newingr)

###
#   Modify DataFrame data with one hot encoding of ingredients
#
def one_hot_encoding(row):
    try:
        ohe = np.zeros(config['ingredients'],)
        ingred = row['ingredients']
        row.drop(labels=['ingredients', 'id'], inplace=True)
        for ing in ingred:
            y = ing.split()
            for z in y:
                z = fix_ingr(z)
                if z in newingr:
                    ohe[newingr.index(z)] = 1
        return np.concatenate((row, ohe))
    except Exception as e:
        print(e)

###
#   Save one hot encoded CSV files for quicker access
#
def save_csv(train=True):
    vals = []
    if train:
        rows = df.iterrows()
        file = "./new_train.csv"
        print("Performing One Hot Encoding for training data.")
    else:
        rows = test.iterrows()
        print("Performing One Hot Encoding for testing data.")
        file = "./new_test.csv"
        
    for i, row in rows:
        if (train and i % 2500 == 0) or (train and i == len(df.index) - 1):
            print("Training Encoded: " + str(i) + "/" + str(len(df.index) - 1))
        elif train == False and i % 2500 == 0 or i == len(df.index) - 1:
            print("Testing Encoded: " + str(i) + "/" + str(len(test.index) - 1))
        vals.append(one_hot_encoding(row))
    
    print("Creating new DataFrame (please be patient).")
    newdf = pd.DataFrame(vals)
    print("Saving to CSV (please be patient, file sizes can be > 500MB).")
    newdf.to_csv(file, index=False)

if config['new_csv']:
    save_csv()
    save_csv(False)
    
###
#   Train the model
#
def make_model():
    model = Sequential()
    model.add(Dense(config['ingredients'], input_dim=config['ingredients'], activation=config['layer_activation']))
    model.add(Dense(len(le.classes_), activation=config['output_activation']))
    model.compile(loss='categorical_crossentropy', optimizer=config['optimizer'], metrics=['accuracy'])
    return model

def get_results():
    print("Firing the laz0rs.");
    estimator = KerasClassifier(build_fn=make_model)
    estimator.fit(Xtrain, ytrain, epochs=config['epochs'], batch_size=config['batch_size'], verbose=1)
    print("Classifying / predicting results.")
    ynew = estimator.predict(test.values)
    ynew = np.array(ynew, dtype=np.int32)
    estimates = le.inverse_transform(ynew)
    result = {'id': ids, 'cuisine': estimates}
    nr = pd.DataFrame(data=result)
    nr.to_csv(str(time.time()) + ".csv", index=False)
    
if config['results']:    
    print("Reading CSV data files.")
    df = pd.read_csv('./new_train.csv')
    test = pd.read_csv('./new_test.csv')
    
    Xtrain = df.values[:,1:]
    ytrain = df.values[:,0]
    
    get_results()
    print("Laz0rs have been fired.")