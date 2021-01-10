#!/usr/bin/env python
# coding: utf-8

# In[17]:


# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""    
    
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True)
    
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat,delta)) 
    return combined
#    
if __name__ == "__main__":
     print ("In main, Call extract_features(audio,signal_rate) as parameters")
     


# In[1]:


import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)


# In[21]:


#train_models.py

import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")


#path to training data
source   = "train_dataset\\"   

#path where training speakers will be saved
dest = "train_models\\"

train_file = "train.txt"        


# 훈련시킬 파일 위치
file_paths = open(train_file,'r')

count = 1

# Extracting features for each speaker (5 files per speakers)
# 한명당 5개의 파일
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print (path)
    
    # read the audio
    sr,audio = read(source + path)
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
    if count == 5:    
        gmm = mixture.GaussianMixture(n_components = 16, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        
        # dumping the trained gaussian model
        picklefile = path.split("\\")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape) 
        features = np.asarray(())
        count = 0
    count = count + 1
    


# In[6]:


#test_gender.py
import os
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time


#path to training data
source   = "test_dataset\\"   

modelpath = "train_models\\"

test_file = "test.txt"        

file_paths = open(test_file,'r')


gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian gender Models
models    = [cPickle.load(open(fname,'rb'), encoding='utf-8') for fname in gmm_files]
speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname 
              in gmm_files]

# Read the test directory and get the list of test audio files 

output_speakers = []

for path in file_paths:   
    path = path.strip()   
    print (path)
    sr,audio = read(source + path)
    vector   = extract_features(audio,sr)
    
    log_likelihood = np.zeros(len(models)) 
    
    for i in range(len(models)):
        gmm    = models[i]         #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    output_speakers.append(speakers[winner])


# In[24]:


files = open("subtitles.srt",'r', encoding='UTF8')
lines = files.readlines()
files.close()

with open("subtitles_added.srt", "w") as f:
    count = 0
    for i in range(len(lines)):
        if((i-2)%4 == 0):
            lines[i] = "( " + output_speakers[count] + " ) " + lines[i]
            count+=1
        f.write(lines[i])
    f.close()




