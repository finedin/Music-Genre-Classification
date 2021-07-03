#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])


# In[31]:

small = tracks['set', 'subset'] == 'small'
small = small.to_frame()

# In[62]:

small_dictionary = small.T.to_dict('records')

# In[68]:

#small_dictionary[0][3]

# In[32]:

# find mp3 files genre's
genre_list = tracks["track", "genre_top"].to_frame()

# In[70]:

genre_dictionary = genre_list.T.to_dict('records')


# In[90]:


import os
from os import path
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import librosa.display


path = 'fma_small/'
wavpath = 'wav_file/'
entries = os.listdir(path)


k = 0
print(len(entries))
for first_entry in entries: # loop for fma_small directory
    print(str(k) + ". folder")
    if(k < 156):
        
        first_path = path + first_entry + '/'
        dynamic_path =  os.listdir(first_path) # update os path
        wavpath_new = wavpath + first_entry + '/'
        try:
            # Create target Directory
            os.mkdir(wavpath_new)
        except FileExistsError:
            print()
        
        for entry in dynamic_path: # loop for subfolders inside fma_small directory

            if(entry.split('.')[1] == 'mp3'):
                name = entry.split('.mp3')[0]

                # music name to int
                music_number = int(name)
                if(small_dictionary[0][music_number] == True):
                    music_genre = genre_dictionary[0][music_number]
                
                
                    second_path = first_path + entry

                    sound = AudioSegment.from_mp3(second_path)
                    destination = wavpath + music_genre + '/' + name + ".wav"

                    sound.export(destination, format="wav")
                
                    # wav files to png
                    #wav_name = wavpath_new + name + "jpg"
                    cmap = plt.get_cmap('inferno')
                    plt.figure(figsize=(8,8))
                    y, sr = librosa.load(destination)
                    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
                    plt.axis('off');
                    plt.savefig(wavpath + music_genre + '/' + name + ".png")
                    plt.clf()


            
    k = k + 1

