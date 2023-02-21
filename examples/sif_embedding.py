# Modified by Jin in Feb 2023
# Pipeline for running SIF on triplets data

import sys
import os
import numpy as np
import json
sys.path.append('../src')
import data_io, params, SIF_embedding
import shutil
from string import punctuation, digits
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = stopwords.words("english")
import emoji
import translators as ts
import translators.server as tss

#############################################################
#  Input paths and parameters (no need to change these)
#############################################################
"""
wordfile: word vector file, can be downloaded from GloVe website
weightfile: each line is a word and its frequency
weightpara: The parameter in the SIF weighting scheme, 
            usually in the range [3e-5, 3e-3]
rmpc: number of principal components to remove in SIF weighting scheme

"""
# wordfile = '../data/glove.840B.300d.txt'
# sentences = ['this is an example sentence',
#              'this is another sentence that is slightly longer']
# (words, We) = data_io.getWordmap(wordfile)

rmpc = 1
weightpara = 1e-3

# set parameters
params = params.params()
params.rmpc = rmpc

weightfile = '../auxiliary_data/enwiki_vocab_min200.txt'
words_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/words.json"
We_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/We.npy"

# Load pre-calculated words and We (provided by SAFE)
with open(words_path, 'r') as f:
  words = json.load(f)

print("words.json loaded")

We = np.load(We_path)
print("We loaded.")

# Simple sentences for initial test
sentences = ['Willy is a cat',
             'Willy is a very cute and handsome cat']


#############################################################
#  My data paths
#############################################################
# TODO: this part is the paths for debugging data
# 381 sets from Jan for debugging
triplets_root_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/" \
                    "jhuang24/safe_data/jan01_jan02_2023_triplets"

save_sif_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
               "safe_data/jan01_jan02_2023_triplets_sif"


#############################################################
#  Process Data
#############################################################
# text preprocessing
# Note: code is copied from SAFE
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, ' ', text)

    text = text.strip()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]

    stemmer= PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    lemmatizer=WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


# Remove Emojis in texts
def give_emoji_free_text(text):
    # allchars = [str for str in text.decode('utf-8')]
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]
    # clean_text = ' '.join([str for str in text.decode('utf-8').split() if not any(i in str for i in emoji_list)])
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text



def generate_sif_embedding(data_dir,
                           save_dir,
                           weight=weightfile,
                           param=weightpara,
                           words_file=words,
                           parameters=params):
    """

    :param data_dir:
    :param save_dir:
    :param weight:
    :param param:
    :param words_file:
    :param parameters:
    :return:
    """
    # Loop thru all the dir and files
    for path, subdirs, files in os.walk(data_dir):
        for name in files:
            # Check the subfolder, and make dir for save path if it does not exist
            current_folder = path.split("/")[-1]
            target_save_dir = os.path.join(save_dir, current_folder)

            if not os.path.isdir(target_save_dir):
                os.mkdir(target_save_dir)
                print("Making directory: ", target_save_dir)

            # Check whether this is a json file
            if name.endswith('.json'):
                # Load json file
                with open(os.path.join(path, name), 'r') as f:
                    json_data = json.load(f)

                #####################################################
                #  Pre-processing
                #####################################################
                sentences = json_data["description"]
                # print("Original:", sentences)

                # Remove Emojis first
                sentences = give_emoji_free_text(sentences)

                if len(sentences) == 0:
                    print("Skipping file: ", name)
                    continue

                # Translate Russian to English
                from_language, to_language = 'ru', 'en'
                sentences = tss.google(sentences, from_language, to_language)
                # result is a whole string
                # print("English translation:", sentences)


                # clean up texts
                sentences = clean_text(sentences)
                # print("Cleaned sentences:", sentences)

                #####################################################
                #  Word2Vector
                #####################################################
                # Load word weights
                # word2weight['str'] is the weight for the word 'str'
                # print("Calculating words2weight.")
                word2weight = data_io.getWordWeight(weight, param)

                # weight4ind[i] is the weight for the i-th word
                # print("Calculating weight4ind.")
                weight4ind = data_io.getWeight(words_file, word2weight)

                # Calculate weight and index
                # print("Load samples...")
                # x is the array of word indices, m is the binary mask indicating
                # whether there is a word in that location
                try:
                    x, m = data_io.sentences2idx(sentences, words)
                    w = data_io.seq2weight(x, m, weight4ind) # get word weights
                except:
                    print("Skipping file: ", name)

                # Get SIF embedding
                # print("Calculate embedding...")
                # embedding[i,:] is the embedding for sentence i
                embedding = SIF_embedding.SIF_embedding(We, x, w, parameters)
                print(embedding.shape) # [nb_words, 300]

                # Save embedding into npy
                file_name = name.split(".")[0] + "_sif_embedding.npy"
                file_save_path = os.path.join(save_sif_dir, current_folder, file_name)
                np.save(file_save_path, embedding)
                print("Saving one embedding to: ", file_save_path)


if __name__ == '__main__':
    generate_sif_embedding(data_dir=triplets_root_dir,
                           save_dir=save_sif_dir)
