# Modified by Jin in Feb 2023
# Pipeline for running SIF on triplets data

import sys
import numpy as np
import json
sys.path.append('../src')
import data_io, params, SIF_embedding

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

    # load word weights
    # word2weight['str'] is the weight for the word 'str'
    print("Calculating words2weight.")
    word2weight = data_io.getWordWeight(weight, param)

    # weight4ind[i] is the weight for the i-th word
    print("Calculating weight4ind.")
    weight4ind = data_io.getWeight(words_file, word2weight)

    # load sentences
    print("Load samples...")
    x, m = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weight(x, m, weight4ind) # get word weights

    # get SIF embedding
    print("Calculate embedding...")
    embedding = SIF_embedding.SIF_embedding(We, x, w, parameters) # embedding[i,:] is the embedding for sentence i
    print(embedding.shape)



if __name__ == '__main__':
    generate_sif_embedding(data_dir=triplets_root_dir,
                           save_dir=save_sif_dir)
