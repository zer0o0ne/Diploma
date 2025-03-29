from collections import defaultdict
import itertools
import re
import subprocess
import json
from multiprocessing import Process, Queue
import os
import timeit
import ripser_count


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from stats_count import *
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42) # For reproducibility.

max_tokens_amount  = 128 # The number of tokens to which the tokenized text is truncated / padded.
    
layers_of_interest = [i for i in range(12)]  # Layers for which attention matrices and features on them are 
                                             # calculated. For calculating features on all layers, leave it be
                                             # [i for i in range(12)].

model_path = tokenizer_path = "bert-base-uncased"

ntokens_array = np.array([128] * 100)
batch_size = 1
DUMP_SIZE = 50

adj_filenames = [
    "/trinity/home/n.semenov/Diploma/tda4atd/diploma_llama_symmetry_250/attentions/1of5.npy",
    "/trinity/home/n.semenov/Diploma/tda4atd/diploma_llama_symmetry_250/attentions/2of5.npy",
    "/trinity/home/n.semenov/Diploma/tda4atd/diploma_llama_symmetry_250/attentions/3of5.npy",
    "/trinity/home/n.semenov/Diploma/tda4atd/diploma_llama_symmetry_250/attentions/4of5.npy",
    "/trinity/home/n.semenov/Diploma/tda4atd/diploma_llama_symmetry_250/attentions/5of5.npy",
]
# sorted by part number
adj_filenames

dim = 1
lower_bound = 1e-3


def subprocess_wrap(queue, function, args):
    queue.put(function(*args))
#     print("Putted in Queue")
    queue.close()
    exit()



def get_only_barcodes(adj_matricies, ntokens_array, dim, lower_bound):
    """Get barcodes from adj matricies for each layer, head"""
    barcodes = {}
    layers, heads = range(adj_matricies.shape[1]), range(adj_matricies.shape[2])
    for (layer, head) in itertools.product(layers, heads):
        matricies = adj_matricies[:, layer, head, :, :]
        barcodes[(layer, head)] = ripser_count.get_barcodes(matricies, ntokens_array, dim, lower_bound, (layer, head))
    return barcodes

def format_barcodes(barcodes):
    """Reformat barcodes to json-compatible format"""
    print(barcodes)
    return [{d: b[d] if isinstance(b[d], list) else b[d].tolist() for d in b} for b in barcodes]

def save_barcodes(barcodes, filename):
    """Save barcodes to file"""
    formatted_barcodes = defaultdict(dict)
    for layer, head in barcodes:
        formatted_barcodes[layer][head] = format_barcodes(barcodes[(layer, head)])
    json.dump(formatted_barcodes, open(filename, 'w'))
    
def unite_barcodes(barcodes, barcodes_part):
    """Unite 2 barcodes"""
    for (layer, head) in barcodes_part:
        barcodes[(layer, head)].extend(barcodes_part[(layer, head)])
    return barcodes

def split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits):
    splitted_ids = np.array_split(np.arange(ntokens.shape[0]), number_of_splits) 
    splitted = [(adj_matricies[ids], ntokens[ids]) for ids in splitted_ids]
    return splitted


queue = Queue()
number_of_splits = 2
barcodes_file = "small_gpt_web/barcodes/"
for i, filename in enumerate(tqdm(adj_filenames, desc='Calculating barcodes')):
    barcodes = defaultdict(list)
    adj_matricies = np.load(filename, allow_pickle=True) # samples X 
    print(f"Matricies loaded from: {filename}")
    ntokens = ntokens_array[i*batch_size*DUMP_SIZE : (i+1)*batch_size*DUMP_SIZE]
    splitted = split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits)
    for matricies, ntokens in tqdm(splitted, leave=False):
        p = Process(
            target=subprocess_wrap,
            args=(
                queue,
                get_only_barcodes,
                (matricies, ntokens, dim, lower_bound)
            )
        )
        p.start()
        barcodes_part = queue.get() # block until putted and get barcodes from the queue
#         print("Features got.")
        p.join() # release resources
#         print("The process is joined.")
        p.close() # releasing resources of ripser
#         print("The proccess is closed.")
        
        barcodes = unite_barcodes(barcodes, barcodes_part)
    part = filename.split('/')[-1].split('.')[0]
    save_barcodes(barcodes, barcodes_file + part + '.json')

ripser_feature_names=[
    'h0_s', 
    'h0_e',
    'h0_t_d', 
    'h0_n_d_m_t0.75',
    'h0_n_d_m_t0.5',
    'h0_n_d_l_t0.25',
    'h1_t_b',
    'h1_n_b_m_t0.25',
    'h1_n_b_l_t0.95', 
    'h1_n_b_l_t0.70',  
    'h1_s',
    'h1_e',
    'h1_v',
    'h1_nb'
]


output_dir = "small_gpt_web/"
adj_filenames = ["small_gpt_web/barcodes/" + f for f in os.listdir("small_gpt_web/barcodes")]
adj_filenames

def reformat_barcodes(barcodes):
    """Return barcodes to their original format"""
    formatted_barcodes = []
    for barcode in barcodes:
        formatted_barcode = {}
        for dim in barcode:
            formatted_barcode[int(dim)] = np.asarray(
                [(b, d) for b,d in barcode[dim]], dtype=[('birth', '<f4'), ('death', '<f4')]
            )
        formatted_barcodes.append(formatted_barcode)
    return formatted_barcodes

features_array = []

for filename in tqdm(adj_filenames, desc='Calculating ripser++ features'):
    barcodes = json.load(open(filename))
    print(f"Barcodes loaded from: {filename}", flush=True)
    features_part = []
    for layer in barcodes:
        features_layer = []
        for head in barcodes[layer]:
            ref_barcodes = reformat_barcodes(barcodes[layer][head])
            features = ripser_count.count_ripser_features(ref_barcodes, ripser_feature_names)
            features_layer.append(features)
        features_part.append(features_layer)
    features_array.append(np.asarray(features_part))

ripser_file = output_dir + 'features/llama_ripser.npy'
ripser_file

features = np.concatenate(features_array, axis=2)
features.shape

np.save(ripser_file, features)



