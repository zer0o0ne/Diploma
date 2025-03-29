import re
import os
import json
import timeit
import torch
# import ripser_count
import numpy as np
import pandas as pd
from time import time
from math import ceil
import itertools
from collections import defaultdict
from tqdm.auto import tqdm
from multiprocessing import Pool
from multiprocessing import Process, Queue
from transformers import AutoTokenizer, AutoModelForCausalLM

from stats_count import *

class Logger:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        
    def __call__(self, text, filename = "log.txt", verbose = True, debug=True):
        if debug:
            text = str(text)
            if verbose:
                print(text)
            with open(self.path + "/" + filename, "a") as file:
                file.write(text + "\n")

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

def subprocess_wrap(queue, function, args):
    queue.put(function(*args))
#     print("Putted in Queue")
    queue.close()
    exit()

# def get_only_barcodes(adj_matricies, ntokens_array, dim, lower_bound):
#     """Get barcodes from adj matricies for each layer, head"""
#     barcodes = {}
#     layers, heads = range(adj_matricies.shape[1]), range(adj_matricies.shape[2])
#     for (layer, head) in itertools.product(layers, heads):
#         matricies = adj_matricies[:, layer, head, :, :]
#         barcodes[(layer, head)] = ripser_count.get_barcodes(matricies, ntokens_array, dim, lower_bound, (layer, head))
#     return barcodes

def format_barcodes(barcodes):
    """Reformat barcodes to json-compatible format"""
    return [{d: b[d].tolist() for d in b} for b in barcodes]

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

def split_matricies_and_lengths_ripser(adj_matricies, ntokens, number_of_splits):
    splitted_ids = np.array_split(np.arange(ntokens.shape[0]), number_of_splits) 
    splitted = [(adj_matricies[ids], ntokens[ids]) for ids in splitted_ids]
    return splitted

def grab_attention_weights(model, tokenizer, sentences, MAX_LEN, need_symmetry):
    inputs = tokenizer.batch_encode_plus([text_preprocessing(s) for s in sentences],
                                       return_tensors='pt',
                                       add_special_tokens=True,
                                       max_length=MAX_LEN,             # Max length to truncate/pad
                                       pad_to_max_length=True,         # Pad sentence to max length)
                                       truncation=True
                                      )
    device = list(model.children())[0].device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)
    attention = model(**inputs, output_attentions=True).attentions
    # layer X sample X head X n_token X n_token
    attention = np.asarray([layer.cpu().detach().float().numpy() for layer in attention], dtype=np.float16)
    for i in range(attention.shape[-1]):
        attention[:, :, :, i, i] = 0

    if need_symmetry:
        attention = attention + attention.transpose((0, 1, 2, 4, 3))
    
    return attention

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # text = f"You must answer the following question: {text}. Your answer must contain 'Yes' or 'No' and nothing else"
    # print(text)
    return text

def load_pinocchio(num_samples):
    lines = []
    with open('dataset.jsonl') as f:
        lines = f.read().splitlines()

    line_dicts = [json.loads(line) for line in lines]
    data = pd.DataFrame(line_dicts)
    data = data[data.domain != "Multilingual"]
    data = data[data.answer != "Not Sure Enough"]
    data = data.sample(n = num_samples, random_state = 42)
    return data

def function_for_v(list_of_v_degrees_of_graph):
    return sum(map(lambda x: np.sqrt(x*x), list_of_v_degrees_of_graph))

def split_matricies_and_lengths(adj_matricies, ntokens_array, num_of_workers):
    splitted_adj_matricies = np.array_split(adj_matricies, num_of_workers)
    splitted_ntokens = np.array_split(ntokens_array, num_of_workers)
    assert all([len(m)==len(n) for m, n in zip(splitted_adj_matricies, splitted_ntokens)]), "Split is not valid!"
    return zip(splitted_adj_matricies, splitted_ntokens)

# CONFIGURATIONS
np.random.seed(42)
max_tokens_amount  = 196 # The number of tokens to which the tokenized text is truncated / padded.
max_tokens_answer = 6
stats_cap          = 500 # Max value that the feature can take. Is NOT applicable to Betty numbers.
stats_name = "s_e_v_c_b0b1" # The set of topological features that will be count (see explanation below)
thresholds_array = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75] # The set of thresholds
num_of_workers = 4


model_name = "Undi95/Meta-Llama-3-8B-Instruct-hf"
need_symmetry = True
num_samples = 250
log_path = "diploma_llama_symmetry_250"

need_model_answer = False
need_attentions = True
need_topological = True
# need_ripser = True

batch_size = 1 # batch size
DUMP_SIZE = 50 # number of batches to be dumped
# END OF CONFIGURATIONS

thrs = len(thresholds_array)                           # ("t" in the paper)
data = load_pinocchio(num_samples)
number_of_batches = ceil(len(data['claim']) / batch_size)
batched_sentences = np.array_split(data['claim'].values, number_of_batches)
number_of_files = ceil(number_of_batches / DUMP_SIZE)

model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             device_map = "auto", torch_dtype = torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
log = Logger(log_path)

if need_model_answer:
    model_answers = []
    is_right = []
    for claim, true_answer in tqdm(zip(data["claim"].values, data["answer"].values)):
        inputs = tokenizer(text_preprocessing(claim), return_tensors = "pt")
        device = list(model.children())[0].device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)
        answer = model.generate(**inputs, max_new_tokens = max_tokens_answer)
        answer = tokenizer.decode(answer[0][inputs["input_ids"].shape[1]:])
        log(f"Model answer: {answer}, True answer: {true_answer}")
        model_answers.append(answer)
        is_right.append(true_answer.lower() in answer.lower())
    data["model_answer"] = model_answers
    data["is_right"] = is_right
    data.to_csv("data_with_model_answers.csv")
    log("Model answers were written")


adj_matricies = []
adj_filenames = []

attn_path = log_path + "/attentions"
os.makedirs(attn_path, exist_ok = True)
adj_filenames = os.listdir(attn_path)
adj_filenames = [log_path + "/" + filename for filename in adj_filenames]

if need_attentions:
    os.makedirs(log_path + "/attentions", exist_ok = True)
    for i in tqdm(range(number_of_batches), desc="Weights calc"): 
        attention_w = grab_attention_weights(model, tokenizer, batched_sentences[i], max_tokens_amount, need_symmetry)
        # sample X layer X head X n_token X n_token
        adj_matricies.append(attention_w)
        if (i+1) % DUMP_SIZE == 0: # dumping
            log(f'Saving: shape {adj_matricies[0].shape}')
            adj_matricies = np.concatenate(adj_matricies, axis=1)
            adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
            filename = log_path + "/attentions/" + str(ceil(i/DUMP_SIZE)) + "of" + str(number_of_files) + '.npy'
            adj_filenames.append(filename)
            np.save(filename, adj_matricies)
            adj_matricies = []
            
    if len(adj_matricies):
        filename = log_path + "/attentions/" + str(ceil(i/DUMP_SIZE)) + "of" + str(number_of_files) + '.npy'
        log(f'Saving: shape {adj_matricies[0].shape}')
        adj_matricies = np.concatenate(adj_matricies, axis=1)
        adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
        np.save(filename, adj_matricies)
log("Attentions must be available here")

if need_topological:
    pool = Pool(num_of_workers)
    stats_tuple_lists_array = []
    ntokens_array = np.array([128] * len(data))
    for i, filename in enumerate(tqdm(adj_filenames, desc='Вычисление признаков')):
        adj_matricies = np.load(filename, allow_pickle=True)
        ntokens = ntokens_array[i*batch_size*DUMP_SIZE : (i+1)*batch_size*DUMP_SIZE]
        splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
        args = [(m, thresholds_array, ntokens, stats_name.split("_"), stats_cap) for m, ntokens in splitted]
        stats_tuple_lists_array_part = pool.starmap(
            count_top_stats, args
        )
        stats_tuple_lists_array.append(np.concatenate([_ for _ in stats_tuple_lists_array_part], axis=3))
    stats_tuple_lists_array = np.concatenate(stats_tuple_lists_array, axis=3)
    np.save(log_path + "/topological_features.npy", stats_tuple_lists_array)
    log("Topological features computed")

# if need_ripser:
#     queue = Queue()
#     number_of_splits = 1
#     dim = 1
#     lower_bound = 1e-3
#     barcodes_file = log_path + "/barcodes"
#     os.makedirs(barcodes_file, exist_ok = True)
#     for i, filename in enumerate(tqdm(adj_filenames, desc='Calculating barcodes')):
#         barcodes = defaultdict(list)
#         adj_matricies = np.load(filename, allow_pickle=True) # samples X 
#         print(f"Matricies loaded from: {filename}")
#         ntokens = ntokens_array[i*batch_size*DUMP_SIZE : (i+1)*batch_size*DUMP_SIZE]
#         splitted = split_matricies_and_lengths_ripser(adj_matricies, ntokens, number_of_splits)
#         for matricies, ntokens in tqdm(splitted, leave=False):
#             p = Process(
#                 target=subprocess_wrap,
#                 args=(
#                     queue,
#                     get_only_barcodes,
#                     (matricies, ntokens, dim, lower_bound)
#                 )
#             )
#             p.start()
#             barcodes_part = queue.get() # block until putted and get barcodes from the queue
#     #         print("Features got.")
#             p.join() # release resources
#     #         print("The process is joined.")
#             p.close() # releasing resources of ripser
#     #         print("The proccess is closed.")
            
#             barcodes = unite_barcodes(barcodes, barcodes_part)
#         part = filename.split('_')[-1].split('.')[0]
#         save_barcodes(barcodes, barcodes_file + '/' + part + '.json')

#     ripser_feature_names=[
#         'h0_s', 
#         'h0_e',
#         'h0_t_d', 
#         'h0_n_d_m_t0.75',
#         'h0_n_d_m_t0.5',
#         'h0_n_d_l_t0.25',
#         'h1_t_b',
#         'h1_n_b_m_t0.25',
#         'h1_n_b_l_t0.95', 
#         'h1_n_b_l_t0.70',  
#         'h1_s',
#         'h1_e',
#         'h1_v',
#         'h1_nb'
#     ]

#     features_array = []
#     barc_filenames = [barcodes_file + "/" + f for f in os.listdir(barcodes_file)]

#     for filename in tqdm(barc_filenames, desc='Calculating ripser++ features'):
#         barcodes = json.load(open(filename))
#         print(f"Barcodes loaded from: {filename}", flush=True)
#         features_part = []
#         for layer in barcodes:
#             features_layer = []
#             for head in barcodes[layer]:
#                 ref_barcodes = reformat_barcodes(barcodes[layer][head])
#                 features = ripser_count.count_ripser_features(ref_barcodes, ripser_feature_names)
#                 features_layer.append(features)
#             features_part.append(features_layer)
#         features_array.append(np.asarray(features_part))

#     features = np.concatenate(features_array, axis=2)
#     np.save(log_path + "/ripser_features.npy", features)
#     log("Ripser features computed")