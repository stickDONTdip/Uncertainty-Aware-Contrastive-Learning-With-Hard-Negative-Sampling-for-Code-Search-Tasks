import argparse
import logging
import os
import pickle

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import json
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from math import log
import numpy as np
import multiprocessing

logger = logging.getLogger(__name__)
cpu_cont = 16

nl_dict = dict()


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 nl_tokens,
                 nl_ids,
                 url,
                 ):
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


def convert_examples_to_features(js, tokenizer):
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:128 - 4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = 128 - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(nl_tokens, nl_ids, js['url'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path=None, pool=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer))

        for idx, example in enumerate(self.examples[:3]):
            print("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
            print("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].nl_ids)


def encode_nl(encoder, nltokenizer, file_path):
    query_dataset = TextDataset(nltokenizer, file_path, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=64, num_workers=4)

    vec = torch.tensor([]).to('cuda')
    tr = 0
    for batch in query_dataloader:
        if tr % 100 == 0:
            print("already count", tr, " batch ")
        nl_inputs = batch.to('cuda')
        with torch.no_grad():
            nl_vec = encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
        vec = torch.cat((vec, nl_vec), dim=0)
        tr += 1
    return vec


def compute_K(dl, avdl):
    k1 = 1.2
    b = 0.75
    return k1 * ((1 - b) + b * (float(dl) / float(avdl)))


def score_BM25(n, f, qf, N, dl, avdl):
    # n - number of documents containing the term
    # f - frequency of the term in the document
    # qf - frequency of the term in the query
    # N - total number of documents
    # dl - length of the document
    # avdl - average length of documents in the collection
    k1 = 1.2
    k2 = 100
    K = compute_K(dl, avdl)
    first = log(((N - n + 0.5) / (n + 0.5)))
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2 + 1) * qf) / (k2 + qf)
    return first * second * third


def bm25_count(query, nl_inputs):
    dict1 = dict()
    dictq = dict()
    avg_len = sum([len(i) for i in nl_inputs]) / len(nl_inputs)
    for docidx, tokens in enumerate(nl_inputs):
        for i in tokens:
            if i in dict1:
                if docidx in dict1[i]:
                    dict1[i][docidx] += 1
                else:
                    dict1[i][docidx] = 1
            else:
                d = dict()
                d[docidx] = 1
                dict1[i] = d
    for i in query:
        if i in dictq:
            dictq[i] += 1
        else:
            dictq[i] = 1
    sim_matrix = np.zeros(len(nl_inputs))
    for token in query:
        if token not in dict1:
            continue
        doc_dict = dict1[token]  # retrieve index entry
        for docid, freq in doc_dict.items():  # for each document and its word frequency
            score = score_BM25(n=len(doc_dict), f=freq, qf=dictq[token], N=len(nl_inputs),
                               dl=len(nl_inputs[docid]), avdl=avg_len)  # calculate score
            sim_matrix[docid] += score
    return sim_matrix


def read(file_path):
    nl = []
    url = []
    print("start to read file: {}".format(file_path))
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            nl.append(js['docstring_tokens'])
            url.append(js['url'])
            nl_dict[js['url']] = line
    avg_len = sum([len(i) for i in nl]) / len(nl)
    return nl, url, avg_len


def process(file_path, encoder, nltokenizer):
    vec = encode_nl(encoder, nltokenizer, file_path)
    vec_batch = torch.reshape(vec, (-1, 1, 768))
    sim_matrix = torch.tensor([])
    count = 0
    for i in vec_batch:
        matrix = torch.einsum("ac,bc->ab", i, vec).cpu()
        matrix = torch.sort(matrix, dim=-1, descending=True)[1][:, :500]
        sim_matrix = torch.cat((sim_matrix, matrix), dim=0)
        count += 1
        if count % 1000 == 0:
            print("already count", count, " queries ")
    np.savetxt(file_path[:-11] + 'sort_ids.txt', sim_matrix.numpy(), fmt='%d')

    sort_ids = np.loadtxt(file_path[:-11] + 'sort_ids.txt', dtype=int)
    nl_inputs, url, avg_len = read(file_path)
    candidate = []
    count = 0
    nl_inputs = np.array(nl_inputs)
    for query, sort_id in zip(nl_inputs, sort_ids):
        scores = bm25_count(query, nl_inputs[sort_id])
        sort_neg = np.argsort(scores, axis=-1, kind='quicksort', order=None)[::-1][50]
        candidate.append(sort_id[sort_neg])
        count += 1
        if count % 1000 == 0:
            print("already count", count, " queries ")

    url = np.array(url)
    url = url[candidate]
    with open(file_path[:-11] + 'neg2.jsonl', 'w') as file:
        for i in url:
            line = json.loads(nl_dict[i])
            file.write(json.dumps(line) + '\n')


pool = multiprocessing.Pool(cpu_cont)
model_name_or_path = ''
model = RobertaModel.from_pretrained(model_name_or_path).to('cuda')
tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
lang = 'php'
file_path = f'../dataset/{lang}/train.jsonl'
process(file_path, model, tokenizer)
