# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from unittest import removeResult
import torch.nn.functional as F
import argparse
import logging
import os

import pickle
import random
import torch
import json
from random import choice
import numpy as np
from itertools import cycle
from model_cocosoda import Model
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)
from tqdm import tqdm
import multiprocessing

cpu_cont = 16

import sys

sys.path.append("dataset")
from utils import save_json_data, save_pickle_data


def lalign(x, y, alpha=2):
    x = torch.tensor(x)
    y = torch.tensor(y)
    return (x - y).norm(dim=1).pow(alpha).mean()


def lunif(x, t=2):
    x = torch.tensor(x)
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


def cal_r1_r5_r10(ranks):
    r1, r5, r10 = 0, 0, 0
    data_len = len(ranks)
    for item in ranks:
        if item >= 1:
            r1 += 1
            r5 += 1
            r10 += 1
        elif item >= 0.2:
            r5 += 1
            r10 += 1
        elif item >= 0.1:
            r10 += 1
    result = {"R@1": round(r1 / data_len, 3), "R@5": round(r5 / data_len, 3), "R@10": round(r10 / data_len, 3)}
    return result



class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


def convert_examples_to_features_unixcoder(js, tokenizer, args):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'] if "url" in js else js["retrieval_idx"])


class TextDataset_unixcoder(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pooler=None):
        self.examples = []
        data = []
        n_debug_samples = args.n_debug_samples
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
                    if args.debug and len(data) >= n_debug_samples:
                        break
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
                    if args.debug and len(data) >= n_debug_samples:
                        break
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)
                    if args.debug and len(data) >= n_debug_samples:
                        break
                        # if "test" in file_path:
        #     data = data[-200:]
        for js in data:
            self.examples.append(convert_examples_to_features_unixcoder(js, tokenizer, args))

        if "train" in file_path:
            # self.examples = self.examples[:128]
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids)


class TrainDataset(Dataset):
    def __init__(self, posdataset, negdataset):
        self.posdataset = posdataset
        self.negdataset = negdataset

    def __len__(self):
        return len(self.posdataset.examples)

    def __getitem__(self, item):
        return self.posdataset[item] + self.negdataset[item]


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all gpus
    torch.backends.cudnn.deterministic = True


def contrastive_loss(a_vec=None, b_vec=None, neg_vec=None, temper=0.03):
    if neg_vec is None:
        distance_matrix = torch.mm(a_vec, b_vec.t())
        loss_fct = CrossEntropyLoss()
        label = torch.arange(a_vec.size(0)).long().to(distance_matrix.device)
        loss = loss_fct(distance_matrix / temper, label)
    else:
        distance_matrix = torch.mm(a_vec, b_vec.t())
        neg_distance_matrix = torch.mm(a_vec, neg_vec.t())
        matrix = torch.cat([distance_matrix, neg_distance_matrix], dim=1)
        label = torch.arange(b_vec.size(0)).long().to(distance_matrix.device)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(matrix / temper, label)
    return loss


def train(args, model, tokenizer, pool):
    """ Train the model """
    pos_dataset = TextDataset_unixcoder(tokenizer, args, args.train_data_file, pool)
    neg_dataset = TextDataset_unixcoder(tokenizer, args, args.neg_data_file, pool)
    train_dataset = TrainDataset(pos_dataset, neg_dataset)
    # train_dataset = TextDataset_unixcoder(tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4,
                                  drop_last=True)

    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * args.num_train_epochs)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Num quene = %d", args.moco_k)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)

    model.zero_grad()
    model.train()
    tr_num, tr_loss, un_loss, loss_class, best_mrr = 0.0, 0.0, 0.0, 0.0, 0.0
    # if args.model_type ==  "multi-loss-cocosoda" :
    if args.model_type in ["no_aug_cocosoda", "multi-loss-cocosoda"]:
        if args.do_continue_pre_trained:
            logger.info("do_continue_pre_trained")
        elif args.do_fine_tune:
            logger.info("do_fine_tune")
    for idx in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):

            # get inputs
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)

            # get code and nl vectors
            kl_loss, uncertain, nl, code, nl_candidate, code_candidate = model(code_inputs=code_inputs,
                                                                               nl_inputs=nl_inputs,
                                                                               labels='train')

            neg_code_inputs = batch[2].to(args.device)
            neg_code = model(neg_inputs=neg_code_inputs)
            contrastive = contrastive_loss(a_vec=nl, b_vec=code, neg_vec=neg_code)
            # contrastive = contrastive_loss(a_vec=nl, b_vec=code)
            contrastive += contrastive_loss(a_vec=nl, b_vec=nl_candidate)
            contrastive += contrastive_loss(a_vec=code, b_vec=code_candidate)

            loss = contrastive + kl_loss.sum()
            uncertain = uncertain.mean()
            tr_loss += loss.item()
            un_loss += uncertain.item()
            tr_num += 1
            if step % 100 == 0:
                logger.info(
                    "epoch {} step {} loss {} un_loss {}".format(idx, step,
                                                                 round(tr_loss / tr_num, 5),
                                                                 round(un_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0
                un_loss = 0
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        results = evaluate(args, model, tokenizer, args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

            # save best model
        if results['eval_mrr'] > best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  " + "*" * 20)
            logger.info("  Best mrr:%s", round(best_mrr, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, file_name, pool, eval_when_training=False):
    # if "unixcoder" in args.model_name_or_path or "coco" in args.model_name_or_path :
    dataset_class = TextDataset_unixcoder
    # else:
    # dataset_class = TextDataset
    query_dataset = dataset_class(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    code_dataset = dataset_class(tokenizer, args, args.codebase_file, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation on %s *****" % args.lang)
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    nl_vecs = []
    code_vecs = []
    for batch in query_dataloader:
        nl_inputs = batch[-1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())

    for batch in code_dataloader:
        with torch.no_grad():
            code_inputs = batch[0].to(args.device)
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())

    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)
    model.train()

    scores = np.matmul(nl_vecs, code_vecs.T)

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = []
    code_urls = []
    for example in code_dataset.examples:
        code_urls.append(example.url)
    for example in query_dataset.examples:
        nl_urls.append(example.url)
    ranks = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)
    if args.save_evaluation_reuslt:
        evaluation_result = {"nl_urls": nl_urls, "code_urls": code_urls, "sort_ids": sort_ids[:, :10], "ranks": ranks}
        save_pickle_data(args.save_evaluation_reuslt_dir, "evaluation_result.pkl", evaluation_result)
    result = cal_r1_r5_r10(ranks)
    result["eval_mrr"] = round(float(np.mean(ranks)), 3)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    # soda
    parser.add_argument('--do_zero_short', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--agg_way', default="avg", choices=["avg", "cls_pooler", "avg_cls_pooler"],
                        help="base is codebert/graphcoder/unixcoder", required=False)
    parser.add_argument('--weight_decay', default=0.01, type=float, required=False)
    parser.add_argument('--do_single_lang_continue_pre_train', action='store_true',
                        help='do_single_lang_continue_pre_train', required=False)
    parser.add_argument('--save_evaluation_reuslt', action='store_true', help='save_evaluation_reuslt', required=False)
    parser.add_argument('--save_evaluation_reuslt_dir', type=str, help='save_evaluation_reuslt', required=False)
    parser.add_argument('--epoch', type=int, default=50,
                        help="random seed for initialization")
    # new continue pre-training
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--loaded_model_filename", type=str, required=False,
                        help="loaded_model_filename")

    parser.add_argument("--max_steps", default=100, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_warmup_steps", default=0, type=int, help="num_warmup_steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")

    # debug
    parser.add_argument('--use_best_mrr_model', action='store_true', help='cosine_space', required=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    parser.add_argument("--max_codeblock_num", default=10, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--eval_frequency", default=1, type=int, required=False)

    # model type
    parser.add_argument('--model_type', default="base",
                        choices=["base", "cocosoda", "multi-loss-cocosoda", "no_aug_cocosoda"],
                        help="base is codebert/graphcoder/unixcoder", required=False)

    # options for moco v2
    parser.add_argument('--mlp', action='store_true', help='use mlp head')

    ## Required parameters
    parser.add_argument("--train_data_file", default="dataset/java/train.jsonl", type=str, required=False,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default="saved_models/pre-train", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default="dataset/java/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="dataset/java/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--neg_data_file", default="dataset/java/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="dataset/java/codebase.jsonl", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")

    parser.add_argument("--lang", default="java", type=str,
                        help="language.")

    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=50, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=100, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=0, type=int,
                        help="Optional Data Flow input sequence length after tokenization.", required=False)

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=4, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=3407,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()
    return args


def create_model(args, model, tokenizer, config=None):

    if (args.loaded_model_filename) and ("pytorch_model.bin" in args.loaded_model_filename):
        logger.info("reload pytorch model from {}".format(args.loaded_model_filename))
        model.load_state_dict(torch.load(args.loaded_model_filename), strict=False)
        # model.from_pretrain
    if args.model_type == "base":
        model = Model(model, config)
    if (args.loaded_model_filename) and ("pytorch_model.bin" not in args.loaded_model_filename):
        logger.info("reload model from {}".format(args.loaded_model_filename))
        model.load_state_dict(torch.load(args.loaded_model_filename))
    if (args.loaded_codebert_model_filename):
        logger.info("reload pytorch model from {}".format(args.loaded_codebert_model_filename))
        model.load_state_dict(torch.load(args.loaded_codebert_model_filename), strict=False)
    # logger.info(model.model_parameters())

    return model


def main():
    args = parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    pool = multiprocessing.Pool(cpu_cont)

    # Set seed
    set_seed(args.seed)

    # build model

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    model = create_model(args, model, tokenizer, config)

    logger.info("Training/evaluation parameters %s", args)
    args.start_step = 0

    model.to(args.device)

    # Training
    if args.do_train:
        train(args, model, tokenizer, pool)

    # Evaluation
    results = {}

    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if (not args.only_save_the_nl_code_vec) and (not args.do_zero_short):
            model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.eval_data_file, pool)
        logger.info("***** Eval valid results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:

        logger.info("runnning test")
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if (not args.only_save_the_nl_code_vec) and (not args.do_zero_short):
            model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.test_data_file, pool)
        logger.info("***** Eval test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))
        save_json_data(args.output_dir, "result.jsonl", result)
    return results


if __name__ == "__main__":
    main()
