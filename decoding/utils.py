
# this is for transformers==2.2.0

from __future__ import absolute_import, division, print_function, unicode_literals

##############################################################
# For save and load GPT2.0

import pickle


def saveModel(model, modelname):
    msd = model.state_dict()
    for pm in msd:
        msd[pm] = msd[pm].cpu()
    torch.save(msd, modelname + '.pt')
    return


def loadModel(modelname, device=None, alpha=None):
    pathname, filename = os.path.split(modelname)
    modelname = os.path.join(pathname, filename)
    if modelname.endswith('.pt'):
        modelname = modelname[:-3]
    if modelname.endswith('.pkl'):
        modelname = modelname[:-4]
    print(modelname + '.pkl&.pt')

    model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
    shortcut_path = 'gpt2'
    tokenizer = tokenizer_class.from_pretrained(shortcut_path)
    #     model = model_class.from_pretrained('/home/wushan/D:/cache/gpt2-pytorch_model.bin')
    model = model_class.from_pretrained(shortcut_path)
    model.load_state_dict(torch.load(modelname + '.pt'))

    for param_tensor in model.state_dict():
        pn, pm = param_tensor, model.state_dict()[param_tensor]
#         print(pn, "\t", pm.size(),"\t",type(pm), "\t", pm.device)#torch.cuda.device_of(pm))

    if device is not None:
        model.to(device)

    return model, tokenizer


##############################################################
# loading lib and data

import argparse
# import logging
from tqdm import trange
import os, random

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=0.0,
                          filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model,
                    length,
                    context,
                    num_samples=1,
                    temperature=1,
                    top_k=0,
                    top_p=0.0,
                    repetition_penalty=1.0,
                    is_xlnet=False,
                    is_xlm_mlm=False,
                    xlm_mask_token=None,
                    xlm_lang=None,
                    device='cuda'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if is_xlnet:
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat(
                    (generated,
                     torch.zeros((1, 1), dtype=torch.long, device=device)),
                    dim=1)
                perm_mask = torch.zeros(
                    (1, input_ids.shape[1], input_ids.shape[1]),
                    dtype=torch.float,
                    device=device)
                perm_mask[:, :,
                          -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]),
                                             dtype=torch.float,
                                             device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {
                    'input_ids': input_ids,
                    'perm_mask': perm_mask,
                    'target_mapping': target_mapping
                }

            if is_xlm_mlm and xlm_mask_token:
                # XLM MLM models are direct models (predict same token, not next token)
                # => need one additional dummy token in the input (will be masked and guessed)
                input_ids = torch.cat((generated,
                                       torch.full((1, 1),
                                                  xlm_mask_token,
                                                  dtype=torch.long,
                                                  device=device)),
                                      dim=1)
                inputs = {'input_ids': input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] *
                                               inputs["input_ids"].shape[1],
                                               device=device).view(1, -1)

            outputs = model(
                **inputs
            )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (
                temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits,
                                                    top_k=top_k,
                                                    top_p=top_p)
            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(filtered_logits,
                                          dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits,
                                                         dim=-1),
                                               num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


##### Usage:

#     device = 'cuda:1'
#     device = device.lower().replace('gpu', 'cuda')
#     if ':' in device:
#         device, os.environ["CUDA_VISIBLE_DEVICES"] = device.split(':')

#     # load from this modelname
#     modelname = '/home/wushan/D:/37.5W.Overnight_GPTBTa300W.Para_acc.95.0.pt'

#     model, tokenizer = loadModel(modelname, device=device)


##############################################################
# data process
# def preprocess(line):
#     line = line.strip()
#     # 首字母去转小写
#     line = line[0].lower() + line[1:]
#     # 去掉问号
#     if line.endswith('?'):
#         line = line[:-1].strip()
#     # 去掉括号
#     while '(' in line and ')' in line:
#         line = line[:line.find('(')] + line[line.find(')') + 1:]
#     return line
def preprocess(line):
    line = line.strip()
    # # 首字母去转小写
    # line = line[0].lower() + line[1:]
    line = line.lower()
    # 去掉问号
    if line.endswith('?'):
        line = line[:-1].strip()
    # 去掉括号
    while '(' in line and ')' in line:
        line = line[:line.find('(')] + line[line.find(')') + 1:]
    return line


import random
paras = {}


# direct=2: two direction
# direct=1: s1 -> s2
# direct=0: s2 -> s1
def getParaData(
        fn='/home/wushan/paraphrase/wikianswers-paraphrases-1.0/paras.txt',
        sampleN=10000,
        direct=2):
    global paras
    fns = [fn] if type(fn) is str else fn
    fn = '\t'.join(fns)
    if paras.get(fn, None) is None:
        ps = []
        for f in fns:
            for line in open(f):
                qs = line.strip().split('\t')
                ps.append(qs)
        paras[fn] = ps
    ps = paras.get(fn, None)
    raw_texts, raw_outs = [], []
    for i in range(sampleN):
        qs = random.choice(ps)
        if direct == 2:
            q1, q2 = random.sample(qs, 2)
        elif direct == 1:
            q1, q2 = qs[0], qs[1]
        else:
            q1, q2 = qs[1], qs[0]
        raw_texts.append(preprocess(q1))
        raw_outs.append(preprocess(q2))
    return raw_texts, raw_outs


##############################################################
# Test Sample:

import random


def sampleParas(
        model,
        tokenizer,
        device,
        fn='/home/wushan/paraphrase/wikianswers-paraphrases-1.0/paras.txt',
        sampleN=10):
    raw_texts, raw_outs = getParaData(fn=fn, sampleN=sampleN)
    training = model.training
    model.train(False)
    for raw_text, raw_out in random.sample(list(zip(raw_texts, raw_outs)), 5):

        context_tokens = tokenizer.encode(raw_text + ' PR:',
                                          add_special_tokens=False)
        out = sample_sequence(
            model=model,
            length=30,
            context=context_tokens,
            device=device,
            top_k=10,
        )
        #         out = out[:, len(context_tokens):].tolist()
        out = out[:, len(context_tokens):].tolist()
        text = tokenizer.decode(out[0], clean_up_tokenization_spaces=True)
        text = text.split('\n')[0]
        print(raw_text)
        print(out[0])
        print('->', raw_out)
        print('-:', text)
        #         text = replace(text,toNatural=False)
        # print('::', text)
        # for o in out:
        #     text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
        #     # text = text[: text.find(args.stop_token) if args.stop_token else None]
        #     print('-:',text)
    model.train(training)


##############################################################
# Decoding Scores:

# import sys
# import gc
# dp_softmax = {}

# def clear_dp():
#     global dp_softmax
#     dp_softmax.clear()
#     del dp_softmax
#     gc.collect()
#     dp_softmax = {}


def decodingScore(words,
                  predictWords,
                  eos=False,
                  tokenizer=None,
                  device=None,
                  model=None,
                  dp_softmax={}):
    raw_tokens = tokenizer.encode(words + ' PR:', add_special_tokens=False)
    out_tokens = tokenizer.encode(predictWords, add_special_tokens=False)
    if eos:
        out_tokens += [198]  # eos
    rawout = raw_tokens + out_tokens

    loss = 0
    for i in range(len(out_tokens)):
        x = rawout[:len(raw_tokens) + i]
        y = [out_tokens[i]]
        key = tuple(x + y)
        if key in dp_softmax:
            score = dp_softmax[key]
        else:
            with torch.no_grad():
                x = torch.tensor(x, dtype=torch.long, device=device)
                x = x.unsqueeze(0).repeat(1, 1)
                outputs = model(input_ids=x)
                score = torch.log_softmax(outputs[0][:, -1, :], dim=-1)
                score = score.cpu().numpy()
                dp_softmax[key] = score
#         print (score.shape) #(1, 50257)
        loss += -score[0, y[0]]
    return loss


##############################################################
# read nls and onls from xxx.example(Overnight original format)
import re


def getStcFromExample(fileName, removeBrackets=False):
    nls = []
    onls = []
    for line in open(fileName):
        if line.strip().startswith('(utterance'):
            l = line[line.find('"') + 1:line.rfind('"')]
            l = re.sub(r' ([\d]+)([^\d\s]+) ', r" \1 \2 ", ' ' + l + ' ')
            l = l.replace('(', ' ( ').replace(')', ' ) ')
            # l = ' '.join([norm_word(li) for li in l.split() if li])
            nls.append(l)
        elif line.strip().startswith('(original'):
            l = line[line.find('"') + 1:line.rfind('"')]
            l = re.sub(r' ([\d]+)([^\d\s]+) ', r" \1 \2 ", ' ' + l + ' ')
            l = l.replace('(', ' ( ').replace(')', ' ) ')
            # l = ' '.join([norm_word(li) for li in l.split() if li])
            if removeBrackets:
                while '(' in l:
                    l = l[:l.find('(')] + l[l.find(')') + 1:]
            l = ' '.join(l.strip().split())
            onls.append(l)
    return nls, onls

# codict = {}
# codict_stem = {}

func_words = [
    'of', 'to', 'in', 'and', 'as', 'from', 'for', 'with', 'that', 'have', 'by',
    'on', 'upon', 'about', 'above', 'ahead', 'after', 'a', 'an', 'although',
    'at', 'also', 'along', 'around', 'always', 'away', 'any', 'up', 'under',
    'until', 'before', 'between', 'beyond', 'behind', 'because', 'what',
    'when', 'would', 'could', 'who', 'whom', 'whose', 'which', 'where', 'why',
    'without', 'whether', 'down', 'during', 'despite', 'over', 'off', 'only',
    'other', 'out', 'the', 'then', 'through', 'throughout', 'that', 'these',
    'this', 'those', 'there', 'therefore', 'since', 'so', 'can', 'many',
    'much', 'more', 'may', 'might', 'must', 'ever', 'even'
]

import json


def getCodict(
    codictfn='/home/wushan/paraphrase/wikianswers-paraphrases-1.0/codict_stem.json',
    alignfn='/home/wushan/paraphrase/wikianswers-paraphrases-1.0/word_alignments.txt'
):
    if codictfn:
        codict = json.load(open(codictfn))
    else:
        codict = {}
        i = 0
        for line in open(alignfn):
            ws0, ws1, align = line.strip().split('\t')
            ws0, ws1, align = ws0.strip().split(), ws1.strip().split(
            ), align.strip().split()
            ws0 = [porter_stemmer.stem(w) for w in ws0]
            ws1 = [porter_stemmer.stem(w) for w in ws1]
            for w in ws0 + ws1:
                codict[w] = codict.get(w, 0) + 1
            for a in align:
                a0, a1 = a.strip().split('-')
                w = ws0[int(a0)] + connector + ws1[int(a1)]
                codict[w] = codict.get(w, 0) + 1
            i += 1
            # print(str(len(codict))+'\t'+str(i),end='\r')
            print(str(i), end='\r')
    return codict


import math
import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()


# todo: 分开stem和不stem，分开获得和使用codict
def getCof(w0, w1, codict, stem=True, discR=1, connector='#^#'):
    if stem:
        w0 = porter_stemmer.stem(w0)
        w1 = porter_stemmer.stem(w1)
    cof = 1.0 * (codict.get(w0 + connector + w1, 0) +
                 0.0000000001) / (codict.get(w0, 0) + 1)
    cof = math.pow(cof, discR)
    return cof
