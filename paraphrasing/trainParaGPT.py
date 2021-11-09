# coding=utf-8

from utils import saveModel, loadModel
from utils import top_k_top_p_filtering, sample_sequence
from utils import preprocess, getParaData
from utils import sampleParas
from utils import decodingScore
from utils import getStcFromExample
from utils import getDevAcc

import sys 
import os
# load model
if True:
    # device = 'cuda:2'
    # device = device.lower().replace('gpu', 'cuda')
    # if ':' in device:
    #     device, os.environ["CUDA_VISIBLE_DEVICES"] = device.split(':')
    device = 'cuda'
    # os.environ["CUDA_VISIBLE_DEVICES"] is Given

    if len(sys.argv) > 1:
        modelFileName = sys.argv[1]
        print('start model from file:', modelFileName)
        sys.stdout.flush()
        model, tokenizer = loadModel(modelFileName, device)
    else:
        model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
        shortcut_path = 'gpt2'
        tokenizer = tokenizer_class.from_pretrained(shortcut_path)
        model = model_class.from_pretrained(shortcut_path)
    print(model)
    model.to(device)

    ### python xx.py modelFile .50w
    fileFix = ''
    if len(sys.argv) > 2:
        fileFix = sys.argv[2]

    domain = None
    if len(sys.argv) > 3:
        domain = sys.argv[3]

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# train model
if True:
    model.train()
    optimizer = optim.Adam([{
        'params': model.parameters()
    }], lr=5e-6)  #,weight_decay=0.001) # 原先是1e-5
    optimizer.zero_grad()
    ls = []
    bs = 20
    trainN = 1500
    for epoch in range(100):
        # test domain acc of top1 selection
        if epoch % 2 == 0:
            domainsRight = getDevAcc(
                model,
                tokenizer,
                device,
                domains=[domain],
                file='test',  # test/train
                sampleN=30,  # sampleN > 3000 means all
                intopK=1,  # 1 means only acc; >1 means beamsizeaz
                choiceK=10000,
                model2=None,
                tokenizer2=None,
                reverse=False,
                verbose=False)
            acc = np.mean(domainsRight)
            saveModel(
                model, '/home/wushan/D:/' + str(epoch * trainN / 10000.0) +
                'W.BT1d_GPT300W.Para' + fileFix + '_acc.' + str(acc)[:5])
            print('saved')
        # 训练数据
        # raw_texts, raw_outs = getParaData(
        #     '/home/wushan/sp/sssp_ws/dataOvernight/overnight.train.paras',
        #     1500)
        cus, BTen = getParaData([
            '/home/wushan/download/overnight_data/' + domain + '.autoQA.txt',
        ],
                                trainN,
                                direct=1)
        # reverse:
        raw_texts, raw_outs = BTen, cus
        # 开始训练：raw_texts -> raw_outs
        print('epoch:', epoch, flush=True)
        loss = 0
        for n, (raw_text, raw_out) in enumerate(zip(raw_texts, raw_outs)):
            # raw_text  += ' PR:'
            # print ('train',raw_text,'->',raw_out)
            raw_tokens = tokenizer.encode(raw_text + ' PR:',
                                          add_special_tokens=False)
            out_tokens = tokenizer.encode(raw_out, add_special_tokens=False)
            out_tokens += [198]  # eos
            # print ('train',raw_tokens,'->',out_tokens)
            xs = torch.tensor(raw_tokens + out_tokens,
                              dtype=torch.long,
                              device=device)
            ys = torch.tensor([out_tokens], dtype=torch.long, device=device)
            los = 0
            for i in range(len(out_tokens)):
                x = xs[:len(raw_tokens) + i]
                x = x.unsqueeze(0).repeat(1, 1)
                # print ('x:',x)
                outputs = model(input_ids=x)
                # y = torch.tensor(out_tokens[i], dtype=torch.long, device=device)
                # print(outputs[0][:, -1,:][0,y])
                y = ys[:, i]
                # print ('y:',y)
                # print(F.nll_loss(outputs[0][:, -1,:],y))
                score = torch.log_softmax(outputs[0][:, -1, :], dim=-1)
                l = F.nll_loss(score, y)
                loss += l
                los += l
            ls.append(los.cpu().data.numpy() / len(out_tokens))
            if n > 0 and n % bs == 0:
                print(n, 'loss:', np.mean(ls[-bs * 200:]))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = 0

## usage:  CUDA_VISIBLE_DEVICES=1 python trainParaGPT.py xxxx_pretrained namefix
"""CUDA_VISIBLE_DEVICES=2 nohup \
python trainValidParaGPT_1d_AutoQA.py \
/home/wushan/D:/260W.GPT.Para.50w_acc.65.0.pt autoQA socialnetwork \
>trainValidParaAutoQA_GPTa.log 2>&1 &
"""
