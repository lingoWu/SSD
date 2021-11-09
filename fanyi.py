# coding=utf-8

import re
import time

import http.client
import hashlib
import urllib
import random
import json


def getTranslation(q, toLang="zh"):  # q: just a stc

    appid = '20160520000021546'  # 填写你的appid
    secretKey = 'Pt_uQBAJCIccCTbDmkLq'  # 填写你的密钥

    httpClient = None
    myurl = '/api/trans/vip/translate'

    fromLang = 'auto'  #原文语种
    #     toLang = 'zh'   #译文语种
    salt = random.randint(32768, 65536)

    #     q= 'apple'
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
            salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        print(result)

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()
    return result['trans_result'][0]['dst']

def getTranslations(qs, toLang="zh",maxchar=4000,wt=1): # q的序列即可，方法内会自己拆分成合适的batch大小

    appid = '20160520000021546'  # 填写你的appid
    secretKey = 'Pt_uQBAJCIccCTbDmkLq'  # 填写你的密钥

    httpClient = None
    myurl_ = '/api/trans/vip/translate'

    fromLang = 'auto'  #原文语种
    #     toLang = 'zh'   #译文语种
    salt = random.randint(32768, 65536)

    #     q= 'apple'
    q_batch = [qs[0]]
    for q in qs[1:]:
        q_ = q_batch[-1] + '\n' + q
        if len(q_.encode('utf-8')) > maxchar:
            q_batch.append(q)
        else:
            q_batch[-1] = q_
    
    # print(q_batch)
    res = []
    for i,qb in enumerate(q_batch):
        sign = appid + qb + str(salt) + secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()
        myurl = myurl_ + '?appid=' + appid + '&q=' + urllib.parse.quote(
            qb) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
                salt) + '&sign=' + sign

        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)

            # response是HTTPResponse对象
            response = httpClient.getresponse()
            result_all = response.read().decode("utf-8")
            result = json.loads(result_all)
            print(result)

        except Exception as e:
            print(e)
            print(qb)
            print(myurl)
            exit()
        finally:
            if httpClient:
                httpClient.close()
        res += [r['dst'] for r in result['trans_result']]
        if True: # i+1 < len(q_batch):
            time.sleep(wt)
    return res
