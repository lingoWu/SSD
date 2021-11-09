
# 定义全局变量
parse = False
general = True
generate = True
regex = False
geo880 = False
geo440 = False
# right = False
# rules = {}  # rules = {'$x':[(nl1,lf1),(nl2,lf2),,,],,,}


def print_rules(rules):
    # rules = {'$x':[(nl1,lf1),(nl2,lf2),,,],,,}
    for k, v in rules.items():
        print(k, '->')
        for e in v:
            print('\t' + '<' + e[0] + ',' + e[1] + '>')
    return


def get_generalRules(verbose=False):
    # 读取general的规则
    level = 0
    rules = {}
    for line in open('/home/wushan/sp/sssp_ws/dataOvernight/general.grammar'):
        if '#' in line:
            line = line[:line.find('#')]
        if len(line.strip()) == 0: continue
        level += line.count('(') - line.count(')')
        if line.strip().startswith('(when'):
            ws = [
                w for w in line[line.rfind('(when') +
                                5:].strip().strip('()').strip().split() if w
            ]
            if ws[0] in ['and', 'or']:
                right = eval((' ' + ws[0] + ' ').join(ws[1:]))
            else:
                right = eval(ws[0])
        elif line.strip().startswith('(rule'):
            if not right:
                continue
            line = line.strip()
            ruleName = line.split()[1]
            l1 = 1 + line[1:].find('(')
            r1 = 1 + line[1:].find(')')
            l2 = r1 + 1 + line[r1 + 1:].find('(')
            r2 = r1 + 1 + line[r1 + 1:-1].rfind(')')
            rs = rules.get(ruleName, [])
            rs.append([line[l1 + 1:r1], line[l2 + 1:r2]])
            rules[ruleName] = rs
        elif line.count('(') - line.count(')') < 0 and level == 0:
            right = False
        else:
            if verbose:
                print(line.strip())
    # print ('rules:',rules)
    print_rules(rules)
    return rules


import sys


# mergeRule用于将两个rules的LF部分结合,NL部分比较简单另外处理
# r1需要被扩展的LF，r2是某个变量需要被扩展成的LF
# ext非空就是general终结符可替换domain的constant
# JOIN类的需要单独处理,
# $NP1||$ROOT->lambda^x^(call^@listValue^(var^x)) # $NP1
# $RelNP^of^$EventNP1||$NP1->^lambda^x^(call^@getProperty^(var^x)^($RelNP)) # $RelNP of $EventNP1
# season||$RelNP->ConstantFn^(string^season) # season of $EventNP1
# $EventNP0^$CP00||$EventNP1->JoinFn^backward^betaReduce # season of $EventNP0 $CP00
# $Rel0NP^$EntityNP1||$EventNP0->^call^@getProperty^($EntityNP1)^(call^@reverse^($Rel0NP)) # season of $Rel0NP $EntityNP1 $CP00
# player||$Rel0NP->ConstantFn^(string^player) # season of player $EntityNP1 $CP00
# kobe^bryant||$EntityNP1->ConstantFn^en.player.kobe_bryant # season of player kobe bryant $CP00
# whose^$RelNP^is^$EntityNP1||$CP00->^lambda^s^(call^@filter^(var^s)^($RelNP)^(string^=)^($EntityNP1))
# number^of^fouls^[over^a^season]||$RelNP->ConstantFn^(string^num_fouls)
# 3||$EntityNP1->ConstantFn^(number^3^foul)
# 一般情况都是r1中的某个 xx (lambda s (xxx var s))需要被替换
## (第vth个变量需要替换,从NL部分的变量位置得出LF中的变量位置)
def mergeRule(r1, r2, vth, ext=None, JOIN=False, verbose=False):

    # 去除多余空格
    r1 = ' '.join([w for w in r1.split() if w])
    r2 = ' '.join([w for w in r2.split() if w])

    # 有JoinFn的已经有另行处理
    if 'JoinFn' in r1:
        return r2
    if 'JoinFn' in r2:
        return r1
    if JOIN:
        if verbose:
            print('to JOIN:', r1, '|', r2, '|', vth)
        l = r1.rfind('getProperty')
        # l = l + r1[l:].find('(')
        l = r1[:l].rfind('(')
        for r in range(l, len(r1)):
            if r1[r] == ')' and r1[l:r + 1].count('(') == r1[l:r +
                                                             1].count(')'):
                break
        ins = mergeRule(r2, r1[l + 1:r], vth=r2.count('lambda') - 1)
        r1 = r1[:l + 1] + ins + r1[r:]
        if verbose:
            print('JOIN=>', r1)
        # exit()
        return r1

    # r2中有'ConstantFn'则剥去'ConstantFn(xxx)'只留下xxx
    if ('ConstantFn' in r2 or len(r2.strip().split()) == 1):
        r2 = r2.replace('ConstantFn', '').strip()
        if r2.strip().startswith('('):
            r2 = r2[r2.find('(') + 1:r2.rfind(')')].strip()

    # ext非空，表示知道要将什么符号扩展成r2，即单纯的ConstantFn替换原先的general终结符
    # 将r1中的要扩展符换成r2，如果r2左右分别有右左括号则除掉两边的括号
    # 循环替换为r2，返回替换好的即可
    if ext and ext in r1:
        # r1.replace(ext,r2) # change to the below
        while ext in r1:
            # if ext in r1: # replace one ext
            i = r1.find(ext)
            r1 = r1[:i] + r2 + r1[i + len(ext):]
            l = r1[:i].rfind('(')
            r = i + r1[i:].find(')')
            if l >= 0 and r >= i and ' ' not in r1[l + 1:r].strip():
                r1 = r1[:l] + r1[l + 1:r] + r1[r + 1:]
        # print (r1,r2,'exit')
        # exit()
        return r1

    # ext为空或者r1中不含有ext
    # 即，一般情况，第vth个lambda变量替换了
    i = r1.find('lambda')
    if verbose:
        print(r1, '|', r2, '|', vth)
    if i < 0:
        print('no lambda')
        print('to extend:', r1, '|', r2, '|', vth, '|', ext)


#         exit()
    while vth > 0:
        if 'lambda' not in r1[i + 6:]:
            if verbose:
                print(r1, i, vth)
        i = i + 6 + r1[i + 6:].find('lambda')
        vth -= 1
    vn = r1[i:].strip().split()[1]  # var name
    # 找到了vth个变量名为vn
    #     if 'lambda' in r2:
    #         print (r1,'|',r2,'|',vth,'lambda in r2')
    namen = r1.count('lambda ' + vn + ' ')
    while namen > 1:
        # 对vn重新编号，第i个变成_ivn，lambda和对应辖域的var都变
        # 第一个vn不变，用于后面的remove处理
        l = r1.rfind('lambda ' + vn)
        for r in range(l, len(r1)):
            if r1[r] == ')' and r1[l:r + 1].count('(') == r1[l:r +
                                                             1].count(')'):
                break
        r1 = r1[:l] + r1[l:r].replace('lambda '+vn+' ','lambda '+'_'+str(namen)+vn+' ')\
                .replace('var '+vn,'var '+'_'+str(namen)+vn)  +  r1[r:]
        namen = r1.count('lambda ' + vn + ' ')
    # xxx ( lambda vn ( xxx ) ) 中剔除 lambda vn ( )
    l = i + r1[i:].find('(')
    for r in range(l, len(r1)):
        if r1[r] == ')' and r1[l:r + 1].count('(') == r1[l:r + 1].count(')'):
            break
    r1 = r1[:l] + r1[l + 1:r] + r1[r + 1:]
    r1 = r1.replace('lambda ' + vn + ' ', ' ')
    # var vn 替换成 r2 即可
    r1 = r1.replace('var ' + vn, r2)
    return r1


def rules_add_vth(rules, terminals, verbose=False):
    # 对每个rules都进行，把单一的终结符直接带入
    for rn in rules:
        i = 0
        while i < len(rules[rn]):
            vi = 0
            stc, lgf = rules[rn][i]
            # 确认没有变量个数不统一的情况
            if '$' in stc and 'lambda' not in lgf:  # and 'JoinFn' not in lgf:
                print('skip', stc, lgf)
                i += 1
                continue
                exit()
                # remove
                #### this step dont removed JoinFn
                rules[rn][i:i + 1] = []
                continue
            # 所有正确遍历的在此输出
            if verbose:
                print('>:', stc, '||', lgf)
            for w in stc.split():
                ## general 中的终结符，比如'$TypeNP’
                ## 如果某个rule中的NL出现了终结符
                ## 比如：$TypeNP || lambda t (call @getProperty (call @singleton (var t)) (string !type))
                ## 则，stc不变
                ## 对应的lgf中：extend terminal $TypeNP
                ## 将变量替换成常量：call @getProperty (call @singleton ($TypeNP)) (string !type)
                if w.startswith('$') and w in terminals:
                    if verbose:
                        print('extend terminal', w)
                    idx = stc.find(w)
                    # stc = stc[:idx] + '' + stc[idx+len(w):]
                    lgf = mergeRule(lgf, w, stc[:idx].count('$') - vi)
                    vi += 1
                    if verbose:
                        print(':', lgf)
                    # break
            rules[rn][i] = [stc, lgf]
            i += 1
    return rules


## w,要被扩展的非终结符
## r,需要使用的扩展规则
## stc,lgf，目前的句子和合成LF
## stcR, lgfR，目前的句子和合成LF所使用的生成式规则序列
def extendByRule(w, r, stc, lgf, stcR, lgfR, verbose=False):
    stcR.append(r[0])
    lgfR.append(w + '->' + r[1])
    if verbose:
        print(stcR, lgfR)
        print(w, '->', r)
    idx = stc.find(w)

    JOIN = False
    ss = stc[idx:].split()
    if len(ss) > 1 and ss[1] == 'JOIN':
        JOIN = True
        stc = stc[:idx] + ' '.join(ss[:1] + ss[2:])

    if 'JoinFn' in r[1]:
        stc = stc[:idx] + r[0] + ' JOIN' + stc[idx + len(w):]
    else:
        stc = stc[:idx] + r[0] + stc[idx + len(w):]

    lgf = mergeRule(lgf, r[1], stc[:idx].count('$'), ext=w, JOIN=JOIN)

    return stc, lgf, stcR, lgfR


def getUnfinishPart(stc, is_nonterminal):
    ws = stc.strip().split()
    unf = ''
    for i in range(len(ws)):
        w = ws[i]
        # 第一个非终结符
        # if w.startswith('$') and w not in terminals: # For General
        # if w.startswith('$') and len(domainRules.get(w,[])) == 0:
        if is_nonterminal(w):
            if i + 1 < len(ws) and ws[i + 1] != 'JOIN':
                stc = ' '.join(ws[:i + 1])
                unf = ' '.join(ws[i + 1:])
                unf = unf.strip()
                break
            else:
                stc = ' '.join(ws[:i + 2])
                unf = ' '.join(ws[i + 2:])
                unf = unf.strip()
                break
    return stc, unf


def extendByRule_withUnfinishStack(w,
                                   r,
                                   stc,
                                   lgf,
                                   stcR,
                                   lgfR,
                                   unfinishStack,
                                   is_nonterminal,
                                   verbose=False):
    # non-terminal must at the right of stc
    stcR.append(r[0])
    lgfR.append(w + '->' + r[1])
    # print (stcR,lgfR)
    # print(w,'->',r)
    idx = stc.find(w)
    # stc[:idx] 内不能有其他非终结符

    JOIN = False
    ss = stc[idx:].split()
    if len(ss) > 1 and ss[1] == 'JOIN':
        JOIN = True
        # just remove one 'JOIN'
        stc = stc[:idx] + ' '.join(ss[:1] + ss[2:])

    if 'JoinFn' in r[1]:
        stc = stc[:idx] + r[0] + ' JOIN' + stc[idx + len(w):]
    else:
        stc = stc[:idx] + r[0] + stc[idx + len(w):]

    # 一但，第一个非终结符后有东西（包括非终结符），直接把他们加到unfinishStack中，unf里可以包括非终结符
    stc, unf = getUnfinishPart(stc, is_nonterminal)
    if len(unf):
        unfinishStack = [unf] + unfinishStack

    lgf = mergeRule(lgf, r[1], stc[:idx].count('$'), ext=w, JOIN=JOIN)

    return stc, lgf, stcR, lgfR, unfinishStack


## 注意query使用时，需要在本机3093端口上开启evalFuwu服务
# from preprocess import query
alias = '''(def @domain edu.stanford.nlp.sempre.overnight.SimpleWorld.domain)
    (def @singleton edu.stanford.nlp.sempre.overnight.SimpleWorld.singleton)
    (def @filter edu.stanford.nlp.sempre.overnight.SimpleWorld.filter)
    (def @getProperty edu.stanford.nlp.sempre.overnight.SimpleWorld.getProperty)
    (def @superlative edu.stanford.nlp.sempre.overnight.SimpleWorld.superlative)
    (def @countSuperlative edu.stanford.nlp.sempre.overnight.SimpleWorld.countSuperlative)
    (def @countComparative edu.stanford.nlp.sempre.overnight.SimpleWorld.countComparative)
    (def @aggregate edu.stanford.nlp.sempre.overnight.SimpleWorld.aggregate)
    (def @concat edu.stanford.nlp.sempre.overnight.SimpleWorld.concat)
    (def @reverse edu.stanford.nlp.sempre.overnight.SimpleWorld.reverse)
    (def @arithOp edu.stanford.nlp.sempre.overnight.SimpleWorld.arithOp)
    (def @sortAndToString edu.stanford.nlp.sempre.overnight.SimpleWorld.sortAndToString)
    (def @ensureNumericProperty edu.stanford.nlp.sempre.overnight.SimpleWorld.ensureNumericProperty)
    (def @ensureNumericEntity edu.stanford.nlp.sempre.overnight.SimpleWorld.ensureNumericEntity)
    (def @listValue edu.stanford.nlp.sempre.overnight.SimpleWorld.listValue)'''


# lgf2lf 用于将合成的lgf处理成可以直接执行的形式
def lgf2lf(lgf):
    for line in alias.strip().split('\n'):
        _, k, v = line.strip().split()
        lgf = lgf.replace(k, v[:-1])
    lgf = lgf.strip()
    if lgf.startswith('ConstantFn'):
        lgf = lgf.replace('ConstantFn', '').strip()
    if not lgf.startswith('('):
        lgf = '( ' + lgf + ' )'
    lgf = lgf.replace('(', ' ( ').replace(')', ' ) ')
    lgf = ' '.join(lgf.strip().split())
    return lgf


# 复制出多个domain的语法
from copy import deepcopy


def get_domainsRules(rules,
                     domains=[
                         'basketball', 'calendar', 'housing', 'publications',
                         'recipes', 'restaurants', 'blocks', 'socialnetwork'
                     ]):
    domainsRules = []
    for domain in domains:
        domainRules = deepcopy(rules)
        for line in open('/home/wushan/sp/sssp_ws/dataOvernight/' + domain +
                         '.grammar'):
            if '#' in line:
                line = line[:line.find('#')]
            line = line.replace('"(', '[').replace(')"', ']')
            if len(line.strip()) == 0:
                continue
            if line.strip().startswith('(rule'):
                line = line.strip()
                ruleName = line.split()[1]
                l1 = 1 + line[1:].find('(')
                r1 = 1 + line[1:].find(')')
                l2 = r1 + 1 + line[r1 + 1:].find('(')
                r2 = r1 + 1 + line[r1 + 1:-1].rfind(')')
                rs = domainRules.get(ruleName, [])
                nlr = line[l1 + 1:r1]
                lgr = line[l2 + 1:r2]
                # if 'ConstantFn' in lgr:
                #     lgr = lgr.replace('ConstantFn','')
                #     lgr = lgr[lgr.find('(')+1:lgr.rfind(')')].strip()
                rs.append([nlr, lgr])
                domainRules[ruleName] = rs
        domainsRules.append(domainRules)
    return domainsRules


def get_domainRules(domain):
    rules = get_generalRules()

    # 找出general规则中不可扩展的终结符
    terminals = set([])
    for rn, rs in rules.items():
        for r in rs:
            # print (rn,'->',r[0],'|',r[1])
            for w in r[0].strip().split():
                if w.startswith('$'):  # or 'JoinFn' in w:
                    if w in rules:
                        # print('--',w,'--')
                        pass
                    else:
                        terminals.add(w)

    rules = rules_add_vth(rules, terminals)

    # 替换后，增加了一些终结符
    for k in list(rules.keys()):
        if len(rules[k]) == 0:
            del rules[k]
            terminals.add(k)
    # print('terminals', terminals)

    domainsRules = get_domainsRules(rules, [domain])

    # remove unseen nonterminals
    # to-do

    return domainsRules[0]


def removeLeftEmpty(rules, maxdepth=6):
    rs = deepcopy(rules)
    for k, v in rs.items():
        # print(k,v) # $Num,[['two', 'ConstantFn (number 2)']]
        ets = v
        dps = [1 for _ in v]
        while (any([e[0].startswith('$') for e in ets])):
            for i, e in enumerate(ets):
                if e[0].startswith('$'):
                    break
            e = ets.pop(i)
            dp = dps.pop(i)
            #             print(i, e)
            #             exit()
            if dp >= maxdepth:
                # del dps[i]
                # del ets[i] # poped.
                print(e)
                continue
            w = e[0].strip().split()[0]
            if w not in rs:  # such as $Rel0NP 在 recipes 没有可扩展的部分
                continue
            for r in rs[w]:
                e0, e1, _, __ = extendByRule(w, r, e[0], e[1], [], [])
                ets.append([e0, e1])
                dps.append(dp + 1)
        if len(ets) > 0:
            rs[k] = v
        # else:
        #     print(k)
        #     del rs[k]
    return rs


def get_states_transword(rules):
    # rules words: [Nonterminal\tExtendSequence\tExtendLogical,...]
    # states dict: {'w1 w2 ...':[l,k,m], ...}
    # 其中l,k,m表示第l,k,m个rules words的公共子前串（长度为词的数量）
    # transwords[i]: 第i个状态对应的可能后续词
    rules_words = []
    states_dict = {}
    transwords = {}
    for k, v in rules.items():
        for e in v:
            rules_words.append(k + '\t' + e[0] + '\t' + e[1])
            ws = e[0].strip().split()
            for i in range(len(ws)):
                w_ = ' '.join(ws[:i + 1])
                states_dict[w_] = states_dict.get(w_,
                                                  []) + [len(rules_words) - 1]
            for i in range(len(ws) - 1):
                w_ = ' '.join(ws[:i + 1])
                transwords[w_] = transwords.get(w_, []) + [ws[i + 1]]
    return rules_words, states_dict, transwords

def onl2rules(stc, rules_words, states_dict, transwords):
    stateStack = []
    for i in range(len(stc)):
        pass
        
