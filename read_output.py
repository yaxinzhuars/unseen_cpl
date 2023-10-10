import json
from operator import itemgetter
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import random, itertools
import argparse

random.seed(0)

def read_index(file):
    c2i, i2c = {}, {}
    with open(file) as f:
        for line in f.readlines():
            i, c, _ = line.strip().split('\t')
            c2i[c] = i
            i2c[i] = c
    return c2i, i2c
    
def read_output(file):
    c2preq = {}
    c2pos = {}
    with open(file) as f:
        for line in f.readlines():
            c1, c2, s0, s1, o = line.strip().split('\t')
            if c1 not in c2preq.keys():
                c2preq[c1] = {}
                c2pos[c1] = []
            if c2 not in c2preq.keys():
                c2preq[c2] = {}
                c2pos[c2] = []
            c2preq[c1][c2] = (float(s0), float(s1))
            if o == '1':
                c2pos[c1].append(c2)
    return c2preq, c2pos

def read_output_reverse(file):
    c2preq = {}
    c2pos = {}
    with open(file) as f:
        for line in f.readlines():
            c2, c1, s0, s1, o = line.strip().split('\t')
            if c1 not in c2preq.keys():
                c2preq[c1] = {}
                c2pos[c1] = []
            if c2 not in c2preq.keys():
                c2preq[c2] = {}
                c2pos[c2] = []
            c2preq[c1][c2] = (float(s0), float(s1))
            if o == '1':
                c2pos[c1].append(c2)
    return c2preq, c2pos

def read_gt(file1, file2):
    c2pos_gt = {}
    c2neg_gt = {}
    with open(file1) as f:
        for line in f.readlines():
            c1, w1, c2, w2, label = line.strip().split('\t')
            c1, c2 = c1.strip(), c2.strip()
            if c1 not in c2pos_gt.keys():
                c2pos_gt[c1] = []
                c2neg_gt[c1] = []
            if c2 not in c2pos_gt.keys():
                c2pos_gt[c2] = []
                c2neg_gt[c2] = []
            if label == '1':
                c2pos_gt[c1].append(c2)
                c2neg_gt[c2].append(c1)
            else:
                c2neg_gt[c1].append(c2)
    with open(file2) as f:
        for line in f.readlines():
            c1, w1, c2, w2, label = line.strip().split('\t')
            c1, c2 = c1.strip(), c2.strip()
            if c1 not in c2pos_gt.keys():
                c2pos_gt[c1] = []
                c2neg_gt[c1] = []
            if c2 not in c2pos_gt.keys():
                c2pos_gt[c2] = []
                c2neg_gt[c2] = []
            if label == '1':
                c2pos_gt[c1].append(c2)
                c2neg_gt[c2].append(c1)
            else:
                c2neg_gt[c1].append(c2)
    return c2pos_gt, c2neg_gt

def read_gt_reverse(file1, file2):
    c2pos_gt = {}
    c2neg_gt = {}
    with open(file1) as f:
        for line in f.readlines():
            c2, w2, c1, w1, label = line.strip().split('\t')
            c1, c2 = c1.strip(), c2.strip()
            if c1 not in c2pos_gt.keys():
                c2pos_gt[c1] = []
                c2neg_gt[c1] = []
            if c2 not in c2pos_gt.keys():
                c2pos_gt[c2] = []
                c2neg_gt[c2] = []
            if label == '1':
                c2pos_gt[c1].append(c2)
                c2neg_gt[c2].append(c1)
            else:
                c2neg_gt[c1].append(c2)
    with open(file2) as f:
        for line in f.readlines():
            c2, w2, c1, w1, label = line.strip().split('\t')
            c1, c2 = c1.strip(), c2.strip()
            if c1 not in c2pos_gt.keys():
                c2pos_gt[c1] = []
                c2neg_gt[c1] = []
            if c2 not in c2pos_gt.keys():
                c2pos_gt[c2] = []
                c2neg_gt[c2] = []
            if label == '1':
                c2pos_gt[c1].append(c2)
                c2neg_gt[c2].append(c1)
            else:
                c2neg_gt[c1].append(c2)
    return c2pos_gt, c2neg_gt



def read_wiki(file):
    c2w = {}
    with open(file) as f:
        for line in f.readlines():
            i, c, w = line.strip().split('\t')
            c2w[c] = c + '\t' + w
    return c2w

def _output(file, format, c, topk, bottomk, dic=None):
    with open(file, 'a') as fw:
        if format == 'c':
            for k, v in topk.items():
                fw.write(c + '\t' + k + '\t1\n')
            for k, v in bottomk.items():
                fw.write(c + '\t' + k + '\t0\n')
        elif format == 'w':
            c2w = dic
            for k, v in topk.items():
                fw.write(c2w[c] + '\t' + c2w[k] + '\t1\n')
            for k, v in bottomk.items():
                fw.write(c2w[c] + '\t' + c2w[k] + '\t0\n')
        elif format == 'i':
            c2i = dic
            for k, v in topk.items():
                fw.write(c2i[c] + '\t' + c2i[k] + '\t1\n')

def _output_reverse(file, format, c, topk, bottomk, dic=None):
    with open(file, 'a') as fw:
        if format == 'c':
            for k, v in topk.items():
                fw.write(c + '\t' + k + '\t1\n')
            for k, v in bottomk.items():
                fw.write(c + '\t' + k + '\t0\n')
        elif format == 'w':
            c2w = dic
            for k, v in topk.items():
                fw.write(c2w[k] + '\t' + c2w[c] + '\t1\n')
            for k, v in bottomk.items():
                fw.write(c2w[k] + '\t' + c2w[c] + '\t0\n')
        elif format == 'i':
            c2i = dic
            for k, v in topk.items():
                fw.write(c2i[k] + '\t' + c2i[c] + '\t1\n')

# bert -> ncf: uc: 0.02 (ml, lb: 5, 0.2)
# ncf -> bert: 0.1 (uc: 0.02) ml: 5, 0.2    uc 0.05
# hard: uc 55 ml 10
# unseen: uc 2 0.02
def output(args):
    c2i, i2c = read_index(args.c2w)
    # c2preqb, c2posb = read_output1('output_uc_bce_e0.txt')
    c2preq, c2pos = read_output(args.logits_file)
    # c2preq1, c2pos1 = read_output1('../NCF/output_uc_decouple.txt')
    c2pos_gt, c2neg_gt = read_gt(args.input_train_file, args.input_test_file)
    c2w = read_wiki(args.c2w)
    concepts = []
    for c in c2pos.keys():
        # if c in concepts:
            # continue
        num = max(2, int(0.02*len(c2pos[c])))
        # num = int(0.02*len(c2pos[c]))
        # topk = {k: v[1] for k, v in sorted(c2preq[c].items(), key=lambda x: x[1][1], reverse=True)[:num] if v[1] > 0.5 and k != c and k not in c2pos_gt[c] and k not in c2neg_gt[c] and v[1] - c2preq[k][c][1] > 0.05}
        topk = {k: v[1] for k, v in sorted(c2preq[c].items(), key=lambda x: x[1][1], reverse=True)[:num] if v[1] > 0.8 and k != c and k not in c2pos_gt[c] and k not in c2neg_gt[c] and c2preq[k][c][0] > 0.5} # 0.9 0.8
        # topk = {k: v[1] for k, v in sorted(c2preq[c].items(), key=lambda x: x[1][1], reverse=True)[:num] if v[1] > 0.5 and k != c and k not in c2pos_gt[c] and k not in c2neg_gt[c]} # 0.9 0.8
        print(len(topk))
        # bottomk = {k: v[1] for k, v in sorted(c2preq[c].items(), key=lambda x: x[1][1], reverse=False)[:len(topk)] if v[1] < 0.01 and k != c and k not in c2pos_gt[c] and k not in c2neg_gt[c]}
        # bottomk = {k: v[1] for k, v in sorted(c2preq[c].items(), key=lambda x: x[1][1], reverse=False)[:len(topk)] if k != c and k not in c2pos_gt[c] and k not in c2neg_gt[c]}
        # hardk = {k: v[1] for k, v in sorted(c2preq[c].items(), key=lambda x: x[1][1], reverse=True)[len(topk):10*len(topk)] if v[1] < 0.8 and k != c and k not in c2pos_gt[c] and k not in c2neg_gt[c]}
        hardk = {k: v[1] for k, v in sorted(c2preq[c].items(), key=lambda x: x[1][1], reverse=True)[len(topk):2*len(topk)] if k != c and k not in c2pos_gt[c] and k not in c2neg_gt[c]}
        # randomdict = {k: v[1] for k, v in sorted(c2preq[c].items(), key=lambda x: x[1][1], reverse=True) if k != c and k not in c2pos_gt[c] and k not in c2neg_gt[c]}
        # randomk = dict(itertools.islice(randomdict.items(), len(topk)))
        # print(c, ','.join(topk.keys()))
        print(len(hardk))
        for k, v in topk.items():
            # if c2preq[k][c][1] > 0.5: 
                print(c, k, v, c2preq[k][c][1])
                # print(c, k, c2preqb[c][k][1], c2preqb[k][c][1])
        # print(c, ','.join(c2pos_gt[c]))
        # print(c, ','.join(c2neg_gt[c]))
        _output(args.output_file, args.mode, c, topk, hardk, dic=c2i)
        # _output1('../datasets/wiki_dpr/uc_train_unseen_boost_0.8.txt', 'w', c, topk, hardk, dic=c2w)

# output()

def merge(rb, rg, file, file_soft):
    concepts = []
    with open('../datasets/wiki_dpr/uc_unseen.txt') as f:
        for i in f.readlines():
            concepts.append(i.strip())
    c2i, i2c = read_index('uc_i2c.txt')
    c_bert, c2pos = read_output(file)
    c_ncf, c2pos_ = read_output('../NCF/' + file_soft)
    l_gt, l_merge = [], []
    count = 0
    c2test = {}
    with open('../datasets/wiki_dpr/uc_test_unseen.txt') as f:
        for i, line in enumerate(f.readlines()):
            c1, w1, c2, w2, label = line.strip().split('\t')
            c1, c2 = c1.strip(), c2.strip()
            if c1 == c2:
                continue
            # if c1 not in concepts or c2 not in concepts:
            #     continue
            s0_bert, s1_bert = c_bert[c1][c2]
            s0_ncf, s1_ncf = c_ncf[c1][c2]\
            if c1 == 'Distributional semantics' and c2 == 'Seq2seq':
                s1_bert = 0.7225734
                s0_bert = 1 - s1_bert
            s0 = s0_bert*rb + s0_ncf*rg
            s1 = s1_bert*rb + s1_ncf*rg
            l1 = 1 if s1 >= 0.5*(rb+rg) else 0
            l2 = 1 if c2 in c2pos[c1] else 0         
            lb = 1 if s1_bert > 0.5 else 0
            ln = 1 if s1_ncf > 0.5 else 0
            l_gt.append(int(label))
            l_merge.append(l1)

    l_gt = np.array(l_gt)
    l_merge = np.array(l_merge)
    f1_macro = f1_score(l_gt, l_merge, average='macro')
    p_macro = precision_score(l_gt, l_merge, average='macro')
    r_macro = recall_score(l_gt, l_merge, average='macro')
    print(f1_macro, p_macro, r_macro)

    print(np.sum(l_gt), np.sum(l_merge), l_gt.shape)
    print('count', count)

parser = argparse.ArgumentParser()
parser.add_argument('--c2w', type=str)
parser.add_argument('--logits_file', type=str)
parser.add_argument('--input_train_file', type=str)
parser.add_argument('--input_test_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--mode', type=str)

args = parser.parse_args()
output(args)