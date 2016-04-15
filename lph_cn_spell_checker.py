# -*- coding:utf-8 -*-
__author__ = 'liupenghe'

import os
import collections

import jieba

from sxpCi import ci_list

def cn_ci(dir_path):
    for rdf in ci_list:
        jieba.add_word(rdf[0])
    all_text = u""
    for file_name in os.listdir(dir_path):
        if file_name.find(".txt") != -1:
            file_path = "/".join([dir_path, file_name])
            with open(file_path, "r") as f:
                all_text += f.read().decode("utf-8")

    terms = jieba.cut(all_text)

    return [ci for ci in ','.join(terms).split(',') if ci not in [u'', u" "]]



def cn_train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model




CNWORDS = cn_train(cn_ci("cn_texts"))

def cn_hanzi():
    with open("hanzi.txt", "r") as f:
        hanzi = f.read().decode("utf-8")
        return hanzi

cn_hanzi_str = cn_hanzi()

def cn_edits1(ci):
    splits     = [(ci[:i], ci[i:]) for i in range(len(ci)   + 1)]
    print splits
    deletes    = [a + b[1:] for a, b in splits if b if a + b[1:] in CNWORDS]
    print deletes
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1 if a + b[1] + b[0] + b[2:] in CNWORDS]
    print transposes
    replaces   = [a + c + b[1:] for a, b in splits for c in cn_hanzi_str if b if a + c + b[1:] in CNWORDS]
    print replaces
    inserts    = [a + c + b     for a, b in splits for c in cn_hanzi_str if a + c + b in CNWORDS]
    print inserts
    return set(deletes + transposes + replaces + inserts)


def cn_known_edits2(ci):
    return set(e2 for e1 in cn_edits1(ci) for e2 in cn_edits1(e1) if e2 in CNWORDS)


def cn_correct(ci):
    candidates = cn_edits1(ci) or cn_known_edits2(ci)
    return max(candidates, key=CNWORDS.get)

print len(CNWORDS.items()[0][0])


cn_edits1(u"咳树")

print cn_correct(u"传然")