
# 中文词自动纠错讲解
## 本文档主要演示如何通过python实现一个中文词组的自动纠错，如输入“咳数”，输出“咳嗽”
## 程序原理：
### 给定一待纠错词w,我们需要从一系列候选词中选出一最可能的。也就是：argmax(p(c|w)), c in 候选词表。根据贝叶斯原理，p(c|w) = p(w|c) * p(c) / p(w). 又对任意可能的c,p(w)一样，故也就是求使argmax(p(w|c) * p(c))成立的c.



```python
# -*- coding:utf-8 -*-
__author__ = 'liupenghe'

import os
import collections
import jieba
from sxpCi import ci_list
```

### 初始化所有潜在中文词的先验概率 文本集：50篇医学文章分词后，统计各个中文词的出现频率即为其先验概率


```python
#对给定的语料库分词
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


#统计语料库中各个单词出现的概率
def cn_train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model
CNWORDS = cn_train(cn_ci("cn_texts"))
```

    Building prefix dict from /Library/Python/2.7/site-packages/jieba/dict.txt ...
    Dumping model to file cache /var/folders/30/cd4n0jcj4_1gnjml7xh8nf500000gn/T/jieba.cache
    Loading model cost 1.939 seconds.
    Prefix dict has been built succesfully.


### CNWORDS 即为我们的单词模型，其为字典格式， key为单词，value为该单词先验概率（词频）。另外，为了弥补可能出现的新词，我们做了平滑处理，对新词默认其出现频率为1.


```python
# 查看其中的3个验证一下
for w, wf in CNWORDS.items()[:3]:
    print w, wf
```

    第二 27
    小腿 26
    编译 16


### 当给定一待纠错单词时，我们需要找出可能的正确单词列表。这里我们根据字符距离来找出可能的正确单词列表，我们来回顾一下两个单词之间的字符距离，如果一个单词转变为另一个单词需要编辑n下，如删掉一个字符，替换一个字符，交换两个字符位置，增减一个字符，那么我们就说这两个单词间的字符距离为n。考虑到中文的特殊性，这里我将中文的一个字看成一个字符，当然中文字符会比26个英文字母要多的多。


```python
#载入所有中文字
def cn_hanzi():
    with open("hanzi.txt", "r") as f:
        hanzi = f.read().decode("utf-8")
        return hanzi

cn_hanzi_str = cn_hanzi()
```


```python
#根据字符距离构造可能的单词列表，这里只计算与待检查单词字符距离为1的单词
def cn_edits1(ci):
    splits     = [(ci[:i], ci[i:]) for i in range(len(ci)   + 1)]
    # 比待检查单词少一个字的单词
    deletes    = [a + b[1:] for a, b in splits if b if a + b[1:] in CNWORDS] 
    # 交换待检查单词中的任意相邻两个字的位置
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1 if a + b[1] + b[0] + b[2:] in CNWORDS] 
    # 使用所有中文字替换待检查字中的某个字
    replaces   = [a + c + b[1:] for a, b in splits for c in cn_hanzi_str if b if a + c + b[1:] in CNWORDS]
    # 向待检查词中插入新字
    inserts    = [a + c + b     for a, b in splits for c in cn_hanzi_str if a + c + b in CNWORDS]
    return set(deletes + transposes + replaces + inserts)
```

### 由于中文字有5000多个，因而由字符距离1来构造出来的可能候选单词列表将会很大，因而我们对构造出来的单词做了一次验证后再将其加入候选集合中，即我们判断了下该词是否为有效单词，根据其是否在我们的单词模型中。

### 如果觉得只考虑单词距离为1的单词数量不够用，我们可以继续加入与待检查单词距离为2的单词


```python
def cn_known_edits2(ci):
    return set(e2 for e1 in cn_edits1(ci) for e2 in cn_edits1(e1) if e2 in CNWORDS)
```

### 到这一步，我们的模型基本构造完成，完成最后的修改函数吧。


```python
def cn_correct(ci):
    # 候选词列表
    candidates = cn_edits1(ci) or cn_known_edits2(ci)
    # 找出其中概率最大的单词
    return max(candidates, key=CNWORDS.get)
```

### 测试一下吧


```python
print cn_correct(u'咳数')
```

    咳嗽



```python
print cn_correct(u'呕土')
```

    呕吐



```python
print cn_correct(u'传然')
```

    虽然



```python
print cn_correct(u'感帽')
```

    感染


### 从上面的结果我们可以看出来，程序可以将我们打错的单词自动修改成正确的单词。但不难发现，程序仍然存在问题，如我们打“传然”时，我们的本意可能是“传染”，然而程序却改成了“虽然”；我们打“感帽”我们的本意是“感冒”，却被修改为了“感染”。因而考虑从以下几个方向改进：
- 1,考虑人们的打字习惯，人们通常越往后字打错的可能越大，因而可以考虑每个字在单词中的位置给予一定权重，这中方法有助于改进上面的第一种“传然”－ "虽然"的情况；
- 2,考虑拼音的重要性，对汉语来讲，通常人们打错时拼音是拼对的，只是选择时候选择错了，因而对候选词也可以优先选择同拼音的字。

参考资料：http://norvig.com/spell-correct.html  这时google大牛写的英文单词的自动拼写，本程序主要参考其代码实现。
