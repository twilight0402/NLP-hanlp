# 词典

加载词典
```python
def load_dictionary():
    """
    加载HanLP中的mini词库
    :return: 一个set形式的词库
    """
    IOUtil = JClass('com.hankcs.hanlp.corpus.io.IOUtil')
    # 换成一个较小的词库
    path = HanLP.Config.CoreDictionaryPath.replace('.txt', '.mini.txt')
    dic = IOUtil.loadDictionary([path])
    return set(dic.keySet())


dic = load_dictionary()
print(len(dic))
print(list(dic)[0])
```

# 切分算法
## 完全切分
```python
# 完全切分
def fully_segment(text, dic):
    word_list = []
    # [0, len - 1]
    for i in range(len(text)):                  # i 从 0 到text的最后一个字的下标遍历
        # [i+1, len + 1)
        for j in range(i + 1, len(text) + 1):   # j 遍历[i + 1, len(text)]区间
            # [i,j)
            word = text[i:j]                    # 取出连续区间[i, j]对应的字符串
            if word in dic:                     # 如果在词典中，则认为是一个词
                word_list.append(word)
    return word_list


dic = load_dictionary()
print(fully_segment('商品和服务', dic))
```

## 最长匹配算法
### 正向最长匹配
```
i----------j
i=0
i不动， j从 i+1 ==》 size
每次选最长的ij(在字典中的)
i+=len(ij)
```
```python
# 正向最长匹配
def forward_segment(text, dic):
    word_list = []
    i = 0
    while i < len(text):
        longest_word = text[i]                      # 当前扫描位置的单字
        for j in range(i + 1, len(text) + 1):       # 所有可能的结尾
            word = text[i:j]                        # 从当前位置到结尾的连续字符串
            if word in dic:                         # 在词典中
                if len(word) > len(longest_word):   # 并且更长
                    longest_word = word             # 则更优先输出
        word_list.append(longest_word)              # 输出最长词
        i += len(longest_word)                      # 正向扫描
    return word_list
```

### 逆向最长匹配
```
j----------i
i=size-1
i不动， j从 0 ==》 i-1
每次选最长的ij
i-=len(ij)
```

```python
# 逆向最长匹配
def backward_segment(text, dic):
    word_list = []
    i = len(text) - 1
    while i >= 0:                                   # 扫描位置作为终点
        longest_word = text[i]                      # 扫描位置的单字
        for j in range(0, i):                       # 遍历[0, i]区间作为待查询词语的起点
            word = text[j: i + 1]                   # 取出[j, i]区间作为待查询单词
            if word in dic:
                if len(word) > len(longest_word):   # 越长优先级越高
                    longest_word = word
                    break
        word_list.insert(0, longest_word)           # 逆向扫描，所以越先查出的单词在位置上越靠后
        i -= len(longest_word)
    return word_list
```

### 双向最长匹配
- 同时执行正向和逆向最长匹配，若两者的词数不同，则返回词数更少的那一个。
- 否则，返回两者中单字更少的那一个。当单字数也相同时，优先返回逆向最长匹配的结果。

```python
# 双向最长匹配
def bidirectional_segment(text, dic):
    f = forward_segment(text, dic)
    b = backward_segment(text, dic)
    if len(f) < len(b):                                  # 词数更少优先级更高
        return f
    elif len(f) > len(b):
        return b
    else:
        if count_single_char(f) < count_single_char(b):  # 单字更少优先级更高
            return f
        else:
            return b                                     # 都相等时逆向匹配优先级更高
```


# 字典树（前缀树）
用每个字作为一条路径（节点之间的边）

## 首字散列的二分字典树
首字使用散列表，其余节点使用二分查找 BinTree

## 双数组字典树
`DoubleArrayTrie` `parseText()` 全切分 `parseLongestText()`正向最长切分

# AC自动机
- `goto表(success表)` : 一个前缀树
- `output表` : 表示哪些状态对应的字符串可以输出（存在于语料库中）。当前路径的后缀中，合法的后缀也会被加入到output表中（比如she， 和 he）
- `fail表` : 状态转移失败后，应当回退的最佳状态.（回退到已匹配字符串的最长后缀上）

```python
from pyhanlp import *
words = ["hers", "his", "she", "he"]
ACTrie = JClass('com.hankcs.hanlp.algorithm.ahocorasick.trie.Trie')
trie = ACTrie()
for word in words:
    trie.addKeyword(word)

for emit in trie.parseText("ushers"):
    print("[%d:%d]=%s" % (emit.getStart(), emit.getEnd(), emit.getKeyword()))
```

# 双数组字典树AC自动机
```python
# 双数组字典树AC自动机
words = ["hers", "his", "she", "he"]
map = JClass('java.util.TreeMap')()
for word in words:
    map[word] = word.upper()

acdTrie = JClass('com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie')(map)

for emit in acdTrie.parseText("ushers"):
    print("[%d:%d]=%s" % (emit.begin, emit.end, emit.value))
```

# Hanlp的词典分词实现
![](https://picgogogo.oss-cn-hangzhou.aliyuncs.com/img/20200106115726.png)

## DoubleArrayTrieSegment 双数组字典树
```python
HanLP.Config.ShowTermNature = True
segment = DoubleArrayTrieSegment()
print(segment.seg("江西鄱阳湖干枯，中国最大的淡水湖变成大草原"))

## 传入自己的词典
dict1 = HANLP_DATA_PATH +  "/dictionary/CoreNatureDictionary.mini.txt"
dict2 = HANLP_DATA_PATH + "/DICTIONARY/CUSTOM/上海地名.txt ns"

segment = DoubleArrayTrieSegment([dict1, dict2])
print(segment.seg("上海市虹口区大连西路550号SISU"))

###  enablePartOfSpeechTagging()数字合并和词性标注功能合并，只有打开这个才能看到正确的词性
segment.enablePartOfSpeechTagging(True)    # 激活数字和英文识别
HanLP.Config.ShowTermNature = False
print(segment.seg("上海市虹口区大连西路550号SISU"))

## 遍历分词结果
segment.enablePartOfSpeechTagging(True)
HanLP.Config.ShowTermNature = True
for term in segment.seg("上海市虹口区大连西路550号SISU") :
    print("%s %s" % (term.word, term.nature))
```
## AhoCorasickDoubleArrayTrieSegment 双数组AC自动机
如果用户的词语都很长，使用AC自动机会更快
```python
segment = JClass('com.hankcs.hanlp.seg.Other.AhoCorasickDoubleArrayTrieSegment')
segment = segment()
segment.enablePartOfSpeechTagging(True)

for item in segment.seg("江西鄱阳湖干枯，中国最大的淡水湖变成大草原"):
    print(item.word, item.nature)
```
# 2.9 准确度评测
## 1.混淆矩阵
P(Positive), N(negative)

|预测**\**答案|P|N|
|:--:|:--:|:--:|
|P|TP|FP|
|N|FN|TN|

## 2. 精确度
预测出来的P的正确率
$$
P = \frac{TP}{TP+FP}
$$

$$
P = \frac{正确预测的阳(P)的数量}{预测结果为阳(P)的数量}
$$
## 3. 召回率
正确预测的P占所有P的比率
$$
R = \frac{TP}{TP+FN}
$$

$$
P = \frac{正确预测的阳(P)的数量}{所有的阳(P)的数量}
$$

## 4. F1值

$$
F_1 = \frac{2 \cdot P \cdot R}{P + R}
$$

## 5. OOV & IV
- **OOV(Out Of Vocabulary)** : 未登录词， 词典未收录的词
- **IV(In vocabulary)** : 登陆词，词典已有的词

```

def to_region(segmentation: str) -> list:
    """
    将分词结果转换为区间
    :param segmentation: 商品 和 服务
    :return: [(0, 2), (2, 3), (3, 5)]
    """
    region = []
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):
        end = start + len(word)
        region.append((start, end))
        start = end
    return region

# set & set 交集； set | set 并集； set - set 差集
def prf(gold: str, pred: str, dic) -> tuple:
    """
    计算P、R、F1
    :param gold: 标准答案文件，比如“商品 和 服务”
    :param pred: 分词结果文件，比如“商品 和服 务”
    :param dic: 词典
    :return: (P, R, F1, OOV_R, IV_R)
    """
    A_size, B_size, A_cap_B_size, OOV, IV, OOV_R, IV_R = 0, 0, 0, 0, 0, 0, 0
    with open(gold, encoding="utf-8") as gd, open(pred, encoding="utf-8") as pd:
        for g, p in zip(gd, pd):
            A, B = set(to_region(g)), set(to_region(p))
            A_size += len(A)
            B_size += len(B)
            A_cap_B_size += len(A & B)
            text = re.sub("\\s+", "", g)
            for (start, end) in A:
                word = text[start: end]
                if dic.containsKey(word):
                    IV += 1
                else:
                    OOV += 1

            for (start, end) in A & B:
                word = text[start: end]
                if dic.containsKey(word):
                    IV_R += 1
                else:
                    OOV_R += 1
    p, r = A_cap_B_size / B_size * 100, A_cap_B_size / A_size * 100
    return p, r, 2 * p * r / (p + r), OOV_R / OOV * 100, IV_R / IV * 100



print(to_region('商品 和 服务'))

sighan05 = ensure_data('icwb2-data', 'http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip')
## 词汇表
msr_dict = os.path.join(sighan05, 'gold', 'msr_training_words.utf8')
## 没有分词的文章
msr_test = os.path.join(sighan05, 'testing', 'msr_test.utf8')
## 以下两个是分词的结果
msr_output = os.path.join(sighan05, 'testing', 'msr_output.txt')
msr_gold = os.path.join(sighan05, 'gold', 'msr_test_gold.utf8')

## 训练
DoubleArrayTrieSegment = JClass('com.hankcs.hanlp.seg.Other.DoubleArrayTrieSegment')
segment = DoubleArrayTrieSegment([msr_dict]).enablePartOfSpeechTagging(True)

## re.sub("\\s+", "", line) 删除所有空白字符
with open(msr_gold, encoding="utf-8") as test, open(msr_output, 'w', encoding="utf-8") as output:
    for line in test:
        output.write("  ".join(term.word for term in segment.seg(re.sub("\\s+", "", line))))
        output.write("\n")
print("P:%.2f R:%.2f F1:%.2f OOV-R:%.2f IV-R:%.2f" % prf(msr_gold, msr_output, segment.trie))

```

# 字典树的其他应用
## 停用词过滤
```
# 加载双数组字典树
def load_from_file(path):
    map = JClass("java.util.TreeMap")()
    with open(path, encoding="utf-8") as src:
        for word in src:
            word = word.strip()
            map[word] = word
    return JClass("com.hankcs.hanlp.collection.trie.DoubleArrayTrie")(map)

def load_from_words(*words):
    map = JClass("java.util.TreeMap")()
    for word in words:
        map[word] = word
    return JClass("com.hankcs.hanlp.collection.trie.DoubleArrayTrie")(map)

def remove_stopwords_termlist(termlist, trie):
    return [term.word for term in termlist if not trie.containsKey(term.word)]

def replace_stropwords_text(text, replacement, trie):
    searcher = trie.getLongestSearcher(JString(text), 0)
    offset = 0
    result = ''
    while searcher.next():
        begin = searcher.begin
        end = begin + searcher.length
        if begin > offset:
            result += text[offset: begin]
        result += replacement
        offset = end
    if offset < len(text):
        result += text[offset]
    return result
```

```
HanLP.Config.ShowTermNature = False
trie = load_from_file(HanLP.Config.CoreStopWordDictionaryPath)

text = "停用词的意义相对而言无关紧要吧。"
segment = DoubleArrayTrieSegment()
termlist = segment.seg(text)

print("分词结果：", termlist)
print("分词结果去除停用词：", remove_stopwords_termlist(termlist, trie))
trie = load_from_words("的", "相对而言", "吧")
print("不分词去掉停用词", replace_stropwords_text(text, "**", trie))
```
## 繁简转换
简体s，繁体t，台湾tw， 香港hk
```
## 繁简转换的朴素实现
CharTable = JClass("com.hankcs.hanlp.dictionary.other.CharTable")

print(CharTable.convert("自然語言處理"))
HanLP.convertToTraditionalChinese("自然语言处理")
HanLP.convertToSimplifiedChinese("自然語言處理")

print(HanLP.s2tw("在台湾写代码"))
print(HanLP.tw2s("在臺灣寫程式碼"))
print(HanLP.s2hk("在台湾写代码"))
print(HanLP.hk2s("在臺灣寫代碼"))
print(HanLP.hk2tw("在臺灣寫代碼"))
```