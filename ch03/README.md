# 3.1 语言模型
如下：
![](https://picgogogo.oss-cn-hangzhou.aliyuncs.com/img/20200106175441.png)
- 数据稀疏， 长度越大的句子越难出现
- 计算代价大， 需要计算的$p(w_t \mid w_0 w_1 ...w_{t-1})$ 就越多


## 3.1.2 马尔科夫链与二元语法

只涉及连续的两个单词，
![](https://picgogogo.oss-cn-hangzhou.aliyuncs.com/img/20200107123655.png)
![](https://picgogogo.oss-cn-hangzhou.aliyuncs.com/img/20200107123726.png)

## 3.1.3 n元语法
每个单词的概率取决于前面n个单词。n大于4时计算代价太大，几乎不使用
![](https://picgogogo.oss-cn-hangzhou.aliyuncs.com/img/20200107123759.png)

## 3.1.4 数据稀疏与平滑策略
为了让语料库中没有出现的单词的概率不为0，使用插值平滑。如果一个单词不存在，那个使用其中某个字的出现概率来平滑这个单词的概率。
最简单的一种平滑策略，插值平滑，这是二元语法：
![](https://picgogogo.oss-cn-hangzhou.aliyuncs.com/img/20200107124226.png)

这个一元语法：
![](https://picgogogo.oss-cn-hangzhou.aliyuncs.com/img/20200107124241.png)


# 3.2 中文分词语料库

## 3.2.1 PKU（1998人民日报语料库）
准确率较低
`icwb2-data/training/pku_training.utf8` `icwb2-data/gold/pku_test_gold.utf8`

## 3.2.2 MSR（微软亚洲研究院语料库）

## 3.2.3 繁体中文语料库
香港城市大学：
- `icwb2-data/training/cityu_train.utf8` `icwb2-data/training/as_train.utf8`
台湾中央研究院：
- `icwb2-data/gold/cityu_test_gold.utf8` `icwb2-data/gold/as_testing_gold.utf8`

## 3.2.4 语料库统计 

|语料库|字符数|词语种数|总词频|平均词长|字符数|词语种数|总词频|平均词长|OOV|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|PKU|183万|6万|111万|1.6|17万|1万|10万|1.7|5.75%|
|MSR|405万|9万|237万|1.7|18万|1万|11万|1.7|2.65%|
|AS|837万|14万|545万|1.5|20万|2万|12万|1.6|4.33%|
|CITYU|240万|7万|146万|1.7|7万|1万|4万|1.7|7.40%|


```python
def count_corpus(train_path: str, test_path: str):
    # counter对象， 总词条数(包括重复), 总字符数
    train_counter, train_freq, train_chars = count_word_freq(train_path)
    test_counter, test_freq, test_chars = count_word_freq(test_path)
    
    # 所有没有未登录词出现的次数之和
    test_oov = sum(test_counter[w] for w in (test_counter.keys() - train_counter.keys()))
    return train_chars / 10000, len(
        train_counter) / 10000, train_freq / 10000, train_chars / train_freq, test_chars / 10000, len(
        test_counter) / 10000, test_freq / 10000, test_chars / test_freq, test_oov / test_freq * 100


def count_word_freq(train_path):
    f = Counter()
    with open(train_path, encoding='utf-8') as src:
        for line in src:
            for word in re.compile("\\s+").split(line.strip()):
                f[word] += 1

    return f, sum(f.values()), sum(len(w) * f[w] for w in f.keys())
```
```python
sighan05='E:/Workspaces/Python/Envs/NLP/Lib/site-packages/pyhanlp/static/data/test/icwb2-data'
print('|语料库|字符数|词语种数|总词频|平均词长|字符数|词语种数|总词频|平均词长|OOV|')
for data in 'pku', 'msr', 'as', 'cityu':
    train_path = os.path.join(sighan05, 'training', '{}_training.utf8'.format(data))
    test_path = os.path.join(sighan05, 'gold',
                             ('{}_testing_gold.utf8' if data == 'as' else '{}_test_gold.utf8').format(data))
    print('|%s|%.0f万|%.0f万|%.0f万|%.1f|%.0f万|%.0f万|%.0f万|%.1f|%.2f%%|' % (
            (data.upper(),) + count_corpus(train_path, test_path)))
```


# 3.3 训练

## 3.3.1 加载语料库
用 `NatureDictionaryMaker` (`DictionaryMaker` & `NGramDictionaryMaker`)，整合了一元语法和二元语法。 `DictionaryMaker`统计一元语法， `NGramDictionaryMaker`统计二元语法

## 3.3.2 统计一元语法


```python
# 加载类
CorpusLoader = SafeJClass('com.hankcs.hanlp.corpus.document.CorpusLoader')
NatureDictionaryMaker = SafeJClass('com.hankcs.hanlp.corpus.dictionary.NatureDictionaryMaker')

sighan05='E:/Workspaces/Python/Envs/NLP/Lib/site-packages/pyhanlp/static/data/test/icwb2-data'
corpus_path = os.path.join(sighan05, 'training', 'msr_training.utf8')
sents = CorpusLoader.convert2SentenceList(corpus_path)
```
```python
def train_bigram(corpus_path, model_path):
    sents = CorpusLoader.convert2SentenceList(corpus_path)
    for sent in sents:
        for word in sent:
            word.setLabel("n")
    
    maker = NatureDictionaryMaker()
    maker.compute(sents)
    # 大坑，只能保存在已有的目录下
    maker.saveTxtTo(model_path)
```
```python
corpus_path = corpus_path
model_path = os.path.join(sighan05, "tests","data", "my_cws_model")
train_bigram(corpus_path, model_path)
```

保存后产生三个文件，一元语法模型 `*.txt`, 二元语法模型 `*.ngram.txt`, 词性标注 `*.tr.txt`

一元模型的例子：
```python
和 n 2
和服 n 1
商品 n 2
始##始 begin 3
服务 n 2
末##末 end 3
物美价廉 n 1
货币 n 1

```

## 3.3.3 统计二元语法
pass

# 3.4 预测
## 3.4.1 加载模型
- `CoreDictionary` : 加载一元语法模型
- `CoreBiGramTableDictionary` : 加载二元语法模型
```python
def  load_bigram(model):
    # 对应一元分词结果
    HanLP.Config.CoreDictionaryPath = model_path + ".txt"
    # 对应二元分词结果
    HanLP.Config.BiGramDictionaryPath = model_path + ".ngram.txt"
    CoreDictionary = SafeJClass("com.hankcs.hanlp.dictionary.CoreDictionary")
    CoreBiGramTableDictionary = SafeJClass("com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary")
    
    # 获得一元分词的结果
    print(CoreDictionary.getTermFrequency("商品"))
    # 获得二元分词的结果
    print(CoreBiGramTableDictionary.getBiFrequency("商品", "和"))
```

## 3.4.2 构建词网
- 将词与词之间建立一个有向图，有向图的边表示两个`p(a|b)`.那么寻找最大概率的句子，就是寻找具有最大权值的路径。
- 给定一个句子，这个句子有多种分词方式，那个所有的分词方式可以画在一个图里面，这个图就是词网。词网中的节点表示词库中存在的词，那么在词网中找一个最优路径，也就是从词库中寻找最优的分词方法（当然也有可能不在词库中，因为有插值平滑）

## 3.4.3 节点间距离计算
使用了两层平滑，外层的平滑策略是拉普拉斯平滑（加一平滑）
![](https://picgogogo.oss-cn-hangzhou.aliyuncs.com/img/20200111010436.png)

考虑到运算的方便，使用常见的取对数操作。因为取了负对数，所以，最大路径和转换成了最小路径和：
![](https://picgogogo.oss-cn-hangzhou.aliyuncs.com/img/20200111010533.png)



## 3.4.4 词图上的维特比算法
一种动态规划算法，可以算出在有向图无环图中的最小路径
维特比算法的实现：
```python
def generate_wordnet(sent, trie):
    """
    生成词网
    :param sent: 句子
    :param trie: 词典（unigram）
    :return: 词网
    """
    searcher = trie.getSearcher(JString(sent), 0)
    wordnet = WordNet(sent)
    while searcher.next():
        wordnet.add(searcher.begin + 1,
                    Vertex(sent[searcher.begin:searcher.begin + searcher.length], searcher.value, searcher.index))
    # 原子分词，保证图连通
    vertexes = wordnet.getVertexes()
    i = 0
    while i < len(vertexes):
        if len(vertexes[i]) == 0:  # 空白行
            j = i + 1
            for j in range(i + 1, len(vertexes) - 1):  # 寻找第一个非空行 j
                if len(vertexes[j]):
                    break
            wordnet.add(i, Vertex.newPunctuationInstance(sent[i - 1: j - 1]))  # 填充[i, j)之间的空白行
            i = j
        else:
            i += len(vertexes[i][-1].realWord)

    return wordnet


def viterbi(wordnet):
    nodes = wordnet.getVertexes()
    # 前向遍历
    for i in range(0, len(nodes) - 1):
        for node in nodes[i]:
            for to in nodes[i + len(node.realWord)]:
                to.updateFrom(node)  # 根据距离公式计算节点距离，并维护最短路径上的前驱指针from
    # 后向回溯
    path = []  # 最短路径
    f = nodes[len(nodes) - 1].getFirst()  # 从终点回溯
    while f:
        path.insert(0, f)
        f = f.getFrom()  # 按前驱指针from回溯
    return [v.realWord for v in path]
```

完整代码:
```python
def train_bigram(corpus_path, model_path):
    sents = CorpusLoader.convert2SentenceList(corpus_path)
    for sent in sents:
        for word in sent:
            word.setLabel("n")
    
    maker = NatureDictionaryMaker()
    maker.compute(sents)
    # 大坑，只能保存在已有的目录下
    maker.saveTxtTo(model_path)
    
def load_bigram(model_path, verbose=True， ret_viterbi=True):
    HanLP.Config.CoreDictionaryPath = model_path + ".txt"  # unigram
    HanLP.Config.BiGramDictionaryPath = model_path + ".ngram.txt"  # bigram

    if verbose:
        print("商品：", CoreDictionary.getTermFrequency("商品"))
        print("商品和", CoreBiGramTableDictionary.getBiFrequency("商品", "和"))
        sent = '商品和服务'
        wordnet = generate_wordnet(sent, CoreDictionary.trie)
        print("词网：\n", wordnet)
        print("维特比路径：\n", viterbi(wordnet))
        
    return ViterbiSegment().enableAllNamedEntityRecognize(False).enableCustomDictionary(
        False) if ret_viterbi else DijkstraSegment().enableAllNamedEntityRecognize(False).enableCustomDictionary(False)
    
train_bigram(corpus_path, model_path)
load_bigram(model_path)
```

实际上，调用已经写好的维特比类很简单：
```python
vite = ViterbiSegment()
vite.seg("商品和服务")

for item in list:
    print(item)
```
```python
商品
和
服务
```

## 3.4.5 与用户词典的集成
用户可以自定义词典
- 低优先级下，首先不考虑用户词典
- 高优先级下，优先考虑用户词典

```python
ViterbiSegment = SafeJClass('com.hankcs.hanlp.seg.Viterbi.ViterbiSegment')

segment = ViterbiSegment()
sentence = "社会摇摆简称社会摇"

segment.enableCustomDictionary(False)
print("不挂载词典：", segment.seg(sentence))

CustomDictionary.insert("社会摇", "nz 100")
segment.enableCustomDictionary(True)

print("低优先级词典：", segment.seg(sentence))

segment.enableCustomDictionaryForcing(True)
print("高优先级词典：", segment.seg(sentence))
```

# 3.5 评测

```python
sighan05='E:/Workspaces/Python/Envs/NLP/Lib/site-packages/pyhanlp/static/data/test/icwb2-data'
msr_dict = os.path.join(sighan05, 'gold', 'msr_training_words.utf8')
msr_train = os.path.join(sighan05, 'training', 'msr_training.utf8')
msr_model = os.path.join(test_data_path(), 'msr_cws')
msr_test = os.path.join(sighan05, 'testing', 'msr_test.utf8')
msr_output = os.path.join(sighan05, 'testing', 'msr_bigram_output.txt')
msr_gold = os.path.join(sighan05, 'gold', 'msr_test_gold.utf8')


train_bigram(msr_train, msr_model)  # 训练
segment = load_bigram(msr_model, verbose=False)  # 加载

result = CWSEvaluator.evaluate(segment, msr_test, msr_output, msr_gold, msr_dict)  # 预测打分
print(result)
```