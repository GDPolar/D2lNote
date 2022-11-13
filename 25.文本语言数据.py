import collections
import random
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  
    """将"时间机器"数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 为简单起见，只留下所有的小写字母
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}') # 文本总行数: 3221
print(lines[0]) # the time machine by h g wells
print(lines[10]) # twinkled and his usually pale face was flushed and animated the

def tokenize(lines, token='word'):  
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)

print(tokens[10])
# ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']


def count_corpus(tokens):  
    """统计 token 的频率"""
    # 这里的 tokens 是 1D 列表或 2D 列表
    # len(tokens) == 0 防止 token[0] 不存在导致报错
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# 将 token 映射到数字
class Vocab:  
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                                    for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            # 频次小于阈值，抛弃
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
# [('<unk>', 0), ('the', 1), ('i', 2), ('and', 3), ('of', 4), ('a', 5), ('to', 6), ('was', 7), ('in', 8), ('that', 9)]

for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])
# 文本: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
# 索引: [1, 19, 50, 40, 2183, 2184, 400]
# 文本: ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
# 索引: [2186, 3, 25, 1044, 362, 113, 7, 1421, 3, 1045, 1]


# 将上述函数打包
def load_corpus_time_machine(max_tokens=-1): 
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
# (170580, 28)

vocab.token_freqs[:10]
# [('the', 2261),
# ('i', 1267),
# ('and', 1245),
# ('of', 1155),
# ('a', 816),
# ('to', 695),
# ('was', 552),
# ('in', 541),
# ('that', 443),
# ('my', 440)]

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
#[(('of', 'the'), 309),
# (('in', 'the'), 169),
# (('i', 'had'), 130),
# (('i', 'was'), 112),
# (('and', 'the'), 109),
# (('the', 'time'), 102),
# (('it', 'was'), 99),
# (('to', 'the'), 85),
# (('as', 'i'), 78),
# (('of', 'a'), 73)]

def seq_data_iter_random(corpus, batch_size, num_steps):  
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括 num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，因为要给最后一个子序列留个标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为 num_steps 的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # random.shuffle() 打乱列表
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从 pos 位置开始的长度为 num_steps 的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
# 每组 X 独立
# X:  tensor([[21, 22, 23, 24, 25],
#         [11, 12, 13, 14, 15]]) 
# Y:  tensor([[22, 23, 24, 25, 26],
#         [12, 13, 14, 15, 16]])
# X:  tensor([[ 1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10]]) 
# Y:  tensor([[ 2,  3,  4,  5,  6],
#         [ 7,  8,  9, 10, 11]])
# X:  tensor([[16, 17, 18, 19, 20],
#         [26, 27, 28, 29, 30]]) 
# Y:  tensor([[17, 18, 19, 20, 21],
#         [27, 28, 29, 30, 31]])


def seq_data_iter_sequential(corpus, batch_size, num_steps):  
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
# 相邻的 X 相邻
# X:  tensor([[ 4,  5,  6,  7,  8],
#         [19, 20, 21, 22, 23]]) 
# Y: tensor([[ 5,  6,  7,  8,  9],
#         [20, 21, 22, 23, 24]])
# X:  tensor([[ 9, 10, 11, 12, 13],
#         [24, 25, 26, 27, 28]]) 
# Y: tensor([[10, 11, 12, 13, 14],
#         [25, 26, 27, 28, 29]])
# X:  tensor([[14, 15, 16, 17, 18],
#         [29, 30, 31, 32, 33]]) 
# Y: tensor([[15, 16, 17, 18, 19],
#         [30, 31, 32, 33, 34]])


# 将上面的两个采样函数包装到一个类中
class SeqDataLoader:  
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,  
                           use_random_iter=False, max_tokens=10000):
    """返回《时光机器》数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab