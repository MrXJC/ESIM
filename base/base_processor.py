import os
import pickle
from pathlib import Path
import numpy as np
from utils.tokenizer import Segment_jieba
from utils.vocab import Vocab
from utils.w2v import Embedding
from utils import load_from_pickle, dump_to_pickle
from abc import abstractmethod


class BaseProcessor:
    def __init__(self, logger, config, data_name, data_path, embed_path = None, user_dict = None, vocab_path = None, stop_word=None, max_len = 50, query_max_len=20,
                 target_max_len=20, test_split=0.0, training=True):
        self.logger = logger
        self.reset = config.reset
        self._data_dir = Path('data') / data_name

        self.query_max_len = query_max_len
        self.target_max_len = target_max_len
        self.max_len = max_len

        if training:
            embedding_path = self._data_dir / embed_path
            print(embedding_path.absolute())
            self._embedding = Embedding(str(embedding_path), logger=logger)

        print(f"Begin to build segment and ..... feature engnieer .... ngram .....")
        self._segment = Segment_jieba(user_dict=str(self._data_dir / user_dict))

        if training:
            print(f"Begin to build vocab")
            self._vocab = Vocab(str(self._data_dir / 'RAW' / vocab_path), self._segment, self._embedding)
            self.word2idx, self.idx2word = self._vocab.word2idx, self._vocab.idx2word
            dump_to_pickle(str(self._data_dir / 'vocab.pkl'), (self.word2idx, self.idx2word), self.reset)
        else:
            print(f"load the vocab")
            (self.word2idx, self.idx2word) = load_from_pickle(str(self._data_dir / 'vocab.pkl'))

        self.vocab_size = len(self.word2idx)
        if training:
            filename = str(self._data_dir / 'RAW' / data_path)
            # train_test_split and exist
            self._get_train_and_test(filename, test_split)

    @abstractmethod
    def get_labels(self):
        raise NotImplementedError

    @abstractmethod
    def split_line(self, line):
        raise NotImplementedError

    def nums_label(self):
        return len(self.get_labels())

    def _get_train_and_test(self, filename, test_split):
        if not self.reset and os.path.exists(
                str(self._data_dir / 'train.pkl')):
            return
        print(f"Begin to build dataset")
        if os.path.exists(filename):

            features, length = self.handle_from_file(filename)
            if test_split != 0.0 and test_split < 0.5:
                idx_full = np.arange(length)
                np.random.shuffle(idx_full)
                index = length - int(test_split * length)
                with open(str(self._data_dir / 'train.pkl'), "wb") as f:
                    train_features = dict()
                    for key, value in features.items():
                        train_features[key] = np.array(value)[idx_full[:index]]
                    pickle.dump(train_features, f)

                with open(str(self._data_dir / 'test.pkl'), "wb") as f:
                    test_features = dict()
                    for key, value in features.items():
                        test_features[key] = np.array(value)[idx_full[index:]]
                    pickle.dump(test_features, f)
            else:
                with open(str(self._data_dir / 'train.pkl'), "wb") as f:
                    pickle.dump(features, f)

        else:
            raise FileNotFoundError(f" DataSet file not found in {filename}")

    def get_eval(self, filename):
        if not self.reset and os.path.exists(str(self._data_dir / 'eval.pkl')):
            return
        print(f"Begin to build eval dataset")
        if os.path.exists(filename):
            features, _ = self.handle_from_file(
                filename)
            with open(str(self._data_dir / 'eval.pkl'), "wb") as f:
                pickle.dump(features, f)
        else:
            raise FileNotFoundError(f" DataSet file not found in {filename}")

    def handle_from_file(self, filename):
        features = {
            'query': [],
            'target': [],
            'label': []}
        with open(filename, 'r') as fe:
            for idx, line in enumerate(fe):
                if idx < self.skip_row:
                    continue
                q, t, label = self.split_line(line)
                _query, _target, _label = self.handle(q, t, label)
                features['query'].append(_query)
                features['target'].append(_target)
                features['label'].append(_label)

                if idx % 10000 == 0:
                    print(idx, _query[:30], _target[:30], _label)
                    print(self.convert_ids_to_tokens(_query[:30]), "\n", self.convert_ids_to_tokens(_target[:30]))

        return features, idx - self.skip_row + 1

    def handle_on_batch(self, qs, ts):
        query, target = [], []
        for q, t in zip(qs, ts):
            _query, _target, _ = self.handle(q, t)
            query.append(_query)
            target.append(_target)
        return query, target

    def handle(self, q, t, label=None):
        query = [self.word2idx.get(qw, self.word2idx['UNK']) for qw in self._segment(q, ifremove=False)['tokens']]
        target = [self.word2idx.get(tw, self.word2idx['UNK']) for tw in self._segment(t, ifremove=False)['tokens']]
        query = self.pad2longest(query, self.query_max_len)
        target = self.pad2longest(target, self.target_max_len)
        label_map = {label: i for i, label in enumerate(self.get_labels())}
        if label:
            label = label_map[label]
        return query, target, label

    @staticmethod
    def pad2longest(data_ids, max_len):
        if isinstance(data_ids[0], list):
            s_data = np.array([s[:max_len] + [0] * (max_len - len(s[:max_len])) for s in data_ids])

        elif isinstance(data_ids[0], int):
            s_data = np.array(data_ids[:max_len] + [0] * (max_len - len(data_ids[:max_len])))

        else:
            raise TypeError("list type is required")
        return s_data

    # setting read-only attributes
    @property
    def seg(self):
        return self._segment

    @property
    def embedding(self):
        return self._embedding

    @property
    def vocab(self):
        return self._vocab

    # QA
    def handle_from_file_bert(self, filename):
        features = {
            'input_ids': [],
            'input_mask': [],
            'segment_ids': [],
            'label_id': []}

        label_map = {label: i for i, label in enumerate(self.get_labels())}

        with open(filename, 'r') as fe:
            for idx, line in enumerate(fe):
                if idx < self.skip_row:
                    continue
                q, t, label = self.split_line(line)
                input_ids, input_mask, segment_ids, label_id = self.handle_bert(q, t, label)
                features['input_ids'].append(input_ids)
                features['input_mask'].append(input_mask)
                features['segment_ids'].append(segment_ids)
                features['label_id'].append(label_map[label_id])
                if idx % 10000 == 0:
                    print(idx, input_ids[:30], input_mask[:30], segment_ids[:30], label_id)
                    print(self.tokenizer.convert_ids_to_tokens(input_ids))

        return features, idx - self.skip_row + 1

    def handle_bert_on_batch(self, qs, ts):
        input_ids, input_mask, segment_ids, label_id = [], [], [], []
        for q, t in zip(qs, ts):
            _input_ids, _input_mask, _segment_ids, _label_id = self.handle_bert(q, t, 0)
            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            label_id.append(_label_id)
        return input_ids, input_mask, segment_ids, label_id

    def handle_bert(self, q, t, label):

        tokens_a = self.tokenizer.tokenize(q)  # 分词

        tokens_b = None
        if t:
            tokens_b = self.tokenizer.tokenize(t)  # 分词
            # “-3” 是因为句子中有[CLS], [SEP], [SEP] 三个标识，可参见论文
            # [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            _truncate_seq_pair(tokens_a, tokens_b, self.max_len- 3)
        else:
            # "- 2" 是因为句子中有[CLS], [SEP] 两个标识，可参见论文
            # [CLS] the dog is hairy . [SEP]
            if len(tokens_a) > self.query_max_len - 2:
                tokens_a = tokens_a[:(self.query_max_len - 2)]

        # [CLS] 可以视作是保存句子全局向量信息
        # [SEP] 用于区分句子，使得模型能够更好的把握句子信息

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)  # 句子标识，0表示是第一个句子，1表示是第二个句子，参见论文

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = self.convert_tokens_to_ids(tokens)  # 将词转化为对应词表中的id

        # input_mask: 1 表示真正的真正的 tokens， 0 表示是 padding tokens
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_len - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_len
        assert len(input_mask) == self.max_len
        assert len(segment_ids) == self.max_len

        label_id = label

        return input_ids, input_mask, segment_ids, label_id

    @property
    def data_dir(self):
        return self._data_dir

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.idx2word[i])
        return tokens

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """ 截断句子a和句子b，使得二者之和不超过 max_length """

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()