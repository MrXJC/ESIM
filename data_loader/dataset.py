# -*- coding:utf-8 -*-
import os
import torch
import pickle

DEBUG_INDEX = 128


class BertDataset:
    def __init__(self, filename, debug):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                features = pickle.load(f)

            self.input_ids, self.input_mask, self.segment_ids, self.label_id = features['input_ids'], \
                features['input_mask'], features['segment_ids'], features['label_id']
            if debug:
                self.input_ids, self.input_mask, self.segment_ids, self.label_id = \
                    self.input_ids[:DEBUG_INDEX], self.input_mask[:DEBUG_INDEX], self.segment_ids[:DEBUG_INDEX], self.label_id[:DEBUG_INDEX]

        else:
            raise FileNotFoundError(f"train data file not found in {filename}")

    def batch(self, index):
        return self.input_ids[index], self.input_mask[index], self.segment_ids[index], self.label_id[index]

    def __getitem__(self, index):
        input_ids, input_mask, segment_ids, label_id = self.input_ids[
            index], self.input_mask[index], self.segment_ids[index], self.label_id[index]
        return torch.LongTensor(input_ids), torch.LongTensor(input_mask), \
               torch.LongTensor(segment_ids), torch.LongTensor([label_id])

    def __len__(self):
        return len(self.label_id)


class Dataset:
    def __init__(self, filename, debug):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                features = pickle.load(f)

            self.query, self.target, self.label = features['query'], features['target'], features['label']
            if debug:
                self.query, self.target, self.label = self.query[:DEBUG_INDEX], self.target[:DEBUG_INDEX], self.label[:DEBUG_INDEX]
        else:
            raise FileNotFoundError(f"train data file not found in {filename}")

    def batch(self, index):
        return self.query[index], self.target[index], self.label[index]

    def __getitem__(self, index):
        query, target, label = self.query[index], self.target[index], self.label[index]
        return torch.LongTensor(query), torch.LongTensor(target), torch.LongTensor([label])

    def __len__(self):
        return len(self.label)
