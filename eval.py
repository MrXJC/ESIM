import argparse
import data_loader.data_loaders as module_data
import data_loader.processor as module_processor
from parse_config import ConfigParser
import model.model as module_arch
from agent import Agent, metric
import pandas as pd
import numpy as np
import torch.nn as nn


def eval(config, filename):
    logger = config.get_logger('test')

    # setup data_loader instances
    processor = config.initialize(
        'processor', module_processor, logger, config,  training=False)
    processor.get_eval(filename)
    test_data_loader = config.initialize('data_loader', module_data, processor.data_dir, mode="eval",
                                         debug=config.debug_mode)

    # build model architecture, then print to console
    model = config.initialize(
        'arch',
        module_arch,
        vocab_size=processor.vocab_size, num_labels=processor.nums_label())

    #logger.info(model)
    agent = Agent(model, config=config, test_data_loader=test_data_loader)
    return agent.test(detail=True), processor


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-e', '--eval', default="eval.csv", type=str,
                      help='the path of eval file (default: eval.csv)')
    args.add_argument('-reset', '--reset', default=False, type=bool,
                      help='debug')

    config = ConfigParser(args)
    args = args.parse_args()
    (qs, ts, labels, outputs, _), processor = eval(config, args.eval)

    metric.classifiction_metric(outputs, labels, processor.get_labels())

    result_path = processor.data_dir / "RAW/result.csv"
    df = {"q": [], "t": [], 'label': [], "output": [], "wrong": []}
    for index in range(len(qs)):
        df["wrong"].append("true" if labels[index] == int(np.argmax(outputs[index])) else "false")
        df["q"].append(
            "".join(map(lambda x: processor.idx2word[x] if x != 0 else '', qs[index])))
        df["t"].append(
            "".join(map(lambda x: processor.idx2word[x] if x != 0 else '', ts[index])))
        df["label"].append(labels[index][0])
        df["output"].append(outputs[index])
    pd.DataFrame(df, columns=["q", "t", "label", "output", "wrong"]).to_csv(result_path)
