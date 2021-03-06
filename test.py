import argparse
import data_loader.data_loaders as module_data
import data_loader.processor as module_processor
from parse_config import ConfigParser
import model.model as module_arch
from agent import Agent


def test(config):
    logger = config.get_logger('test')
    # setup data_loader instances
    processor = config.initialize('processor', module_processor, logger, config, training=False)
    test_data_loader = config.initialize('data_loader', module_data, processor.data_dir, mode="test", debug=config.debug_mode)

    # build model architecture, then print to console
    model = config.initialize(
        'arch',
        module_arch,
        vocab_size=processor.vocab_size,num_labels = processor.nums_label())

    logger.info(model)
    agent = Agent(model, config=config, test_data_loader=test_data_loader)
    return agent.test()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    print(test(config))
