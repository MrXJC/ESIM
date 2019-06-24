import argparse
import data_loader.processor as module_processor
import model.model as module_arch
from parse_config import ConfigParser
from agent import Agent

def predict(config, s1, s2):
    logger = config.get_logger('test')
    # setup data_loader instances
    processor = config.initialize('processor', module_processor, logger, config,  training=False)
    # build model architecture, then print to console
    # build model architecture, then print to console
    # build model architecture, then print to console
    model = config.initialize(
        'arch',
        module_arch,
        vocab_size=processor.vocab_size,num_labels = processor.nums_label())
    #logger.info(model)
    agent = Agent(model, config=config)

    batch = processor.handle_on_batch(s1, s2)
    print(batch)
    print(agent.predict(batch))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    args.add_argument('-s1', '--sentence1', default=" ", type=str,
                      help='s1')
    args.add_argument('-s2', '--sentence2', default=" ", type=str,
                      help='s2')

    #config = MockConfigParser(resume='saved/models/ATEC_ESIM/0530_101350/model_best.pth')
    config = ConfigParser(args)
    args = args.parse_args()
    # for arg in sys.argv:
    #     print(arg)
    print(args.sentence1, args.sentence2)
    predict(config, [args.sentence1], [args.sentence2])

