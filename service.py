from flask import Flask  # 导入Flask库
from flask import request  # 导入处理request的库
import argparse
from flask_json import FlaskJSON, JsonError, as_json  # 导入处理Json输入输出的库
import data_loader.processor as module_processor
import model.model as module_arch
from parse_config import ConfigParser
from agent import Agent

app = Flask(__name__)  # 初始化app
json = FlaskJSON(app)  # 初始化json处理器


def predict(q, t):
    batch = processor.handle_on_batch(q, t)
    return agent.predict(batch)


@app.route('/api/esim', methods=['POST'])  # 装饰器，用来设定URL路径和接受的方法
@as_json  # 装饰器，用来将函数的return封装成json格式返回
def test():  # 处理访问的函数
    data = request.get_json(
        force=False,
        silent=False,
        cache=True)  # 从request里面读取json数据
    try:  # 处理异常
        _, label = predict(
            data['query'],
            data['target'])  # 从json数据里面读取text字段，生成返回
        response = {'label': list(map(int, label))}
    except (KeyError, TypeError, ValueError):  # 捕获数据类型异常
        raise JsonError(description='Invalid value.')  # 将异常反馈会调用
    return response  # 正常返回，这个response的内容会被转成json格式


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    logger = config.get_logger('test')
    # setup data_loader instances
    processor = config.initialize(
        'processor', module_processor, logger, config)
    # build model architecture, then print to console
    model = config.initialize(
        'arch',
        module_arch,
        vocab_size=processor.vocab_size, num_labels=processor.nums_label())
    # logger.info(model)
    agent = Agent(model, config=config)

    app.run(host='0.0.0.0', port=5000, debug=True)
