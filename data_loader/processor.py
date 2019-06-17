from base import BaseProcessor


class ATECProcessor(BaseProcessor):
    def __init__(self, logger, config, data_name, data_path, embed_path=None, user_dict=None, vocab_path=None, stop_word=None, max_len=50, query_max_len=20,
                 target_max_len=20, test_split=0.0, training=True):
        self.skip_row = 0
        super().__init__(logger, config, data_name, data_path, embed_path, user_dict, vocab_path, stop_word, max_len, query_max_len,
                         target_max_len, test_split, training)

    def get_labels(self):
        """See base class."""
        return [u'0', u'1']

    def split_line(self, line):
        line = line.strip().split('\t')
        q, t, label = line[1], line[2], line[-1]
        return q, t, label
