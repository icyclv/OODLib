class BaseBaseline:

    def __init__(self, config):
        self.config = config

    def score(self, logits=None, features=None):
        raise NotImplementedError
