from baselines.base import BaseBaseline
from baselines.registry import register_baseline


@register_baseline("energy")
class Energy(BaseBaseline):
    def score(self, logits=None, features=None):
        # TODO: 实现 energy 得分
        raise NotImplementedError
