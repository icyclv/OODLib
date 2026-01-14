from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


@register_baseline("gen")
class GEN(BaseBaseline):     

    @torch.no_grad()
    def eval(self, data_loader):

        gamma = 0.1
        M = 100

        self.model.eval()
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model.get_output(images)
            smax = (F.softmax(output, dim=1)).data.cpu().numpy()
            probs_sorted = np.sort(smax, axis=1)[:,M:]
            scores = np.sum(probs_sorted ** gamma * (1 - probs_sorted) ** gamma, axis=1)

            result.append(-scores)

        return np.concatenate(result)

