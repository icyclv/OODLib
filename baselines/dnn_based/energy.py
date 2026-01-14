from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
from tqdm import tqdm


@register_baseline("energy")
class Energy(BaseBaseline):

    @torch.no_grad()
    def eval(self, data_loader):

        T = 1

        self.model.eval()
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model.get_output(images)

            energy = T * torch.logsumexp(output / T, dim=1).data.cpu().numpy()
            result.append(energy)

        return np.concatenate(result)

