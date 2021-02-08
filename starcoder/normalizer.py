import torch

class Normalizer(torch.nn.Module):
    pass

class LayerNormalizer(torch.nn.LayerNorm):
    def __init__(self, input_size):
        super(LayerNormalizer, self).__init__([input_size])
        
