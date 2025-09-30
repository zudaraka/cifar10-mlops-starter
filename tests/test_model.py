from src.model import SmallCNN
import torch

def test_forward():
    m = SmallCNN()
    x = torch.randn(2,3,32,32)
    y = m(x)
    assert y.shape == (2,10)
