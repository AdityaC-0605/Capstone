import torch

def cast_tensor(x, precision):
    if precision == "fp32":
        return x.float()
    if precision == "fp16":
        return x.half()
    if precision == "int8":
        return x.to(torch.int8)
    raise ValueError("Unsupported precision")
