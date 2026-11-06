import torch
import intel_extension_for_pytorch as ipex  # Wakes up the Arc GPU

print(torch.xpu.is_available())