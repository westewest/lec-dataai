import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device:", torch.cuda.get_device_name())
print("CUDA Version:", torch.version.cuda)
print("CUDA Archtecture:", torch.cuda.get_arch_list())
