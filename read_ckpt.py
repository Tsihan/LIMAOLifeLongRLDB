import torch

ckpt_path = '/home/qihan/balsa_project/balsaLifeLongRLDB/tensorboard_logs_/0_jg6x6uzs/checkpoints/epoch=9.ckpt'
checkpoint = torch.load(ckpt_path)
print(ckpt_path)
for name, param in checkpoint.items():
    print("Tensor name:", name)
    print("Value:", param)

