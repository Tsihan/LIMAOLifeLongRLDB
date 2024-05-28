import torch

ckpt_path = '/home/qihan/balsaLifeLongRLDB/tensorboard_logs_/0_drxbl3ln/checkpoints/epoch=44.ckpt'
checkpoint = torch.load(ckpt_path)
print(ckpt_path)
for name, param in checkpoint.items():
    print("Tensor name:", name)
    print("Value:", param)

