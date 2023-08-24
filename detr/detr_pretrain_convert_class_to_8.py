import torch

pretrained_weight = torch.load('detr-r50-e632da11.pth')

num_class = 8
pretrained_weight = ["model"]["class_embed.weight"].resize_(num_class+1 , 256)
pretrained_weight = ["model"]["class_embed.bias"].resize_(num_class+1)

torch.save(pretrained_weight, "detr-r50_%d.pth"%num_class)