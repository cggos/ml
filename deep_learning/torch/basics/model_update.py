import torchvision.models as models

model = models.vgg16(pretrained=False)
model_pretrained = models.resnet152(pretrained=True)

dict_pretrained = model_pretrained.state_dict()
# 加载torchvision中的预训练模型和参数后通过state_dict()方法提取参数,也可以直接从官方model_zoo下载
# dict_pretrained = model_zoo.load_url(model_urls['resnet152'])
dict_model = model.state_dict()

print(f"keys:\n {dict_model.keys()}\n")

# 将dict_pretrained里不属于dict_model的键剔除掉
dict_pretrained = {k: v for k, v in dict_pretrained.items() if k in dict_model}

print(f"keys:\n {dict_pretrained.keys()}\n")

dict_model.update(dict_pretrained)
model.load_state_dict(dict_model)

print(f"keys:\n {model.state_dict().keys()}\n")
