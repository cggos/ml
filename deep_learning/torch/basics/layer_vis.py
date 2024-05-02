import cv2
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as T


def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()

    def remove(self):
        self.hook.remove()


img_path = "./data/cat.194.jpg"
img_raw = cv2.imread(img_path)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

simple_transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
imgT = simple_transform(img_raw)


vgg = models.vgg16(pretrained=True)

print(vgg)

print(vgg.state_dict().keys())

conv_out = LayerActivations(vgg.features, 5)

print(imgT.shape)
imshow(imgT)

o = vgg(Variable(imgT.unsqueeze(0)))

conv_out.remove()

act = conv_out.features

fig = plt.figure(figsize=(20, 50))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
for i in range(30):
    ax = fig.add_subplot(12, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(act[0][i])


cnn_weights = vgg.state_dict()['features.0.weight'].cpu()
fig = plt.figure(figsize=(30, 30))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
for i in range(30):
    ax = fig.add_subplot(12, 6, i + 1, xticks=[], yticks=[])
    imshow(cnn_weights[i])

plt.show()
