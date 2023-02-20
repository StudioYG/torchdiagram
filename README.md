# torchdiagram
Pytorch has a compact 'torchsummary' tool for model visualization but merely providing the structural summary info for the model. Motivated by 'torchsummary' tool and the statistical demand of model, a statistical tool called 'torchdiagram' is born to tally the module distribution for the model and visualize it intuitively in the pie style!


## Usage
```git clone https://github.com/StudioYG/torchdiagram```

```
import torch
from torchdiagram import visualize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = your_model.to(device)
visualize(model, input_size=(channels, H, W))
```
Note that the legend of pie chart depicts Sector index. Module name: Proportion (Quantity)

## Examples
### MNIST Network
```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiagram import visualize

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
visualize(model, (1, 28, 28))
```
```
layer_class:  ['Conv2d', 'Dropout2d', 'Linear']
layer_count:  [2, 1, 2]
```
![image](https://github.com/GYQ-AI/torchdiagram/blob/main/examples/MNIST%20Network.png)

### Resnet18
```
import torch
from torchvision import models
from torchdiagram import visualize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18().to(device)
visualize(model, (3, 224, 224))
```
```
layer_class:  ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'BasicBlock', 'AdaptiveAvgPool2d', 'Linear']
layer_count:  [20, 20, 17, 1, 8, 1, 1]
```
![image](https://github.com/GYQ-AI/torchdiagram/blob/main/examples/Resnet18.png)

### Mobilenetv3-small
```
import torch
from torchvision import models
from torchdiagram import visualize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v3_small().to(device)
visualize(model, (3, 224, 224))
```
```
layer_class:  ['Conv2d', 'BatchNorm2d', 'Hardswish', 'ReLU', 'SqueezeExcitation', 'Identity', 'InvertedResidual', 'AdaptiveAvgPool2d', 'Linear', 'Dropout']
layer_count:  [52, 34, 19, 14, 9, 11, 11, 1, 2, 1]
```
![image](https://github.com/GYQ-AI/torchdiagram/blob/main/examples/Mobilenetv3-small.png)

## References
1. https://github.com/sksq96/pytorch-summary 
2. https://github.com/graykode/modelsummary 
3. https://github.com/TylerYep/torchinfo
## License
```torchdiagram``` is MIT-licensed.
