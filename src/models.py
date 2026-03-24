import torch
import torch.nn as nn
import torchvision.models as models

CONV_LAYERS = {
    'conv1': 1,
    'conv2': 4,
    'conv3': 7,
    'conv4': 9,
    'conv5': 11,
}

class AlexNetProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained AlexNet
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.alexnet = self.alexnet.to(self.device)
        self.alexnet.eval()

        # Storage for captured activations
        self._activations = {}

        # Register a hook on each conv layer's post-ReLU output
        self._hooks = []
        for layer_name, idx in CONV_LAYERS.items():
            hook = self.alexnet.features[idx].register_forward_hook(
                self._make_hook(layer_name)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_name):
        def hook(module, input, output):
            self._activations[layer_name] = output.detach()
        return hook

    def forward(self, x):
        x = x.to(self.device)
        return self.alexnet(x)

    def get_activations(self):
        return dict(self._activations)


def load_model():
    return AlexNetProbe()