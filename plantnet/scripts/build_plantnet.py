"""Build GPU-compatible PlantNet-300K ResNet18 (1081-species plant ID) via litert-torch.
Only patch: ZeroPadMaxPool (ResNet stem MaxPool(-inf PADV2) -> 0-pad + unpadded maxpool)."""
import torch, torch.nn as nn, torch.nn.functional as F, os
from torchvision.models import resnet18
from huggingface_hub import hf_hub_download

class ZeroPadMaxPool(nn.Module):
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), value=0.0)   # exact: maxpool input is post-ReLU >= 0
        return F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

# Fine-tune overrides (defaults reproduce the official ship exactly):
#   PLANTNET_CKPT=w.pth        your own resnet18 classifier state dict
#   PLANTNET_NUM_CLASSES=N     its class count (default 1081)
NCLS = int(os.environ.get("PLANTNET_NUM_CLASSES", "1081"))
net = resnet18(num_classes=NCLS).eval()
ckpt = os.environ.get("PLANTNET_CKPT") or hf_hub_download("cpoisson/plantnet300k-resnet18", "plantnet_resnet18.pth")
net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
net.maxpool = ZeroPadMaxPool()

dummy = torch.randn(1, 3, 224, 224)
import litert_torch
litert_torch.convert(net, (dummy,)).export("plantnet.tflite")
print("saved %.1f MB" % (os.path.getsize("plantnet.tflite") / 1e6))
