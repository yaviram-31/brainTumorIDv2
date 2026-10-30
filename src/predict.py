import torch
from model_def import ConvNet
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import numpy as np

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'model_0.pt'

model = ConvNet(1,4)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

data_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

img = Path(input("Image file to classify: ").strip().strip('"'))
if not img.is_absolute() and not img.exists():
    img = Path(__file__).resolve().parent / img

img_read = Image.open(img).convert('RGB')
img_tensor = data_transforms(img_read).unsqueeze(0)

with torch.no_grad():
    pred = model(img_tensor)
probs = torch.softmax(pred, dim=1)[0]
predicted_idx = probs.argmax().item()
predicted_class = classes[predicted_idx]

for name, p in zip(classes, probs):
    print(f"{name:>10}: {p.item():.1%}")

# Grad-CAM on the last conv layer, for the predicted class
target_layers = [model.layer_stack[2]]
targets = [ClassifierOutputTarget(predicted_idx)]
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]

# Overlay needs an RGB float image in [0, 1], same size as the model input
rgb_img = np.float32(img_read.resize((224,224))) / 255
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(rgb_img)
axes[0].set_title(f"Predicted: {predicted_class}")
axes[1].imshow(cam_image)
axes[1].set_title(f'Grad-CAM: {predicted_class} ({probs[predicted_idx].item():.1%})')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
