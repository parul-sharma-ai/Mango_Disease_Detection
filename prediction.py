import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from models.base_model import BaseFeatureExtractor
from datasets.mango import build_transform

class_names = [
    "Anthracnose",
    "Bacterial_black_spot",
    "Healthy",
    "Nutritional Deficincy",
    "Powdery Mildew"
    ]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_path = "/Users/parulsharma/Documents/Mango Disease Classifier/weights/conv_trans/best_model.pt"

model = BaseFeatureExtractor(config='conv_trans', n_class=5).to(device)

ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt['model_dict'])

model.eval()

transform = build_transform(is_train=False, input_size=224)


def predict(image_path):
    img = Image.open(image_path).convert("RGB")

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    class_name = class_names[predicted.item()]
    confidence = confidence.item()

    return class_name, confidence


'''image_path = "/Users/parulsharma/Documents/Mango Disease Classifier/Test_images/unseen2.jpeg"

class_name, confidence = predict(image_path)

print("Prediction:", class_name)
print("Confidence:", round(confidence*100,2), "%")

img = Image.open(image_path)
plt.imshow(img)
plt.title(f"{class_name} ({confidence:.2f})")
plt.axis("off")
plt.show()'''

import os
import matplotlib.pyplot as plt
from PIL import Image

folder = "/Users/parulsharma/Documents/Mango Disease Classifier/Test_images"

for img_name in os.listdir(folder):
    path = os.path.join(folder, img_name)
    
    # Predict
    class_name, confidence = predict(path)
    
    # Print prediction
    print(img_name, class_name, round(confidence*100, 2), "%")
    
    # Show image with prediction
    img = Image.open(path)
    plt.imshow(img)
    plt.title(f"{class_name} ({confidence*100:.2f}%)")
    plt.axis("off")
    plt.show()