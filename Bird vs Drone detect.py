import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 1. LOAD MODEL
# -------------------------------
def load_model(weights_path, num_classes=2):
    model = mobilenet_v2(weights='IMAGENET1K_V1')

    # Replace classifier output layer
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    # Load trained weights
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.to(device)
    model.eval()
    return model


# -------------------------------
# 2. IMAGE TRANSFORM
# -------------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -------------------------------
# 3. PREDICT FUNCTION
# -------------------------------

def predict_image(model, image_path, class_names):

    image_path = "valid/bird/0afb3e0beab24519_jpg.rf.a3b1ea999c57a7feb86083c2ce28fe1d.jpg"

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = outputs.argmax(1).item()

    predicted_label = class_names[pred_idx]
    confidence = probs[0][pred_idx].item()

    return predicted_label, confidence



# -------------------------------
# 4. MAIN FUNCTION
# -------------------------------

if __name__ == "__main__":
    
    weights_path = "bird_drone_mobilenetv2.pth"
    class_names  = ["bird", "drone"]  # or train_data.classes

    model = load_model(weights_path)

    image_path = "valid/bird/0afb3e0beab24519_jpg.rf.a3b1ea999c57a7feb86083c2ce28fe1d.jpg"
    label, conf = predict_image(model, image_path, class_names)

    print(f"\nPrediction: {label}")
    print(f"Confidence: {conf:.4f}")
