This project implements a Bird vs Drone image classification model using PyTorch and transfer learning with a pretrained MobileNetV2 architecture.
The model classifies aerial images into two categories: Bird or Drone.

Task: Binary image classification (Bird vs Drone)
Model: MobileNetV2 (pretrained on ImageNet)
Framework: PyTorch
Technique: Transfer Learning

Output: Trained .pth model file

project_root/
│
├── train/
│   ├── bird/
│   └── drone/
│
├── valid/
│   ├── bird/
│   └── drone/
│
├── test/
│   ├── bird/
│   └── drone/

Image Preprocessing & Augmentation
Training Transformations
Random resized crop (224×224)
Random horizontal flip
Random rotation (±20°)
Normalization (ImageNet mean & std)
Validation & Testing Transformations
Resize → Center crop (224×224)
Normalization (ImageNet mean & std)


Test Accuracy: 95.93

How to run
pip install torch torchvision
python train.py



