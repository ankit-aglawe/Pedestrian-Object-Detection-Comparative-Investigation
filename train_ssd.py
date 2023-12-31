import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.models.detection import ssd300_vgg16
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import functional as F


def get_mask_pil(imagePath, maskPath):
  img = Image.open(imagePath).convert("RGB")
  mask = Image.open(maskPath)
  mask = np.array(mask)

  return Image.fromarray(mask), img

def get_box(num_objs, masks):
  boxes = []
  for i in range(num_objs):
      pos = np.where(masks[i])
      xmin = np.min(pos[1])
      xmax = np.max(pos[1])
      ymin = np.min(pos[0])
      ymax = np.max(pos[0])
      boxes.append([xmin, ymin, xmax, ymax])
  return boxes


def get_dict(boxes, labels):
  target = {}
  target["boxes"] = boxes
  target["labels"] = labels
  return target

class ModDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.imagePaths = image_paths
        self.maskPaths = mask_paths
        self.transform = transform

    def __getitem__(self, idx):
        imagePath = imagePngPath + self.imagePaths[idx]
        maskPath = masksPath + self.maskPaths[idx]
        mask_pil, img = get_mask_pil(imagePath, maskPath)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask_pil)

        mask = np.array(mask_pil)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        masks = np.zeros((num_objs, mask.shape[0], mask.shape[1]))

        for i, obj_id in enumerate(obj_ids):
            masks[i][mask == obj_id] = 1

        boxes = get_box(num_objs, masks)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        target = get_dict(boxes, labels)

        return T.ToTensor()(img), target

    def __len__(self):
        return len(self.imagePaths)


class ObjectDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ssd300_vgg16(pretrained=False)
        num_classes = 2
        self.model.class_predictor = torch.nn.Conv2d(256, num_classes * 4, kernel_size=(1, 1), stride=(1, 1))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        self.transforms = T.ToTensor()

    def detect_pedestrians(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.model(image_tensor)
        return predictions

def main():
    image_dir = "dataset/PennFudanPed/PNGImages/"
    mask_dir = "dataset/PennFudanPed/PedMasks/"

    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))

    dataset = ModDataset(images, masks, image_dir, mask_dir)
    sample_idx = 8
    sample_image, sample_target = dataset[sample_idx]

    plt.figure(figsize=(8, 8))
    plt.imshow(sample_image.numpy().transpose((1, 2, 0)))
    boxes = sample_target["boxes"].numpy()
    labels = sample_target["labels"].numpy()

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(xmin, ymin, f"Label: {label}", color='r', fontsize=12, backgroundcolor='w')

    plt.axis('off')
    plt.show()

    model_path = "models/ssd_path_to_your_model.pth"
    detector = ObjectDetector(model_path)
    new_image_path = 'test/crosswalk-featured.jpg'
    detections = detector.detect_pedestrians(new_image_path)
    print(detections)

if __name__ == "__main__":
    main()
