import numpy as np
import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.models.detection as detection
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ModDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_dir, mask_dir):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        masks = np.zeros((num_objs, mask.shape[0], mask.shape[1]))
        for i, obj_id in enumerate(obj_ids):
            masks[i][mask == obj_id] = 1

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return T.ToTensor()(image), target

    def __len__(self):
        return len(self.image_paths)


class ObjectDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
        self.model.roi_heads.box_predictor = detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes=2).roi_heads.box_predictor
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        self.transforms = T.Compose([T.ToTensor()])

    def detect_pedestrians(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.model(image_tensor)
        return predictions

def main():
    imagePngPath = "dataset/PennFudanPed/PNGImages/"
    masksPath = "dataset/PennFudanPed/PedMasks/"

    images = sorted(os.listdir(imagePngPath))
    masks = sorted(os.listdir(masksPath))

    # Example usage:
    dataset = ModDataset(images, masks)
    sample_idx = 9
    sample_image, sample_target = dataset[sample_idx]

    plt.figure(figsize=(8, 8))
    plt.imshow(sample_image.numpy().transpose((1, 2, 0)))
    boxes = sample_target["boxes"].numpy()
    labels = sample_target["labels"].numpy()

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(xmin, ymin, f"Label: {label}", color='r', fontsize=12, backgroundcolor='w')

    plt.axis('off')
    plt.show()

    model_path = "models/mobilenet_path_to_your_model.pth"
    detector = ObjectDetector(model_path)
    new_image_path = 'test/crosswalk-featured.jpg'
    detections = detector.detect_pedestrians(new_image_path)
    print(detections)

if __name__ == "__main__":
    main()
