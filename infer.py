#!/usr/bin/environ python3
import imageio.v2 as imageio
import argparse
import numpy as np
from pathlib import Path
from torchvision.utils import draw_bounding_boxes
import torchvision
from main import FasterRCNNLightning
from PIL import Image

from torchvision.transforms import v2 as transforms
import torch

# Parameters
ckpt_path = '3.ckpt'
image_target_dims = [512, 512]
mean, std = [0.485], [0.229]
threshold = 0.7

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def custom_normalize(x, mean, std):
    # Skip if boxes/labels are passed
    if isinstance(x, torchvision.tv_tensors._image.Image):
        x_norm = (x - x.min()) / (x.max() - x.min())
        # Disabled until mean and std are calculated for the dataset
        # (Temporarily disabled) NOT_IMPLEMENTED
        # return (x_norm - mean[0]) / std[0]
        return x_norm
    else:
        return x


def transform(image, mean=mean, std=std):
    transform_compose = transforms.Compose([
        transforms.Resize(image_target_dims, antialias=True),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32),
        transforms.Lambda(lambda x: custom_normalize(x, mean, std))
    ])
    return transform_compose(image)


def visualize(image, preds, threshold):
    image = torch.as_tensor(image, dtype=torch.uint8)
    if image.ndim < 3:
        image = image.unsqueeze(0)
    boxes, labels = preds['boxes'], preds['labels']
    mask = preds['scores'] > threshold
    boxes, labels = boxes[mask], labels[mask]
    labels = [str(x.cpu().numpy()) for x in labels]  # To string, for plot labels
    # Overlay the boxes on image
    preds_to_show = draw_bounding_boxes(image, boxes=boxes, labels=labels)
    return transforms.ToPILImage()(preds_to_show)


def infer(image, ckpt_path=ckpt_path, threshold=threshold, image_target_dims=image_target_dims):
    pil_image = Image.fromarray(image).convert('L')
    pil_image = pil_image.resize(image_target_dims)
    image = np.array(pil_image)
    # Prepare model input by transforming the image
    image = torch.from_numpy(image).unsqueeze(0)
    model_input = transform(image).unsqueeze(0).to(device)

    # Load model using source from checkpoint
    model = FasterRCNNLightning.load_from_checkpoint(ckpt_path).to(device)
    model.eval()  # Turn off training mode
    # Infer
    preds = model(model_input)[0]
    image_with_overlay = visualize(image, preds, threshold)
    image_array = np.array(image_with_overlay)

    return image_array


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser("Inference using model checkpoint")
    parser.add_argument('-m', '--model', required=True, help="Path to checkpoint of model")
    parser.add_argument('-i', '--image', required=True, help="Path to input image")
    parser.add_argument('-o', '--output', help="Output image path")
    args = parser.parse_args()
    ckpt_path = args.model
    image_path = args.image
    if args.output:
        output_path = args.output
    else:
        image_path = Path(image_path)
        output_path = image_path.with_stem(image_path.stem + '_out')
        # cv2 doesn't like Path objects as filename apparently
        image_path, output_path = str(image_path), str(output_path)
    # Infer
    image = imageio.imread(image_path)
    output_image = infer(image, ckpt_path)
    imageio.imwrite(output_path, output_image)
