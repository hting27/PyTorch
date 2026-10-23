"""現場展示：載入最佳模型，預測一張水果照片。"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from cnn import CNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parent/'result/best_cnn.pth')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = CNN(len(checkpoint['classes']))
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    with Image.open(args.image) as im:
        x = np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((checkpoint['image_size'],)*2)).copy()
    x = torch.from_numpy(x).permute(2, 0, 1).float()/255
    norm = checkpoint['normalization']
    with torch.no_grad():
        probs = model(((x-norm['mean'])/norm['std']).unsqueeze(0)).softmax(1)[0]
    for i in probs.argsort(descending=True).tolist():
        print(f"{checkpoint['classes'][i]}: {probs[i].item():.2%}")


if __name__ == '__main__':
    main()
