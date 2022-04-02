from typing import Dict, List
from torch.utils.data import DataLoader
import torch
from torch import Tensor
import numpy as np
from torchvision.models.detection.faster_rcnn import FasterRCNN


def show(test_dataloader: DataLoader, model: FasterRCNN, image_i: int = 0):
    '''
    推論結果をチェックする関数
    '''
    import matplotlib.pyplot as plt
    from PIL import ImageDraw, ImageFont
    from PIL import Image

    # GPUのキャッシュクリア
    torch.cuda.empty_cache()
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # データセットを取得
    images, targets, image_ids = next(iter(test_dataloader))
    images = list(img.to(device) for img in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    images : List[Tensor]
    targets: List[Dict[str, Tensor]]


    # 対象の画像を抽出
    boxes = targets[image_i]['boxes'].cpu().numpy().astype(np.int32)
    sample_image = images[image_i].permute(1,2,0).cpu().numpy()

    model.to(device)
    # modelを推論モードへ
    model.eval()
    # 描画はcpuで行う
    cpu_device = torch.device("cpu")


    # modelに画像のTensorを渡せば推論を実行する(返値もTensor)
    outputs = model(sample_image) # ->()

def main():
    pass

if __name__ == "__main__":
    main()
