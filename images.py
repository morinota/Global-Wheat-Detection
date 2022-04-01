from turtle import st
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np
import torch
import cv2
import os

from torch import Tensor
INPUT_DIR = 'input'


def visualize_image_and_bboxes(images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]], i: int) -> None:
    '''
    DataLoaderから取得してきた1バッチ分のデータセットに対し、指定されたindexの画像データとbboxを可視化する関数
    '''
    # 1バッチ分のデータの中から、指定されたindexの情報を抽出
    boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
    image = images[i].permute(1, 2, 0).cpu().numpy()

    # Figureオブジェクト、Axesオブジェクトの生成
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # 画像にbboxを1つずつ描画
    for box in boxes:
        cv2.rectangle(img=image,
                      pt1=(box[0], box[1]),
                      pt2=(box[2], box[3]),
                      color=(220, 0, 0),
                      thickness=3
                      )

    ax.set_axis_off()
    ax.imshow(image)

    plt.savefig(os.path.join(INPUT_DIR, f'image_and_bboxes_{i}.jpg'),
                tight_layout=True, 
                dpi=64,
                facecolor="lightgray",)
    del fig, ax
