from typing import Dict, List
from torch.utils.data import DataLoader
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from images.images import INPUT_DIR
from torchvision.models.detection.faster_rcnn import FasterRCNN
import matplotlib.pyplot as plt
import cv2
import os


def _draw_bboxes_on_image(image: np.ndarray, bboxes_predicted: ndarray, bboxes_observed: ndarray, png_name: str):
    """画像pixel(ndarray)と、bboxの座標群を渡して、それらをpngとして描画出力する関数

    Parameters
    ----------
    image : np.ndarray
        _description_
    bboxes_predicted : ndarray
        _description_
    bboxes_observed : ndarray
        _description_
    png_name : str
        _description_
    """
    # bboxの色の指定
    colormap_dict = {'predicted':(255, 0, 40), 'actual':(91, 255, 0)}

    # Figureオブジェクト、Axesオブジェクトの生成
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # 画像にbboxを1つずつ描画
    for bbox in bboxes_predicted:
        cv2.rectangle(img=image,
                      pt1=(bbox[0], bbox[1]),
                      pt2=(bbox[2], bbox[3]),
                      color=colormap_dict['predicted'], # 赤に近い色
                      thickness=3, 
                      )

    for bbox in bboxes_observed:
        cv2.rectangle(img=image,
                      pt1=(bbox[0], bbox[1]),
                      pt2=(bbox[2], bbox[3]),
                      color=colormap_dict['actual'], # 青に近い色
                      thickness=3,
                      )

    ax.set_axis_off()
    ax.imshow(image)
    INPUT_DIR = 'input'
    plt.savefig(os.path.join(INPUT_DIR, f'{png_name}.png'),
                tight_layout=True,
                dpi=64,
                facecolor="lightgray",)
    print(f'save the figure')


def show_images_bbox_predicted(test_dataloader: DataLoader, model: FasterRCNN, image_i: int = 0):
    '''
    推論結果を取得して、画像+bbox(推論結果と実測値の両方)としてpng出力する関数
    '''

    # GPUのキャッシュクリア
    torch.cuda.empty_cache()
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # データセットを取得
    images, targets, image_ids = next(iter(test_dataloader))
    # Tensorをdeviceに渡す。(ListやDictに格納されているので)
    images = list(img.to(device) for img in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    images: List[Tensor]
    targets: List[Dict[str, Tensor]]

    # 対象の画像を抽出
    boxes = targets[image_i]['boxes'].cpu().numpy().astype(np.int32)
    sample_image = images[image_i].permute(1, 2, 0).cpu().numpy()

    model.to(device)
    # modelを推論モードへ
    model.eval()
    # 描画はcpuで行う?
    cpu_device = torch.device("cpu")

    # modelに画像のTensorを渡せば推論を実行する(返値もTensor)
    outputs: List[Dict[str, Tensor]] = model(images)  # ->(dataloaderの返値と同じ形?)

    # detection閾値の設定
    detection_threshold = 0.7

    # Tensorをデバイスに渡す
    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    # 指定の画像のbboxesの推定値を取得+ndarrayに格納
    outputs_bboxes = outputs[image_i]['boxes'].cpu(
    ).detach().numpy()
    # 各bboxのスコア(物体である確率?)を取得 + ndarrayに格納
    outputs_scores = outputs[image_i]['scores'].data.cpu().numpy()

    # スコアが閾値より高いbboxのみを残す。
    outputs_bboxes = outputs_bboxes[outputs_scores >= detection_threshold].astype(np.int32)
    # スコアも、選択したbboxと同じ長さにしておく
    outputs_scores = outputs_scores[outputs_scores >= detection_threshold]
    
    # 描画＋png出力
    file_name = f'image_bboxes_predict_{image_i}'
    _draw_bboxes_on_image(image=sample_image,
                          bboxes_predicted=outputs_bboxes,
                          bboxes_observed=boxes,
                          png_name=file_name)
    
