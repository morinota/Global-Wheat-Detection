from email.mime import image
from typing import Dict, List
from torch.utils.data import DataLoader
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from images.images import INPUT_DIR
from matplotlib.dates import MonthLocator
from sympy import Li, re
from torchvision.models.detection.faster_rcnn import FasterRCNN
import matplotlib.pyplot as plt
import cv2
import os



def format_prediction_string(boxes, scores):
    '''
    ある1つの画像に対する、bboxの座標とスコアを受け取って、
    "score x y w h"(提出の様式)に変換する関数
    '''
    # 結果格納用のリストを用意
    pred_strings = []

    # bbox1つ1つに対して処理を実行
    for j in zip(scores, boxes):
        # "score x y w h"のstringを作って、リストに追加.
        pred_strings.append(
            f"{j[0]:.4f} {j[1][0]} {j[1][1]} {j[1][2]} {j[1][3]}"
            )

    # str.join()で、リストの要素をstrで連結し、1つのstrへ変換
    str_pred_strings = " ".join(pred_strings)
    # -> "score1 x1 y1 w1 h1 score2 x2 y2 w2 h2 score3 x3 y3 w3 h3..."
    return str_pred_strings


def predict_object_detection(test_dataloader: DataLoader, model: FasterRCNN):
    '''
    推論結果を取得して、画像+bbox(推論結果)としてpng出力する関数
    '''
    # 使用可能なデバイスを指定。
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # detection閾値の設定
    detection_threshold = 0.7
    results = []

    for images, image_ids in test_dataloader:
        # input側のデータを取得.
        images: List[Tensor]
        # デバイスにTensorを渡す処理
        images = list(image.to(device) for image in images)

        # 推論の処理
        # modelをデバイスに渡す処理
        model.to(device)
        # modelを推論モードへ
        model.eval()
        # 推論(outputを取得)
        outputs = model(images)
        outputs: List[Dict[str, Tensor]]

        # 1バッチ内の各画像の物体検出結果をまとめる
        for i, image in enumerate(images):

            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()

            # スコアが閾値より高いbboxのみを残す。
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            # スコアも、選択したbboxと同じ長さにしておく
            scores = scores[scores >= detection_threshold]

            # 画像idを取得
            image_id = image_ids[i]

            # bboxes内の各bboxの座標の形式を変更(x, y, x+w, y+h)を＝＞(x, y, w, h)に？
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes=boxes,
                                                              scores=scores)
            }
            results.append(result)

    # 最終的に各画像の(id, 物体検出結果(bboxes, scores))が格納されたリストがretun
    results: List[Dict[str, str]]
    
    return results
