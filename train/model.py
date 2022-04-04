import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
import torch


def create_model() -> FasterRCNN:
    # 学習済みモデルを読み込み
    model: FasterRCNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # モデル構成を見てみる(スライス表記で、一番最後のブロックだけ)
    # print(list(model.children())[-1])
    # ->RolHeadという名前で、中に座標回帰と分類器が格納されているbox_predictorがいる。
    '''
    デフォルトではCOCOデータセットで学習がされているので、
    - 分類器の出力は91。
    - 座標回帰の出力は4×91=364(それぞれのクラスに対してbboxの4座標)
    今回はこれを「小麦/背景」の2クラス分類器にしたいので、
    - 分類器の出力は2
    - 座標回帰の出力は4 * 2 = 8
    にしたい。よって以下の様に書き換える。
    '''
    # FasterRCNNPredictorの引数に入力チャネル数と出力チャネル数を指定する事で変更できる。
    # それぞれを取得して書き換えている。(入力チャネル数は取得しただけ?)
    # 出力チャネル数(クラス数)＝ 1 class(小麦) + 背景
    num_classes = 2
    # 入力チャネル数(取得しただけ?)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print(in_features)

    # 書き換え
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features, num_classes=num_classes)

    # 改めて、モデル構成の最後のブロックを表示してみる.
    # print(list(model.children())[-1])

    return model


def main():
   create_model()


if __name__ == "__main__":
    main()
