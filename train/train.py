from torch import Tensor
from torchvision.models.detection.faster_rcnn import FasterRCNN
import torch
from torch.utils.data import DataLoader
from typing import List, Dict
from tqdm import tqdm

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    # プロパティの値を取り出すメソッドを定義。
    # プロパティは、インスタンス変数の様に自然に値にアクセスでき、かつ外から値を変更しづらい。
    @property
    def value(self):  # プロパティ名はvalue?
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def train_model(model: FasterRCNN, train_dataloader: DataLoader):
    # 演算を行うデバイスを設定
    # GPUを使えるかどうかに応じてtorch.deviceオブジェクトを生成
    # torch.deviceはテンソルをどのデバイスに割り当てるかを表すクラス。
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # modelをGPUに渡す
    model.to(device)

    # 更新すべきパラメータと最適化手法の指定
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params=params,
                                lr=0.005,
                                momentum=0.9,
                                weight_decay=0.0005
                                )
    num_epochs = 2
    lr_scheduler = None

    # GPUのキャッシュクリア
    torch.cuda.empty_cache()

    # 学習モードに移行
    model.train()
    # 学習
    for epoch in range(num_epochs):

        for i, batch in enumerate(tqdm(train_dataloader)):

            # batchにはそのミニバッチのimages, targets, image_idsが入ってる。
            images: List[Tensor]
            targets: List[Dict[str, Tensor]]
            images, targets, image_ids = batch

            # 指定のdevice(=GPU)にTensorを転送する(ListやDictにTensorが入ってるから)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            # 誤差関数の値を取得?
            # 学習モードでは画像とターゲット(ground-truch=正解値)を入力する
            # 返値はDict[str, Tensor]でlossが入ってる。(RPNとRCNN両方のloss)
            loss_dict = model(images, targets)

            # RPNとRCNN両方のlossを足し合わせてる?
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            optimizer.zero_grad() # 前のバッチで計算されたgradをリセット
            losses.backward() # 誤差逆伝搬で、各パラメータの勾配gradの値を計算(実は累算してる!だからzero_gradを使ってる)
            optimizer.step() # - grad＊学習率を使って、パラメータを更新

            # 50の倍数のiterationの時、その時のlossの値を出力.
            if (i+1) % 50 == 0:
                print(f"Iteration #{i+1} loss: {loss_value}")

        # update the laerning rate:
        if lr_scheduler is not None:
            lr_scheduler.step()

    # 保存
    model_path = 'model.pth'
    # model.state_dict()として保存した方が無駄な情報を削れてファイルサイズを小さくできるらしい.
    torch.save(model.state_dict(), model_path)

    # 返値としても出力
    return model
