from torch.utils.data import DataLoader
from dataset.my_dataset import MyDataset
import torch

import os
import pandas as pd
from load_data.data import load_data
from dataset.transforms import get_train_transform, get_valid_transform
from load_data.preprocessing import my_preprocessing
from scipy.__config__ import show
from train.model import create_model
from images.images import visualize_image_and_bboxes
from train.train import train_model
from images.test import show_images_bbox_predicted


def main():
    # KaggleAPIからデータロード
    load_data()

    # 出力値側のcsvデータを読み込み
    INPUT_DIR = 'input'
    image_dir_train = os.path.join(INPUT_DIR, 'train')
    image_dir_train = os.path.join(INPUT_DIR, 'train')

    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))

    # DataFrame側の前処理
    train_df, valid_df = my_preprocessing(train_df)

    # MyDatasetオブジェクトを生成
    train_dataset = MyDataset(dataframe=train_df,
                              image_dir=image_dir_train,
                              transforms=get_train_transform()
                              )
    valid_dataset = MyDataset(dataframe=valid_df,
                              image_dir=image_dir_train,
                              transforms=get_valid_transform()
                              )

    # バッチ出力の形式を自作(デフォルトはリストだけど)
    def collate_fn(batch):
        # リストをアンパッキングしてる！　それぞれの要素を分けて取得しやすい為に？？
        return tuple(zip(*batch))

    # 学習用のDataloaderの作成
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=10,  # ミニバッチの個数(小さい方がメモリ節約できる)
        shuffle=False,  # ミニバッチをDatasetからランダムに取り出すかどうか
        drop_last=True,  # 残りのデータ数がバッチサイズより少ない場合、使わないかどうか
        num_workers=2,  # 複数処理をするかどうか,
        # バッチ出力の形を指定(Defaultは各要素(画像、ラベル等)がリストで固められる)
        collate_fn=collate_fn
    )  # -> イテラブルオブジェクト

    # 検証用のDataLoaderの作成
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False,)

    # 使用可能なデバイスを指定。
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # モデルを構築(定義しなおし)
    model = create_model()
    # 学習済みのパラメータを読み込み
    model_path = 'model.pth'
    model.load_state_dict(torch.load(model_path))

    # 予測値bboxの画像出力
    show_images_bbox_predicted(test_dataloader=valid_dataloader,
                               model=model,
                               image_i=0)


if __name__ == '__main__':
    main()
