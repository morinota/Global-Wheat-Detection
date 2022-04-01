from torch.utils.data import DataLoader
from dataset.dataset import MyDataset
import torch
from torchvision import transforms
import os
import pandas as pd
from data import load_data
from transforms import get_train_transform, get_valid_transform
from preprocessing import my_preprocessing


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
        batch_size=64,  # ミニバッチの個数
        shuffle=False,  # ミニバッチをDatasetからランダムに取り出すかどうか
        drop_last=True,  # 残りのデータ数がバッチサイズより少ない場合、使わないかどうか
        num_workers=2,  # 複数処理をするかどうか,
        # バッチ出力の形を指定(Defaultは各要素(画像、ラベル等)がリストで固められる)
        collate_fn=collate_fn
    )  # -> イテラブルオブジェクト

    # 検証用のDataLoaderの作成
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False,)

    # GPU使うかどうか
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(device)
    else:
        device = torch.device('cpu')
        print(device)

    # OpenCVで描画
    # DataLoaderからデータセットを取得
    images, targets, image_ids = next(iter(train_dataloader))
    print(len(images))  # ->trainデータ2708枚の内、batch_size枚がリターン.
    print(len(targets))
    print(type(images))
    # (実行する度に、リターンされるデータセットが変わる?)

    # データ型を変更してる?to(device)ってなんだ?
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    print(type(images))


if __name__ == '__main__':
    main()
