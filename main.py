from torch.utils.data import DataLoader
from dataset.dataset import MyDataset
import torch
from torchvision import transforms
import os
import pandas as pd
from data import load_data


def main():
    # KaggleAPIからデータロード
    load_data()

    # 画像にどのような変形を加えるか(後にDatasetオブジェクトに渡す)
    transform = transforms.Compose([
        # Tensor型に変換
        transforms.ToTensor()
    ])

    # MyDatasetオブジェクトを生成
    INPUT_DIR = 'input'
    image_dir_train = os.path.join(INPUT_DIR, 'train')
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))

    train_dataset = MyDataset(dataframe=train_df,
                              image_dir=image_dir_train,
                              transforms=transform
                              )

    # 学習用のDataloaderの作成
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=True
    )
    # # 検証用のDataLoaderの作成
    # valid_dataloader = DataLoader(
    #     dataset=
    # )

    # GPU使うかどうか
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(device)
    else:
        device = torch.device('cpu')
        print(device)


    # OpenCVで描画
    # DataLoaderからデータセットを取得
    imges, targets, image_ids = next(iter(train_dataloader))
