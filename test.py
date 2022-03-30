import torch
import pandas as pd
import cv2
from cv2 import imread
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    '''
    物体検出用データセット。
    Args:
        Dataset (Dataset): torch.utils.data.Dataset
    '''

    def __init__(self, dataframe: pd.DataFrame, image_dir: str, transforms:= None) -> None:
        """
        Args:
            dataframe (pd.DataFrame): 画像idとbbox座標が格納されたdf
            image_dir (str): 画像データのルートディレクトリ
            transform (BaseCompose, optional): 前処理とデータ拡張. Defaults to None.
        """
        # Datasetクラスのコンストラクタはそのまま。
        super().__init__()

        # オリジナルの部分
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        """インデックスからデータを取得する
        Args:
            index (int): 取得するデータのインデックス
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            画像、バウンディングボックス、ラベル、キーポイントマップ、オフセットマップ、サイズマップ。
            画像のTensor形状は[1, 28, 28]
            バウンディングボックスのTensor形状は[M, 4]。Mはボックス個数だがM=1
            ラベルのTensor形状は[M]
            キーポイントマップのTensor形状は[10, 7, 7]
            オフセットマップのTensor形状は[2, 7, 7]
            サイズマップのTensor形状は[2, 7, 7]
        """
        # 以下は画像データ1つに対して行う処理のイメージ?

        # 指定されたindexの画像idを取得
        image_id = self.image_ids[index]
        # 出力側のbboxの座標をdfから抽出。
        mask = self.df['image_id'] == image_id
        records = self.df[mask]  # ->df.DataFrame

        # 入力側の画像データを読み込み
        image = cv2.imread(
            f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        # 変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 正規化
        image /= 255.0

        return super().__getitem__(index)

    def __len__(self) -> int:
        """データセットのサイズを取得する

        Returns:
            int: データセットのサイズ
        """
        return super().__len__()


def main():
    print(cv2.__version__)
    pass


if __name__ == '__main__':
    main()
