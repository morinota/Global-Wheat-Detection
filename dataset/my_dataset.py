from matplotlib.pyplot import box
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
import cv2


class MyDataset(Dataset):
    '''
    物体検出用データセット。
    Args:
        Dataset (Dataset): torch.utils.data.Dataset
    '''

    def __init__(self, dataframe: pd.DataFrame,
                 image_dir: str,
                 transforms=None) -> None:
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
        """インデックスを渡されてデータを返す
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
        path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = cv2.imread(filename=path, flags=cv2.IMREAD_COLOR)
        # BGRからRGBに配列の順序を変換?
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 正規化?
        image /= 255.0

        # bbox座標の形式変換(DataFrameからndarrayへ)
        # -> (画像内の小麦の数, 4)のndarray
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes: np.ndarray
        # ->各列が(x, y, x+w, y+h)のndarrayへ。
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # wをx+wへ変換
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # hをy+hへ変換

        # bboxのエリアを計算(縦×横)(精度評価指標の計算で用いる)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # areaをndarrayからTensorオブジェクトへ変換?
        area = torch.as_tensor(area, dtype=torch.float32)

        # クラスラベルの指定(今回は1クラスのみ)
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        # 返り値用に、生成した変数をdictにまとめる。
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        # 前処理を定義していれば...
        if self.transforms:
            # 前処理を行う
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': target['labels'],
            }
            sample = self.transforms(**sample)
            image = sample['image']

            # 前処理に合わせて、bboxの値を修正してる?
            target['boxes'] = torch.stack(
                tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        """データセットのサイズを取得する

        Returns:
            int: データセットのサイズ
        """
        return self.image_ids.shape[0]


class MyDataset_for_predict(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(
            f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
