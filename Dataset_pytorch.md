# Pytorch – 自作のデータセットを扱う Dataset クラスを作る方法

## 概要

- Pytorch で自作のデータセットを扱うには、**Dataset クラスを継承したクラスを作成**する必要がある。

## Dataset クラス

Dataset クラスでは、画像やcsvファイルといったリソースで構成されるデータセットから、データを取得する方法について定義する。
基本的には`index`のサンプルが要求(指定?)された時に返す` __getitem__(self, index)`と、データセットのサンプル数が要求された時に返す`__len__(self)`の**2つのメソッドを実装する必要がある**。

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __getitem__(self, index):
        # インデックス index のサンプルが要求されたときに返す処理を実装

    def __len__(self):
        # データセットのサンプル数が要求されたときに返す処理を実装
```

## 参考

- https://pystyle.info/pytorch-how-to-create-custom-dataset-class/
- https://torch.classcat.com/category/object-detection/
- https://medium.com/@hei4/pytorch%E5%88%9D%E5%BF%83%E8%80%85%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AEmnist%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA-35cdfb108f7d
