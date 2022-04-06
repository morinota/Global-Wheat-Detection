from unittest import result
from torch.utils.data import DataLoader
from dataset.my_dataset import MyDataset
import torch

import os
import pandas as pd
from load_data.data import load_data
from dataset.transforms import get_train_transform, get_valid_transform, get_test_transform
from load_data.preprocessing import my_preprocessing
from scipy.__config__ import show
from train.model import create_model
from images.images import visualize_image_and_bboxes
from train.train import train_model
from images.test import show_images_bbox_predicted
from predict.predict import predict_object_detection
from predict.submit import submit


def main():
    # KaggleAPIからデータロード
    load_data()

    INPUT_DIR = 'input'
    image_dir_train = os.path.join(INPUT_DIR, 'train')
    image_dir_test = os.path.join(INPUT_DIR, 'test')

    # 提出用のcsvファイルを読み込み
    test_df = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))

    test_dataset = MyDataset(dataframe=test_df,
                             image_dir=image_dir_test,
                             transforms=get_test_transform()
                             )

    # DataLoaderのバッチ出力の形式を自作(デフォルトはリストだけど)

    def collate_fn(batch):
        # リストをアンパッキングしてる！　それぞれの要素を分けて取得しやすい為に？？
        return tuple(zip(*batch))

    # 推論用のDataloaderの作成
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=4,  # ミニバッチの個数(小さい方がメモリ節約できる)
        shuffle=False,  # ミニバッチをDatasetからランダムに取り出すかどうか
        drop_last=False,  # 残りのデータ数がバッチサイズより少ない場合、使わないかどうか
        num_workers=2,  # 複数処理をするかどうか,
        # バッチ出力の形を指定(Defaultは各要素(画像、ラベル等)がリストで固められる)
        collate_fn=collate_fn
    )  # -> イテラブルオブジェクト

    # 使用可能なデバイスを指定。
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # モデルを構築(定義しなおし)
    model = create_model()
    # 学習済みのパラメータを読み込み
    model_path = 'model.pth'
    model.load_state_dict(torch.load(model_path))

    # 予測値bboxの画像出力
    # show_images_bbox_predicted(test_dataloader=valid_dataloader,
    #                            model=model,
    #                            image_i=0)

    # 推論
    results = predict_object_detection(test_dataloader=test_dataloader,
                                       model=model
                                       )

    # 推論結果をDataFrameにまとめる
    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'],)
    test_df.to_csv('submission.csv', index=False)

    # 結果をkaggleAPIを通してSubmit
    submit(csv_filepath='submission.csv', message='first submission')

if __name__ == '__main__':
    main()
