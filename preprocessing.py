import pandas as pd
import numpy as np
import re
from typing import List


def devide_bbox(train_df: pd.DataFrame):
    '''
    1つのカラムに格納された「bboxの座標」を4つのカラムに格納しなおす
    '''
    # 座標カラムの初期値
    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    def _expand_bbox(x):
        '''
        boundy boxの座標を各カラムに分けて格納しなおす関数。apply()用
        '''
        r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
        if len(r) == 0:
            r = [-1, -1, -1, -1]
        return r

    # bboxの座標を4つのカラムに格納
    train_df[['x', 'y', 'w', 'h']] = np.stack(
        train_df['bbox'].apply(lambda x: _expand_bbox(x))
    )
    # 元のbboxカラムをdrop
    train_df.drop(columns=['bbox'], inplace=True)
    # 各座標のデータ型を指定
    train_df['x'] = train_df['x'].astype(np.float64)
    train_df['y'] = train_df['y'].astype(np.float64)
    train_df['w'] = train_df['w'].astype(np.float64)
    train_df['h'] = train_df['h'].astype(np.float64)

    return train_df

def conduct_holdout_method(train_df: pd.DataFrame):
    '''
    ホールドアウト法
    '''
    # 訓練用画像のユニークなidを取得
    image_ids = train_df['image_id'].unique()

    # まずは画像idを検証用と学習用に分割.(ホールドアウト法)
    valid_ids = image_ids[-665:]
    train_ids = image_ids[:-665]

    # 画像idの分割に基づいて、DataFrameも分割。
    valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    train_df = train_df[train_df['image_id'].isin(train_ids)]

    return train_df, valid_df


def my_preprocessing(train_df: pd.DataFrame):
    train_df = devide_bbox(train_df=train_df)
    train_df, valid_df = conduct_holdout_method(train_df)

    return train_df, valid_df
