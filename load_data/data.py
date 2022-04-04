from sympy import Q
from kaggle import KaggleApi
import shutil
import os
os.environ['KAGGLE_USERNAME'] = "masatomasamasa"
os.environ['KAGGLE_KEY'] = "5530a94bd76bac1415034d9f14cea01f"

def load_data():
    '''
    Kaggle competitionのデータセットを読み込む関数
    '''
    # KaggleApiインスタンスを生成
    api = KaggleApi()
    # 認証を済ませる.
    api.authenticate()

    # コンペ一覧
    # print(api.competitions_list(group=None, category=None,
    #       sort_by=None, page=1, search=None))


    # # 特定のコンペのデータを取得
    compe_name = "global-wheat-detection"

    file_list = api.competition_list_files(competition=compe_name)
    print(file_list)

    # # csvファイルだけを抽出したい
    # file_list_csv = [file.name for file in file_list if '.csv' in file.name]
    # print(file_list_csv)

    # 対象データを読み込み
    INPUT_DIR = r"input"

    api.competition_download_files(competition=compe_name, path=INPUT_DIR)
    # 保存されるファイル名はf'{compe_name}.zip'

    # zipファイルをunpacking
    shutil.unpack_archive(filename=os.path.join(INPUT_DIR, f'{compe_name}.zip'),
                            extract_dir=INPUT_DIR)
