import os



# Kaggle APIを通して提出
from kaggle import KaggleApi 

# predict process

def submit(csv_filepath:str, message:str):
    '''
    Kaggle competitionに結果をSubmitする関数
    '''
    os.environ['KAGGLE_USERNAME'] = "masatomasamasa"
    os.environ['KAGGLE_KEY'] = "5530a94bd76bac1415034d9f14cea01f"

    # KaggleApiインスタンスを生成
    api = KaggleApi()
    # 認証を済ませる.
    api.authenticate()

    compe_name = "global-wheat-detection"
    api.competition_submit(file_name=csv_filepath, message=message, competition=compe_name)

def main():
    # predict something on test dataset
    INPUT_DIR = r'input'
    # submit
    filepath = os.path.join(INPUT_DIR, 'submissions.csv')
    submit(csv_filepath=filepath, message='submission first')

if __name__ == '__main__':
    main()