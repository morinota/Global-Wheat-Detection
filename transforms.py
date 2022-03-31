from torchvision import transforms
import albumentations as A
def get_train_transform()->transforms.Compose:
    '''
    画像にどのような変形を加えるか(後にDatasetオブジェクトに渡す)
    '''
    return transforms.Compose([
        # Tensor型に変換
        transforms.ToTensor()
        # その他処理があれば記述する。
    ])

def get_valid_transform()->transforms.Compose:
    '''
    画像にどのような変形を加えるか(後にDatasetオブジェクトに渡す)
    '''
    return transforms.Compose([
        # Tensor型に変換
        transforms.ToTensor()
        # その他処理があれば記述する。
    ])