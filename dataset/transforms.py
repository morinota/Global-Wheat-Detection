from torchvision import transforms
import albumentations as alb
from albumentations.pytorch import ToTensorV2


def get_train_transform() -> alb.Compose:
    '''
    画像にどのような変形を加えるか(後にDatasetオブジェクトに渡す)
    '''
    return alb.Compose([
        # Tensor型に変換
        ToTensorV2(),
        # その他処理があれば記述する。


    ],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )


def get_valid_transform() -> alb.Compose:
    '''
    画像にどのような変形を加えるか(後にDatasetオブジェクトに渡す)
    '''
    return alb.Compose([
        # Tensor型に変換
        ToTensorV2()
        # その他処理があれば記述する。
    ],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )


def get_test_transform() -> alb.Compose:
    transform_list = [
        # alb.Resize(512, 512),
        ToTensorV2(p=1.0)
    ]
    return alb.Compose(transforms=transform_list)
