from torch.utils.data import DataLoader
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN


def show(val_dataloader: DataLoader, model: FasterRCNN):
    '''
    推論結果をチェックする関数
    '''
    import matplotlib.pyplot as plt
    from PIL import ImageDraw, ImageFont
    from PIL import Image

    # GPUのキャッシュクリア
    torch.cuda.empty_cache()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    # modelを推論モードへ
    model.eval()

    # modelに画像のTensorを渡せば推論を実行する(返値もTensor)
