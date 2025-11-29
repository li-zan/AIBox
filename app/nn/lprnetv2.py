import torch
import torch.nn as nn
import numpy as np
from typing import List
import cv2


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class CBR(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 2), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class LPRNetV2(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNetV2, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            small_basic_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            # nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            CBR(128, 64),
            small_basic_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            CBR(256, 64),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '学', '港', '澳', '警', '使', '领', '应', '急', '挂',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-'
         ]


class LPRv2Predictor:
    def __init__(self, model_path: str, cuda: bool = True, lpr_max_len: int = 8,
                 dropout_rate: float = 0):
        """初始化模型"""
        self.device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")
        self.model = LPRNetV2(lpr_max_len, True, class_num=len(CHARS), dropout_rate=dropout_rate).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)["model"])
        self.model.to(self.device)
        self.model.eval()

    def decode(self, preds):
        last_chars_idx = len(CHARS) - 1

        # greedy decode
        pred_labels = []
        labels = []
        for i in range(preds.shape[0]):
            pred = preds[i, :, :]
            pred_label = []
            for j in range(pred.shape[1]):
                pred_label.append(np.argmax(pred[:, j], axis=0))
            no_repeat_blank_label = []
            pre_c = -1
            for c in pred_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == last_chars_idx):
                    if c == last_chars_idx:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            pred_labels.append(no_repeat_blank_label)

        for _, label in enumerate(pred_labels):
            lb = ""
            for i in label:
                lb += CHARS[i]
            labels.append(lb)

        return labels, pred_labels

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, image: np.ndarray, img_size=(94, 24)) -> List[dict]:
        """执行预测，支持单张图片（np.ndarray格式），返回车牌字符串"""
        self.model.eval()
        image = cv2.resize(image, img_size)
        image = (image.astype('float32') - 127.5) * 0.007843
        image = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            x = self.model(image).cpu().detach().numpy()
            pred_labels, _ = self.decode(x)
        return pred_labels[0]
