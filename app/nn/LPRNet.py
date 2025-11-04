import torch.nn as nn
import torch
import cv2
import numpy as np
from typing import List
from PIL import Image
import os
import random
from torch.utils.data import DataLoader, Dataset
from imutils import paths


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


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
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


def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase:
        return Net.train()
    else:
        return Net.eval()


class LPRPredictor:
    def __init__(self, model_path: str, cuda: bool = True, lpr_max_len: int = 8, dropout_rate: float = 0):
        """初始化模型"""
        self.device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")
        self.model = build_lprnet(
            lpr_max_len=lpr_max_len,
            phase=False,
            class_num=len(CHARS),
            dropout_rate=dropout_rate
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, img_path: str, img_size=(94, 24)):
        """读取并预处理单张图片"""
        img = Image.open(img_path).convert("RGB")
        img = img.resize(img_size)
        img = np.array(img).astype("float32")
        img = img / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(img).unsqueeze(0)

    def decode(self, prebs: np.ndarray) -> List[str]:
        """Greedy Decode 还原车牌字符串"""
        preb_labels = list()
        # print(prebs.shape)  # (128, 68, 18)
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        res = list()
        for i, label in enumerate(preb_labels):
            lb = ""
            for i in label:
                lb += CHARS[i]
            res.append(lb)
        return res

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, source: str | np.ndarray | Image.Image, img_size=(94, 24)) -> List[dict]:
        """执行预测，支持单张图片或文件夹"""
        results = []

        if isinstance(source, Image.Image):  # PIL图片
            img = source.resize(img_size)
            img = np.array(img).astype("float32")
            img = img / 255.0
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prebs = self.model(img_tensor)
            prebs = prebs.cpu().numpy()
            plate = self.decode(prebs)[0]
            results.append({"path": "PIL_Image", "plate": plate})

        elif isinstance(source, np.ndarray):  # cv2图片
            img = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize(img_size)
            img = np.array(img).astype("float32")
            img = img / 255.0
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prebs = self.model(img_tensor)
            prebs = prebs.cpu().numpy()
            plate = self.decode(prebs)[0]
            results.append({"path": "cv2_Image", "plate": plate})

        elif isinstance(source, str):
            if os.path.isfile(source):  # 单张图片
                img_tensor = self._preprocess(source, img_size).to(self.device)
                with torch.no_grad():
                    prebs = self.model(img_tensor)
                prebs = prebs.cpu().numpy()
                plate = self.decode(prebs)[0]
                results.append({"path": source, "plate": plate})

            elif os.path.isdir(source):  # 文件夹
                dataset = LPRDataLoader([source], img_size, lpr_max_len=8)
                dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0,
                                        collate_fn=lambda b: (torch.stack([torch.from_numpy(x[0]) for x in b], 0), None,
                                                              None))

                for batch_i, (images, _, _) in enumerate(dataloader):
                    images = images.to(self.device)
                    with torch.no_grad():
                        prebs = self.model(images)
                    prebs = prebs.cpu().numpy()
                    preds = self.decode(prebs)

                    batch_paths = dataset.imgs[batch_i * len(preds): batch_i * len(preds) + len(preds)]
                    for path, plate in zip(batch_paths, preds):
                        results.append({"path": path, "plate": plate})

            else:
                raise ValueError(f"source 路径无效: {source}")

        return results


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        # Image = cv2.imread(filename)
        Image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c])

        if len(label) == 8:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True
