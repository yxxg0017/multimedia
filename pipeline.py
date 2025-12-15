import argparse
import glob
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from getall import BatchOrnamentExtractor
from model import U2NET, U2NETP
from PIL import Image


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)


class SaliencyGenerator:
    """
    负责使用 U2Net 推理生成显著性图的类
    """

    def __init__(self, model_path: str, model_name: str = "u2net", use_gpu: Optional[bool] = None):
        self.model_path = model_path
        self.model_name = model_name
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        self.net = self._load_model()

    def _load_model(self):
        if self.model_name == "u2netp":
            net = U2NETP(3, 1)
        else:
            net = U2NET(3, 1)

        if self.use_gpu:
            net.load_state_dict(torch.load(self.model_path))
            net.cuda()
        else:
            net.load_state_dict(torch.load(self.model_path, map_location="cpu"))

        net.eval()
        return net

    def _collect_images(self, image_dir: str) -> List[str]:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        img_list: list[str] = []
        for ext in exts:
            img_list.extend(glob.glob(os.path.join(image_dir, ext)))
        # 过滤掉无法识别的文件，避免 PIL 报错
        valid_list: list[str] = []
        for path in sorted(img_list):
            try:
                with Image.open(path) as im:
                    im.verify()  # 仅校验，不加载
                valid_list.append(path)
            except Exception as exc:  # noqa: BLE001
                print(f"跳过无法识别的文件: {path}，原因: {exc}")
        return valid_list

    def run_dir(self, image_dir: str, prediction_dir: str) -> List[str]:
        """
        对目录内所有图片生成显著性图并保存，返回输出文件列表
        """
        os.makedirs(prediction_dir, exist_ok=True)
        img_name_list = self._collect_images(image_dir)

        test_salobj_dataset = SalObjDataset(
            img_name_list=img_name_list,
            lbl_name_list=[],
            transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]),
        )
        dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=0)

        saved_paths: list[str] = []
        for i_test, data_test in enumerate(dataloader):
            inputs_test = data_test["image"].type(torch.FloatTensor)
            if self.use_gpu:
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            with torch.no_grad():
                d1, _, _, _, _, _, _ = self.net(inputs_test)

            pred = d1[:, 0, :, :]
            pred = norm_pred(pred)

            predict_np = pred.squeeze().cpu().data.numpy()
            mask_pil = Image.fromarray((predict_np * 255).astype(np.uint8)).convert("L")

            origin_img = Image.open(img_name_list[i_test])
            mask_pil = mask_pil.resize(origin_img.size, resample=Image.BILINEAR)

            img_name = os.path.basename(img_name_list[i_test])
            base = os.path.splitext(img_name)[0]
            out_path = os.path.join(prediction_dir, f"{base}.png")
            mask_pil.save(out_path)
            saved_paths.append(out_path)

        return saved_paths


def run_pipeline(images_dir: str, output_root: str, model_path: str, model_name: str = "u2net") -> bool:
    """
    统一管线：先跑 U2Net 生成显著性图，再用 BatchOrnamentExtractor 提取纹饰
    """
    saliency_dir = os.path.join(output_root, "u2net_results")
    final_output_dir = os.path.join(output_root, "final_output")

    generator = SaliencyGenerator(model_path=model_path, model_name=model_name)
    saliency_files = generator.run_dir(images_dir, saliency_dir)
    if not saliency_files:
        print("未在输入目录找到可处理的图片。")
        return False

    extractor = BatchOrnamentExtractor(
        original_images_dir=images_dir,
        saliency_maps_dir=saliency_dir,
        output_dir=final_output_dir,
    )
    success = extractor.run_batch_processing()
    if not success:
        return False

    # 额外步骤：核心纹饰透明化，输出 PNG
    core_dir = os.path.join(final_output_dir, "core_ornaments")
    transparent_dir = os.path.join(final_output_dir, "core_ornaments_transparent")
    os.makedirs(transparent_dir, exist_ok=True)

    def make_transparent(pil_img: Image.Image, threshold: int = 15) -> Image.Image:
        img = pil_img.convert("RGBA")
        data = img.getdata()
        new_data = []
        for item in data:
            if item[0] <= threshold and item[1] <= threshold and item[2] <= threshold:
                new_data.append((0, 0, 0, 0))
            else:
                new_data.append(item)
        img.putdata(new_data)
        return img

    for fname in os.listdir(core_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        src_path = os.path.join(core_dir, fname)
        try:
            with Image.open(src_path) as img:
                trans = make_transparent(img, threshold=15)
                base = os.path.splitext(fname)[0]
                dst_path = os.path.join(transparent_dir, f"{base}.png")
                trans.save(dst_path, "PNG")
        except Exception as exc:
            print(f"透明化失败 {fname}: {exc}")

    return True


def build_parser():
    parser = argparse.ArgumentParser(description="U2Net + 纹饰提取一键流水线")
    parser.add_argument(
        "--images",
        default="./test_data/test_images",
        help="原始图片目录（支持 jpg/png/bmp/tiff 等）",
    )
    parser.add_argument(
        "--output",
        default="./outputs",
        help="输出根目录（会在内部创建 u2net_results 与 final_output 子目录）",
    )
    parser.add_argument(
        "--model",
        default=os.path.join(os.getcwd(), "saved_models", "u2net", "u2net.pth"),
        help="U2Net 权重路径",
    )
    parser.add_argument(
        "--model-name",
        default="u2net",
        choices=["u2net", "u2netp"],
        help="选择模型大小：u2net / u2netp",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    ok = run_pipeline(
        images_dir=args.images,
        output_root=args.output,
        model_path=args.model,
        model_name=args.model_name,
    )
    if ok:
        print("✅ 全部流程完成，结果已写入输出目录。")
    else:
        print("❌ 流程未完成，请检查日志与路径。")


if __name__ == "__main__":
    main()

