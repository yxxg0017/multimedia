"""
单独运行的透明化脚本：
- 输入：outputs/final_output/core_ornaments 下的彩色核心纹饰（jpg/png）
- 输出：outputs/final_output/core_ornaments_transparent 下的透明 PNG
- 用法：python make_transparent.py --src ./outputs/final_output/core_ornaments --dst ./outputs/final_output/core_ornaments_transparent --threshold 15
"""

import argparse
import os
from pathlib import Path
from PIL import Image


def make_transparent(img: Image.Image, threshold: int = 15) -> Image.Image:
    """
    将黑色背景转为透明：RGB 都小于等于 threshold 的像素置为透明
    """
    rgba = img.convert("RGBA")
    data = []
    for r, g, b, a in rgba.getdata():
        if r <= threshold and g <= threshold and b <= threshold:
            data.append((0, 0, 0, 0))
        else:
            data.append((r, g, b, a))
    rgba.putdata(data)
    return rgba


def process_dir(src: Path, dst: Path, threshold: int = 15):
    dst.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png"}
    count = 0
    for f in sorted(src.iterdir()):
        if f.suffix.lower() not in exts:
            continue
        try:
            with Image.open(f) as im:
                out = make_transparent(im, threshold=threshold)
                out_path = dst / (f.stem + ".png")
                out.save(out_path, "PNG")
                count += 1
        except Exception as e:
            print(f"跳过 {f.name}: {e}")
    print(f"完成：{count} 个文件已输出到 {dst}")


def main():
    parser = argparse.ArgumentParser(description="核心纹饰黑底透明化")
    parser.add_argument("--src", default="./outputs/final_output/core_ornaments", help="输入目录")
    parser.add_argument("--dst", default="./outputs/final_output/core_ornaments_transparent", help="输出目录")
    parser.add_argument("--threshold", type=int, default=15, help="黑色阈值，默认15")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        print(f"输入目录不存在: {src}")
        return
    process_dir(src, dst, threshold=args.threshold)


if __name__ == "__main__":
    main()

