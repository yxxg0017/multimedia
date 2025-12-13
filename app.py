"""
基于 CSV 数据的全新前端入口：
- story.csv 提供各朝代叙事
- tags.csv 提供纹饰标签及详细字段
- 纹饰图片默认取自 outputs/final_output/core_ornaments/{tag}_core_color.jpg
"""

import csv
import json
import os
import subprocess
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from data_loader import RescaleT, ToTensorLab, SalObjDataset
    from getall import BatchOrnamentExtractor
    from model import U2NET, U2NETP
except ImportError:
    print("警告: 无法导入U2Net相关模块，单图片处理功能可能不可用")

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "outputs"
CORE_IMG_DIR = OUTPUT_DIR / "final_output" / "core_ornaments"
CORE_IMG_TRANSPARENT_DIR = OUTPUT_DIR / "final_output" / "core_ornaments_transparent"
ALL_IMG_DIR = OUTPUT_DIR / "final_output" / "all_ornaments"
RAW_IMG_DIR = OUTPUT_DIR / "u2net_results"

TAGS_CSV = DATA_DIR / "tags.csv"
STORY_CSV = DATA_DIR / "story.csv"


def _read_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    """
    尝试多种编码读取 CSV，返回 (header, rows)
    """
    encodings = ["utf-8-sig", "gbk", "utf-8"]
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = [row for row in reader]
            return header, rows
        except Exception:
            continue
    raise RuntimeError(f"无法读取 CSV 文件: {path}")


def _pick_image(tag: str) -> str:
    """
    根据 tag 选择可用图片路径（优先核心纹饰），返回供前端使用的相对 URL
    """
    candidates = [
        CORE_IMG_TRANSPARENT_DIR / f"{tag}_core_color.png",
        CORE_IMG_DIR / f"{tag}_core_color.jpg",
        CORE_IMG_DIR / f"{tag}_core_color.png",
        ALL_IMG_DIR / f"{tag}_all_color.jpg",
        RAW_IMG_DIR / f"{tag}.png",
    ]
    for p in candidates:
        if p.exists():
            rel = p.relative_to(BASE_DIR).as_posix()
            return f"/assets/{rel}"
    return ""


def load_story() -> Dict[str, str]:
    """
    读取 story.csv -> {dynasty: story_text}
    """
    if not STORY_CSV.exists():
        return {}
    header, rows = _read_csv_rows(STORY_CSV)
    if len(header) < 2:
        return {}

    story_map: Dict[str, str] = {}
    for row in rows:
        if len(row) < 2:
            continue
        dynasty = row[0].strip()
        text = row[1].strip()
        if dynasty:
            story_map[dynasty] = text
    return story_map


def load_tags():
    """
    读取 tags.csv，保留原始表头用于详情展示。
    约定：第 1 列为 tag，第 2 列为名称，第 3 列为朝代/时期。
    """
    header, rows = _read_csv_rows(TAGS_CSV)
    if len(header) < 3:
        raise RuntimeError("tags.csv 至少需要 3 列：tag、名称、朝代/时期")

    records = []
    for row in rows:
        if not row or len(row) < 3:
            continue
        tag = row[0].strip()
        name = row[1].strip()
        dynasty = row[2].strip()
        detail_pairs = []
        for idx, col in enumerate(header):
            if idx >= len(row):
                continue
            detail_pairs.append({"label": col.strip(), "value": row[idx].strip()})
        records.append(
            {
                "tag": tag,
                "name": name,
                "dynasty": dynasty,
                "details": detail_pairs,
                "image": _pick_image(tag),
            }
        )
    return header, records


def build_timeline():
    story_map = load_story()
    header, records = load_tags()

    # 定义朝代时间顺序（按历史先后）
    DYNASTY_ORDER = [
        "新石器时代",
        "商",
        "南北朝",
        "唐",
        "宋",
        "元",
        "明",
        "清",
    ]

    grouped = defaultdict(list)
    for rec in records:
        dynasty = rec["dynasty"].strip()
        grouped[dynasty].append(rec)

    timeline = []
    other_items = []
    
    # 按顺序处理已知朝代
    for dynasty in DYNASTY_ORDER:
        if dynasty in grouped:
            timeline.append(
                {
                    "dynasty": dynasty,
                    "story": story_map.get(dynasty, ""),
                    "items_list": grouped[dynasty],
                }
            )
    
    # 收集"其他"类别的朝代
    for dynasty, items in grouped.items():
        if dynasty not in DYNASTY_ORDER:
            other_items.extend(items)
    
    # 如果有"其他"类别，单独列出一个模块
    if other_items:
        timeline.append(
            {
                "dynasty": "其他",
                "story": "",
                "items_list": other_items,
            }
        )
    
    return header, timeline


@app.route("/assets/<path:filename>")
def assets(filename: str):
    """
    统一静态资源入口，指向项目根目录（主要用于 outputs 下的图片）
    """
    return send_from_directory(BASE_DIR, filename)


# 处理状态存储
processing_status = {
    "running": False,
    "step": "",
    "progress": 0,
    "message": "",
    "error": None,
    "result_image": None,
    "process_steps": {
        "original": None,
        "saliency": None,
        "extracted": None,
        "final": None
    }
}

# 上传文件存储目录
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)


def run_pipeline_async(images_dir: str, output_dir: str, model_path: str):
    """异步运行pipeline"""
    global processing_status
    try:
        processing_status["running"] = True
        processing_status["step"] = "初始化"
        processing_status["progress"] = 0
        processing_status["message"] = "正在启动处理流程..."
        processing_status["error"] = None

        # 调用pipeline.py
        script_path = BASE_DIR / "pipeline.py"
        cmd = [
            "python", str(script_path),
            "--images", images_dir,
            "--output", output_dir,
            "--model", model_path
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 读取输出并更新状态
        for line in process.stdout:
            line = line.strip()
            if "U2Net" in line or "显著性" in line:
                processing_status["step"] = "生成显著性图"
                processing_status["progress"] = 30
            elif "提取" in line or "纹饰" in line:
                processing_status["step"] = "提取纹饰"
                processing_status["progress"] = 60
            elif "透明化" in line or "透明" in line:
                processing_status["step"] = "背景透明化"
                processing_status["progress"] = 85
            elif "完成" in line or "成功" in line:
                processing_status["step"] = "完成"
                processing_status["progress"] = 100
            processing_status["message"] = line
        
        process.wait()
        
        if process.returncode == 0:
            processing_status["step"] = "完成"
            processing_status["progress"] = 100
            processing_status["message"] = "处理完成！"
        else:
            processing_status["error"] = "处理失败，返回码: " + str(process.returncode)
            
    except Exception as e:
        processing_status["error"] = str(e)
    finally:
        processing_status["running"] = False


@app.route("/api/process/start", methods=["POST"])
def start_processing():
    """启动处理流程"""
    global processing_status
    if processing_status["running"]:
        return jsonify({"error": "处理正在进行中"}), 400
    
    data = request.json
    images_dir = data.get("images_dir", str(BASE_DIR / "test_data" / "test_images"))
    output_dir = data.get("output_dir", str(BASE_DIR / "outputs"))
    model_path = data.get("model_path", str(BASE_DIR / "saved_models" / "u2net" / "u2net.pth"))
    
    thread = threading.Thread(
        target=run_pipeline_async,
        args=(images_dir, output_dir, model_path)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})


@app.route("/api/process/status")
def get_status():
    """获取处理状态"""
    return jsonify(processing_status)


def norm_pred(d):
    """归一化预测结果"""
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)


def safe_grayscale(img):
    """安全的灰度转换"""
    if len(img.shape) == 3:
        B = img[:, :, 0].astype(np.float32)
        G = img[:, :, 1].astype(np.float32)
        R_ch = img[:, :, 2].astype(np.float32)
        gray = (R_ch * 30 + G * 59 + B * 11 + 50) / 100
        return np.clip(gray, 0, 255).astype(np.uint8)
    return img


def make_transparent(pil_img, threshold=15):
    """将黑色背景转为透明"""
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


def process_single_image_async(image_path: str, model_path: str):
    """处理单张图片：从实物图到透明背景纹饰图"""
    global processing_status
    try:
        # 转换为绝对路径
        image_path_abs = str(Path(image_path).resolve())
        model_path_abs = str(Path(model_path).resolve())
        
        # 验证文件存在
        if not Path(image_path_abs).exists():
            raise ValueError(f"图片文件不存在: {image_path_abs}")
        if not Path(model_path_abs).exists():
            raise ValueError(f"模型文件不存在: {model_path_abs}")
        
        processing_status["running"] = True
        processing_status["step"] = "加载模型"
        processing_status["progress"] = 10
        processing_status["message"] = "正在加载U2Net模型..."
        processing_status["error"] = None
        processing_status["result_image"] = None
        processing_status["process_steps"] = {
            "original": None,
            "saliency": None,
            "extracted": None,
            "final": None
        }

        # 加载U2Net模型
        net = U2NET(3, 1)
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path_abs))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_path_abs, map_location="cpu"))
        net.eval()

        processing_status["step"] = "生成显著性图"
        processing_status["progress"] = 30
        processing_status["message"] = "正在生成显著性图..."
        
        # 保存原图路径
        result_id = uuid.uuid4().hex[:8]
        original_filename = f"original_{result_id}.jpg"
        original_path = RESULT_DIR / original_filename
        origin_img = Image.open(image_path_abs)
        origin_img.save(original_path, "JPEG", quality=95)
        processing_status["process_steps"]["original"] = f"/assets/results/{original_filename}"

        # 生成显著性图（使用绝对路径）
        dataset = SalObjDataset(
            img_name_list=[image_path_abs],
            lbl_name_list=[],
            transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        data_test = next(iter(dataloader))
        inputs_test = data_test["image"].type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        with torch.no_grad():
            d1, _, _, _, _, _, _ = net(inputs_test)

        pred = d1[:, 0, :, :]
        pred = norm_pred(pred)
        predict_np = pred.squeeze().cpu().data.numpy()
        mask_pil = Image.fromarray((predict_np * 255).astype(np.uint8)).convert("L")
        origin_img = Image.open(image_path_abs)
        mask_pil = mask_pil.resize(origin_img.size, resample=Image.BILINEAR)
        saliency_map = np.array(mask_pil)
        
        # 保存显著性图
        saliency_filename = f"saliency_{result_id}.png"
        saliency_path = RESULT_DIR / saliency_filename
        mask_pil.save(saliency_path, "PNG")
        processing_status["process_steps"]["saliency"] = f"/assets/results/{saliency_filename}"

        processing_status["step"] = "提取纹饰"
        processing_status["progress"] = 50
        processing_status["message"] = "正在提取核心纹饰..."

        # 读取原始图像
        I = cv2.imread(image_path_abs)
        R = saliency_map

        if I is None:
            raise ValueError(f"无法读取图像文件: {image_path_abs}，文件是否存在: {Path(image_path_abs).exists()}")

        # 确保尺寸一致
        if I.shape[:2] != R.shape:
            R = cv2.resize(R, (I.shape[1], I.shape[0]))

        # 步骤3: 提取全部纹饰
        I_G = safe_grayscale(I)

        if R.max() <= 1.0:
            R = (R * 255).astype(np.uint8)

        if np.max(R) == np.min(R):
            R_binary = np.ones_like(R, dtype=np.uint8)
        else:
            _, R_binary = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        R_G = safe_grayscale(R)
        X = R_G.astype(np.int16) - I_G.astype(np.int16)

        if X.max() <= 0:
            X_abs = np.abs(X)
            non_zero_mask = (R_binary > 0)
            if np.any(non_zero_mask):
                X_non_zero = X_abs[non_zero_mask]
                valid_values = X_non_zero[np.isfinite(X_non_zero)]
                X_mean = np.mean(valid_values) if len(valid_values) > 0 else 25
                T = (X_mean + 50) / 2
            else:
                T = 25
            T = np.clip(T, 5, 100)
            Y = np.zeros_like(X_abs, dtype=np.uint8)
            Y[X_abs >= T] = 255
        else:
            non_zero_mask = (R_binary > 0)
            if np.any(non_zero_mask):
                X_non_zero = X[non_zero_mask]
                valid_values = X_non_zero[np.isfinite(X_non_zero)]
                X_mean = np.mean(valid_values) if len(valid_values) > 0 else 75
                T = (X_mean + 150) / 2
            else:
                T = 75
            T = np.clip(T, 10, 200)
            Y = np.zeros_like(X, dtype=np.uint8)
            Y[X >= T] = 255

        final_mask = (Y > 0) & (R_binary > 0)
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        S = cv2.bitwise_and(I, I, mask=final_mask)
        Z = cv2.bitwise_and(I_G, I_G, mask=final_mask)
        
        # 保存提取的纹饰（带背景）
        extracted_filename = f"extracted_{result_id}.png"
        extracted_path = RESULT_DIR / extracted_filename
        extracted_rgb = cv2.cvtColor(S, cv2.COLOR_BGR2RGB)
        extracted_pil = Image.fromarray(extracted_rgb)
        extracted_pil.save(extracted_path, "PNG")
        processing_status["process_steps"]["extracted"] = f"/assets/results/{extracted_filename}"

        processing_status["step"] = "提取核心纹饰"
        processing_status["progress"] = 70
        processing_status["message"] = "正在定位核心纹饰..."

        # 步骤4: 核心纹饰提取
        Zm = cv2.morphologyEx(Z, cv2.MORPH_CLOSE, kernel)
        _, binary = cv2.threshold(Zm, 1, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:
                continue
            components.append({
                'mask': (labels == i).astype(np.uint8),
                'area': area
            })

        if len(components) == 0:
            core_ornament = S
        else:
            # 高斯加权定位核心
            height, width = Z.shape
            mu, sigma_sq = 0, 30000
            max_weight, core_component = -1, None

            for comp in components:
                total_weight = 0
                indices = np.where(comp['mask'] == 1)
                dists = np.sqrt((indices[1] - width/2)**2 + (indices[0] - height/2)**2)
                weights = (1 / np.sqrt(2 * np.pi * sigma_sq)) * np.exp(-(dists - mu)**2 / (2 * sigma_sq))
                total_weight = np.sum(weights)

                if total_weight > max_weight:
                    max_weight = total_weight
                    core_component = comp

            if core_component:
                core_mask = core_component['mask'] * 255
                core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                core_ornament = cv2.bitwise_and(S, S, mask=core_mask)
            else:
                core_ornament = S

        processing_status["step"] = "背景透明化"
        processing_status["progress"] = 90
        processing_status["message"] = "正在处理透明背景..."

        # 转换为PIL并透明化
        core_rgb = cv2.cvtColor(core_ornament, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(core_rgb)
        trans_img = make_transparent(pil_img, threshold=15)

        # 保存结果
        result_filename = f"result_{result_id}.png"
        result_path = RESULT_DIR / result_filename
        trans_img.save(result_path, "PNG")

        processing_status["step"] = "完成"
        processing_status["progress"] = 100
        processing_status["message"] = "处理完成！"
        processing_status["result_image"] = f"/assets/results/{result_filename}"
        processing_status["process_steps"]["final"] = f"/assets/results/{result_filename}"

    except Exception as e:
        processing_status["error"] = str(e)
        processing_status["message"] = f"处理失败: {str(e)}"
    finally:
        processing_status["running"] = False


@app.route("/api/process/upload", methods=["POST"])
def upload_and_process():
    """上传图片并开始处理"""
    global processing_status
    if processing_status["running"]:
        return jsonify({"error": "处理正在进行中"}), 400

    if "file" not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    # 保存上传的文件
    filename = f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
    upload_path = UPLOAD_DIR / filename
    file.save(str(upload_path))
    
    # 验证文件是否保存成功
    if not upload_path.exists():
        return jsonify({"error": "文件保存失败"}), 500
    
    # 验证文件是否为有效图片
    try:
        test_img = Image.open(str(upload_path))
        test_img.verify()
    except Exception as e:
        return jsonify({"error": f"无效的图片文件: {str(e)}"}), 400

    # 获取模型路径（使用绝对路径）
    model_path = request.form.get("model_path", str(BASE_DIR / "saved_models" / "u2net" / "u2net.pth"))
    model_path = str(Path(model_path).resolve())

    # 异步处理（使用绝对路径）
    thread = threading.Thread(
        target=process_single_image_async,
        args=(str(upload_path.resolve()), model_path)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "upload_id": filename})


@app.route("/")
def gallery():
    header, timeline = build_timeline()
    return render_template("gallery.html", header=header, timeline=timeline)


if __name__ == "__main__":
    app.run(host="::", port=5000, debug=True)