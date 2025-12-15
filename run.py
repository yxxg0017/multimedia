import os
import glob
import re
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# ----------------- U2NET Imports (需确保 model.py 和 data_loader.py 在同一目录) -----------------
try:
    from data_loader import RescaleT, ToTensorLab, SalObjDataset
    from model import U2NET
except ImportError:
    print("错误: 缺少 model.py 或 data_loader.py，请确保 U2NET 项目文件完整。")
    exit()

# =================配置区域=================
INPUT_DIR = 'uploads'
OUTPUT_DIR = 'results'
MODEL_DIR = os.path.join(os.getcwd(), 'saved_models', 'u2net', 'u2net.pth')
# 阈值设置：黑色背景透明化阈值
TRANSPARENT_THRESHOLD = 15 
# =========================================

# ----------------- 1. U2NET 辅助函数 -----------------
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

class U2NETProcessor:
    def __init__(self, model_path):
        self.net = U2NET(3, 1)
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_path))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.net.eval()

    def infer(self, img_path):
        """
        运行 U2NET 推理，返回显著性图 (Numpy uint8 格式)
        """
        # 构建简易 Dataset (batch_size=1)
        test_salobj_dataset = SalObjDataset(
            img_name_list=[img_path],
            lbl_name_list=[],
            transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
        )
        test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=0)

        data_test = next(iter(test_salobj_dataloader))
        inputs_test = data_test['image'].type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        with torch.no_grad():
            d1, _, _, _, _, _, _ = self.net(inputs_test)

        # 归一化并恢复尺寸
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        
        # 转为 Numpy
        predict = pred.squeeze().cpu().data.numpy()
        
        # 恢复到原图尺寸
        original_img = Image.open(img_path)
        orig_w, orig_h = original_img.size
        
        # Resize 到原图大小
        im_pil = Image.fromarray(predict * 255).convert('L')
        im_pil = im_pil.resize((orig_w, orig_h), resample=Image.BILINEAR)
        
        return np.array(im_pil) # 返回 uint8 的 Mask 数组

# ----------------- 2. 纹饰提取逻辑 (来自 getall.py) -----------------
class OrnamentExtractor:
    def safe_grayscale(self, img):
        if len(img.shape) == 3:
            B = img[:, :, 0].astype(np.float32)
            G = img[:, :, 1].astype(np.float32)
            R_ch = img[:, :, 2].astype(np.float32)
            gray = (R_ch * 30 + G * 59 + B * 11 + 50) / 100
            return np.clip(gray, 0, 255).astype(np.uint8)
        return img

    def process(self, original_img_path, saliency_mask_np):
        """
        结合原图路径和内存中的 Mask 进行提取
        """
        I = cv2.imread(original_img_path) # BGR
        R = saliency_mask_np # Grayscale Mask
        
        if I is None:
            raise ValueError("无法读取图像文件")
            
        # 确保尺寸一致 (OpenCV 读取可能与 PIL 有微小差异，强制 Resize Mask)
        if I.shape[:2] != R.shape:
            R = cv2.resize(R, (I.shape[1], I.shape[0]))

        # --- 步骤 3: 提取全部纹饰 (corrected_step_three) ---
        I_G = self.safe_grayscale(I)
        
        if np.max(R) <= 1.0: # 防御性编程
            R = (R * 255).astype(np.uint8)
            
        _, R_binary = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        R_G = self.safe_grayscale(R) # 实际上 R已经是灰度，这里为了安全
        
        X = R_G.astype(np.int16) - I_G.astype(np.int16)
        
        # 动态阈值计算
        if X.max() <= 0:
            X_abs = np.abs(X)
            non_zero_mask = (R_binary > 0)
            if np.any(non_zero_mask):
                X_non_zero = X_abs[non_zero_mask]
                T = (np.mean(X_non_zero) + 50) / 2
            else:
                T = 25
            T = np.clip(T, 5, 100)
            Y = np.zeros_like(X_abs, dtype=np.uint8)
            Y[X_abs >= T] = 255
        else:
            non_zero_mask = (R_binary > 0)
            if np.any(non_zero_mask):
                X_non_zero = X[non_zero_mask]
                T = (np.mean(X_non_zero) + 150) / 2
            else:
                T = 75
            T = np.clip(T, 10, 200)
            Y = np.zeros_like(X, dtype=np.uint8)
            Y[X >= T] = 255

        final_mask = (Y > 0) & (R_binary > 0)
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        S = cv2.bitwise_and(I, I, mask=final_mask) # 全部纹饰
        Z = cv2.bitwise_and(I_G, I_G, mask=final_mask)

        # --- 步骤 4: 核心纹饰提取 (step_four) ---
        Zm = cv2.morphologyEx(Z, cv2.MORPH_CLOSE, kernel)
        _, binary = cv2.threshold(Zm, 1, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100: continue
            components.append({'mask': (labels == i).astype(np.uint8)})

        # 如果找不到组件，返回 S (全部纹饰)
        if not components:
            return S 

        # 高斯加权找核心
        height, width = Z.shape
        mu, sigma_sq = 0, 30000
        max_weight, core_component = -1, None
        
        for comp in components:
            total_weight = 0
            indices = np.where(comp['mask'] == 1)
            # 简化计算以加速
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
            return core_ornament
        
        return S # 降级返回

# ----------------- 3. 后处理: 透明化与重命名 (来自 tp.py) -----------------
def make_transparent(pil_img, threshold=15):
    img = pil_img.convert("RGBA")
    datas = img.getdata()
    new_data = []
    for item in datas:
        # 黑色 (R,G,B < threshold) 转透明
        if item[0] <= threshold and item[1] <= threshold and item[2] <= threshold:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    return img

def parse_and_rename(filename):
    base_name = os.path.splitext(filename)[0]
    match_num = re.search(r'(\d+)$', base_name)
    if match_num:
        number_str = match_num.group(1)
        new_number = int(number_str) - 1 # 数字减 1
        text_part = base_name[:match_num.start()].strip()
        parts = text_part.split()
        if not parts: return str(new_number)
        new_base_name = "_".join(parts) # 空格转下划线
        return f"{new_base_name}_{new_number}"
    return None

# ----------------- 主程序 -----------------
def main():
    # 0. 目录检查
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录 '{INPUT_DIR}' 不存在。")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 1. 初始化模型
    print("正在加载 U2NET 模型...")
    if not os.path.exists(MODEL_DIR):
        print(f"错误: 模型文件未找到: {MODEL_DIR}")
        return
    u2net = U2NETProcessor(MODEL_DIR)
    extractor = OrnamentExtractor()
    
    img_list = glob.glob(os.path.join(INPUT_DIR, '*.*'))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    img_list = [f for f in img_list if os.path.splitext(f)[1].lower() in valid_exts]
    
    print(f"开始处理 {len(img_list)} 张图片...")
    
    success_count = 0
    
    for i, img_path in enumerate(img_list):
        filename = os.path.basename(img_path)
        
        try:
            # 2. U2NET 推理 -> 获取 Mask
            mask_np = u2net.infer(img_path)
            
            # 3. 核心纹饰提取 -> 获取 OpenCV 图像 (BGR)
            ornament_bgr = extractor.process(img_path, mask_np)
            
            # 4. 转为 PIL 以进行透明化处理
            ornament_rgb = cv2.cvtColor(ornament_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(ornament_rgb)
            
            # 5. 背景透明化
            trans_img = make_transparent(pil_img, threshold=TRANSPARENT_THRESHOLD)
            
            # 6. 计算新文件名
            new_name_base = parse_and_rename(filename)
            if not new_name_base:
                # 如果不符合重命名规则，保留原名（去掉扩展名）
                new_name_base = os.path.splitext(filename)[0]
                
            save_path = os.path.join(OUTPUT_DIR, f"{new_name_base}.png")
            
            # 7. 保存
            trans_img.save(save_path, "PNG")
            success_count += 1
            # print(f"[{i+1}/{len(img_list)}] 完成: {filename} -> {os.path.basename(save_path)}")

        except Exception as e:
            print(f"[{i+1}/{len(img_list)}] 失败: {filename}, 错误: {e}")

    print("-" * 30)
    print(f"处理结束。成功: {success_count} / 总数: {len(img_list)}")
    print(f"结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()