import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
from scipy import ndimage
import glob

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BatchOrnamentExtractor:
    """
    批量陶瓷纹饰提取器
    处理指定目录下的所有图像，并保存结果到输出目录
    """
    
    def __init__(self, original_images_dir, saliency_maps_dir, output_dir="./final_output/"):
        """
        初始化批量提取器
        
        Args:
            original_images_dir: 原始图像目录路径
            saliency_maps_dir: 显著性图目录路径  
            output_dir: 输出结果目录路径
        """
        self.original_images_dir = original_images_dir
        self.saliency_maps_dir = saliency_maps_dir
        self.output_dir = output_dir
        
        # 创建输出目录结构
        self._create_output_structure()
        
        # 统计信息
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.results = []
    
    def _create_output_structure(self):
        """创建输出目录结构"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建子目录
        sub_dirs = [
            "all_ornaments",      # 全部纹饰结果
            "core_ornaments",     # 核心纹饰结果  
            "reports",           # 提取报告
            "visualizations",    # 可视化结果
            "masks"             # 掩模文件
        ]
        
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(self.output_dir, sub_dir), exist_ok=True)
    
    def safe_grayscale(self, img):
        """安全的灰度转换"""
        if len(img.shape) == 3:
            B = img[:, :, 0].astype(np.float32)
            G = img[:, :, 1].astype(np.float32)
            R_ch = img[:, :, 2].astype(np.float32)
            gray = (R_ch * 30 + G * 59 + B * 11 + 50) / 100
            return np.clip(gray, 0, 255).astype(np.uint8)
        return img
    
    def corrected_step_three(self, original_image_path, saliency_map_path, debug=False):
        """
        修正后的步骤三实现
        """
        # 读取图像
        I = cv2.imread(original_image_path)
        R = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
        
        if I is None or R is None:
            raise ValueError("无法读取图像文件")
        
        # 确保图像尺寸一致
        if I.shape[:2] != R.shape:
            R = cv2.resize(R, (I.shape[1], I.shape[0]))
        
        # 步骤3.1: 转换为灰度图
        I_G = self.safe_grayscale(I)
        
        # 处理显著性图R
        if R.max() <= 1.0:
            R = (R * 255).astype(np.uint8)
        
        # 创建R的二进制掩模
        if np.max(R) == np.min(R):
            R_binary = np.ones_like(R, dtype=np.uint8)
        else:
            _, R_binary = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 修正：RG不需要加掩膜，直接使用显著性图的灰度版本
        R_G = self.safe_grayscale(R)
        
        # 步骤3.2: 计算差值图像 X = RG - IG
        X = R_G.astype(np.int16) - I_G.astype(np.int16)
        
        # 检测差值方向并调整算法
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
        
        # 步骤3.5: 结合Y和R的掩模
        Y_non_zero = (Y > 0)
        R_non_zero = (R_binary > 0)
        final_mask = Y_non_zero & R_non_zero
        
        # 应用形态学操作
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # 提取最终结果
        S = cv2.bitwise_and(I, I, mask=final_mask)
        Z = cv2.bitwise_and(I_G, I_G, mask=final_mask)
        
        return S, Z, final_mask
    
    def step_four_core_ornament_extraction(self, S, Z, debug=False):
        """
        第四步：核心纹饰提取
        """
        # 步骤4.1: 形态学闭运算
        kernel = np.ones((5, 5), np.uint8)
        Zm = cv2.morphologyEx(Z, cv2.MORPH_CLOSE, kernel)
        
        # 步骤4.2: 连通域分析
        _, binary = cv2.threshold(Zm, 1, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:
                continue
            components.append({
                'label': i,
                'area': area,
                'centroid': centroids[i],
                'mask': (labels == i).astype(np.uint8)
            })
        
        if len(components) == 0:
            return np.zeros_like(S), None, components
        
        # 步骤4.3: 高斯加权定位
        height, width = Z.shape
        mu, sigma_sq = 0, 30000
        
        max_weight, core_component = -1, None
        
        for comp in components:
            total_weight = 0
            mask, indices = comp['mask'], np.where(comp['mask'] == 1)
            
            for y, x in zip(indices[0], indices[1]):
                distance = np.sqrt((x - width/2)**2 + (y - height/2)**2)
                weight = (1 / np.sqrt(2 * np.pi * sigma_sq)) * np.exp(-(distance - mu)**2 / (2 * sigma_sq))
                total_weight += weight
            
            comp['weight'] = total_weight
            if total_weight > max_weight:
                max_weight, core_component = total_weight, comp
        
        # 创建核心纹饰图像
        core_mask = core_component['mask'] * 255
        core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        core_ornament = cv2.bitwise_and(S, S, mask=core_mask)
        
        return core_ornament, core_component, components
    
    def process_single_image(self, original_image_path, saliency_map_path, filename):
        """
        处理单张图像
        """
        try:
            print(f"处理图像: {filename}")
            
            # 步骤三：提取全部纹饰
            S, Z, mask = self.corrected_step_three(original_image_path, saliency_map_path)
            
            # 检查步骤三结果
            if np.max(S) == 0:
                print(f"警告: {filename} 步骤三未提取到纹饰，使用显著性图直接提取")
                R = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
                I = cv2.imread(original_image_path)
                if R.max() <= 1.0:
                    R = (R * 255).astype(np.uint8)
                _, binary_mask = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                S = cv2.bitwise_and(I, I, mask=binary_mask)
                Z = cv2.cvtColor(S, cv2.COLOR_BGR2GRAY)
            
            # 步骤四：核心纹饰提取
            core_ornament, core_component, all_components = self.step_four_core_ornament_extraction(S, Z)
            
            # 保存结果
            result_info = self._save_results(S, core_ornament, core_component, 
                                           original_image_path, filename)
            
            self.success_count += 1
            self.results.append({
                'filename': filename,
                'status': 'success',
                'result_info': result_info
            })
            
            print(f"✅ {filename} 处理完成")
            return True
            
        except Exception as e:
            print(f"❌ {filename} 处理失败: {e}")
            self.failed_count += 1
            self.results.append({
                'filename': filename,
                'status': 'failed',
                'error': str(e)
            })
            return False
    
    def _save_results(self, S, core_ornament, core_component, original_image_path, filename):
        """
        保存单张图像的处理结果
        """
        # 读取原始图像获取尺寸信息
        I = cv2.imread(original_image_path)
        base_name = os.path.splitext(filename)[0]
        
        result_info = {
            'filename': filename,
            'base_name': base_name,
            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'file_paths': {}
        }
        
        # 保存全部纹饰
        all_color_path = os.path.join(self.output_dir, "all_ornaments", f"{base_name}_all_color.jpg")
        all_gray_path = os.path.join(self.output_dir, "all_ornaments", f"{base_name}_all_gray.jpg")
        cv2.imwrite(all_color_path, S)
        cv2.imwrite(all_gray_path, cv2.cvtColor(S, cv2.COLOR_BGR2GRAY))
        result_info['file_paths']['all_ornaments'] = {
            'color': all_color_path,
            'gray': all_gray_path
        }
        
        # 保存核心纹饰
        core_color_path = os.path.join(self.output_dir, "core_ornaments", f"{base_name}_core_color.jpg")
        core_gray_path = os.path.join(self.output_dir, "core_ornaments", f"{base_name}_core_gray.jpg")
        cv2.imwrite(core_color_path, core_ornament)
        cv2.imwrite(core_gray_path, cv2.cvtColor(core_ornament, cv2.COLOR_BGR2GRAY))
        result_info['file_paths']['core_ornaments'] = {
            'color': core_color_path,
            'gray': core_gray_path
        }
        
        # 保存掩模
        if core_component:
            core_mask = core_component['mask'] * 255
            mask_path = os.path.join(self.output_dir, "masks", f"{base_name}_mask.jpg")
            cv2.imwrite(mask_path, core_mask)
            result_info['file_paths']['mask'] = mask_path
        
        # 生成并保存报告
        report = self._generate_report(core_component, I.shape, base_name)
        report_path = os.path.join(self.output_dir, "reports", f"{base_name}_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        result_info['file_paths']['report'] = report_path
        
        # 生成可视化结果
        viz_path = os.path.join(self.output_dir, "visualizations", f"{base_name}_viz.jpg")
        self._create_visualization(I, S, core_ornament, core_component, viz_path)
        result_info['file_paths']['visualization'] = viz_path
        
        return result_info
    
    def _generate_report(self, core_component, image_shape, base_name):
        """生成提取报告"""
        report = {
            "filename": base_name,
            "extraction_time": datetime.datetime.now().isoformat(),
            "original_image_dimensions": {
                "width": int(image_shape[1]),
                "height": int(image_shape[0])
            }
        }
        
        if core_component:
            report.update({
                "status": "success",
                "core_ornament_area": int(core_component['area']),
                "area_percentage": float(core_component['area'] / (image_shape[0] * image_shape[1]) * 100),
                "centroid_position": {
                    "x": float(core_component['centroid'][0]),
                    "y": float(core_component['centroid'][1])
                },
                "weight_value": float(core_component['weight'])
            })
        else:
            report.update({
                "status": "no_core_ornament_detected"
            })
        
        return report
    
    def _create_visualization(self, I, S, core_ornament, core_component, save_path):
        """创建可视化结果"""
        # 调整尺寸以便显示
        scale_factor = min(800 / I.shape[1], 600 / I.shape[0])
        new_width = int(I.shape[1] * scale_factor)
        new_height = int(I.shape[0] * scale_factor)
        
        I_small = cv2.resize(I, (new_width, new_height))
        S_small = cv2.resize(S, (new_width, new_height))
        core_small = cv2.resize(core_ornament, (new_width, new_height))
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        images = [
            (cv2.cvtColor(I_small, cv2.COLOR_BGR2RGB), "原始图像"),
            (cv2.cvtColor(S_small, cv2.COLOR_BGR2RGB), "全部纹饰"),
            (cv2.cvtColor(core_small, cv2.COLOR_BGR2RGB), "核心纹饰"),
        ]
        
        for i, (img, title) in enumerate(images):
            ax = axes[i//2, i%2]
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        
        # 添加统计信息
        if core_component:
            info_text = f"核心纹饰信息:\n面积: {core_component['area']}像素\n"
            info_text += f"占比: {core_component['area']/(I.shape[0]*I.shape[1])*100:.2f}%\n"
            info_text += f"权值: {core_component['weight']:.4f}"
            axes[1, 1].text(0.5, 0.5, info_text, transform=axes[1, 1].transAxes, 
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            axes[1, 1].axis('off')
        
        plt.suptitle('陶瓷纹饰提取结果', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def find_matching_pairs(self):
        """
        查找原始图像和显著性图的匹配对
        """
        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        saliency_extensions = ['*.png', '*.jpg', '*.jpeg']
        
        # 获取所有原始图像文件
        original_files = []
        for ext in image_extensions:
            original_files.extend(glob.glob(os.path.join(self.original_images_dir, ext)))
        
        # 获取所有显著性图文件
        saliency_files = []
        for ext in saliency_extensions:
            saliency_files.extend(glob.glob(os.path.join(self.saliency_maps_dir, ext)))
        
        # 创建文件名到路径的映射
        original_map = {os.path.splitext(os.path.basename(f))[0]: f for f in original_files}
        saliency_map = {os.path.splitext(os.path.basename(f))[0]: f for f in saliency_files}
        
        # 查找匹配的文件对
        matching_pairs = []
        common_names = set(original_map.keys()) & set(saliency_map.keys())
        
        for name in common_names:
            matching_pairs.append((original_map[name], saliency_map[name], name))
        
        print(f"找到 {len(matching_pairs)} 对匹配的图像文件")
        return matching_pairs
    
    def run_batch_processing(self):
        """
        运行批量处理
        """
        print("=" * 70)
        print("开始批量陶瓷纹饰提取")
        print("=" * 70)
        print(f"原始图像目录: {self.original_images_dir}")
        print(f"显著性图目录: {self.saliency_maps_dir}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 70)
        
        start_time = datetime.datetime.now()
        
        # 查找匹配的文件对
        matching_pairs = self.find_matching_pairs()
        
        if not matching_pairs:
            print("❌ 未找到匹配的图像文件对")
            return False
        
        print(f"开始处理 {len(matching_pairs)} 对图像...")
        print("-" * 70)
        
        # 处理每个匹配对
        for i, (original_path, saliency_path, filename) in enumerate(matching_pairs, 1):
            print(f"[{i}/{len(matching_pairs)}] ", end="")
            self.process_single_image(original_path, saliency_path, filename)
            self.processed_count += 1
        
        # 生成批量处理报告
        self._generate_batch_report(start_time)
        
        return True
    
    def _generate_batch_report(self, start_time):
        """生成批量处理报告"""
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        batch_report = {
            "batch_processing_report": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "processing_time_seconds": processing_time,
                "total_images_processed": self.processed_count,
                "successful_extractions": self.success_count,
                "failed_extractions": self.failed_count,
                "success_rate": self.success_count / self.processed_count * 100 if self.processed_count > 0 else 0
            },
            "individual_results": self.results
        }
        
        report_path = os.path.join(self.output_dir, "batch_processing_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        print("=" * 70)
        print("批量处理完成!")
        print("=" * 70)
        print(f"总处理图像数: {self.processed_count}")
        print(f"成功提取数: {self.success_count}")
        print(f"失败数: {self.failed_count}")
        print(f"成功率: {batch_report['batch_processing_report']['success_rate']:.2f}%")
        print(f"处理时间: {processing_time:.2f}秒")
        print(f"平均每张图像: {processing_time/self.processed_count:.2f}秒" if self.processed_count > 0 else "N/A")
        print(f"结果保存在: {self.output_dir}")
        print("=" * 70)

def main():
    """
    主函数：批量处理示例
    """
    # 配置路径
    original_images_dir = "./test_data/test_images"  # 原始图像目录
    saliency_maps_dir = "./test_data/u2net_results"  # 显著性图目录
    output_dir = "./final_output"  # 输出目录
    
    # 创建批量提取器
    extractor = BatchOrnamentExtractor(original_images_dir, saliency_maps_dir, output_dir)
    
    # 运行批量处理
    success = extractor.run_batch_processing()
    
    if success:
        print("🎉 批量处理成功完成!")
    else:
        print("💥 批量处理失败")

if __name__ == "__main__":
    main()