import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import re
from matplotlib import pyplot as plt
from Processsing.image_advance import ImageAugmentor  # 假设该模块提供图像增强功能


class WeldingPreprocessor:
    def __init__(self, image_dir, img_size=(32, 32), target_samples=493,
                 grayscale=True, augment=False, mosaic_prob=0.0,
                 save_augmented_dir="../Image_Result/augmented_images"):  # 新增：增强图片保存目录
        """
        初始化焊接图片预处理器，新增增强图片保存功能

        参数:
            save_augmented_dir: 增强后图片的保存目录
            其他参数含义不变...
        """
        self.img_size = img_size
        self.target_samples = target_samples
        self.grayscale = grayscale
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        self.cached_images = []  # 用于Mosaic增强
        self.save_augmented_dir = save_augmented_dir  # 增强图片保存目录
        os.makedirs(self.save_augmented_dir, exist_ok=True)  # 确保目录存在

        # RGB图处理流程
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def _apply_augmentations(self, image):
        """应用图像增强（保持返回PIL Image对象，便于保存）"""
        if not self.augment:
            return image
        # 假设ImageAugmentor.apply_augmentations返回PIL Image
        return ImageAugmentor.apply_augmentations(image)

    def _load_and_preprocess(self, img_path):
        """加载并预处理单个图像（复用原有逻辑）"""
        img = Image.open(img_path).convert('RGB')
        img = self._apply_augmentations(img)
        return self.transform(img)

    def preprocess_images(self, image_folder):
        """预处理电阻点焊图片，保存所有增强后的图片"""
        file_dict = {}
        id_pattern = re.compile(r'(\d+)')  # 匹配文件名中的数字编号

        # 遍历所有图片文件，构建编号-路径映射
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)
                    try:
                        match = id_pattern.search(file)
                        if not match:
                            print(f"警告: 文件名 {file} 无有效编号，已跳过")
                            continue
                        img_num = match.group(1)  # 提取编号（如"123"）

                        if img_num not in file_dict:
                            file_dict[img_num] = {'front': None, 'back': None}

                        lower_file = file.lower()
                        if 'f.' in lower_file:  # 正面图片（含"f."标识）
                            file_dict[img_num]['front'] = file_path
                        elif 'b.' in lower_file:  # 背面图片（含"b."标识）
                            file_dict[img_num]['back'] = file_path

                    except Exception as e:
                        print(f"处理文件 {file} 时出错: {str(e)}，已跳过")

        # 处理并合并图片，同时保存增强后的图片
        data_array = []
        missing_front = []
        missing_back = []
        augmentation_log = []
        first_sample_processed = False  # 标记是否已处理第一组样本

        # 按编号升序处理
        for img_num in sorted(file_dict.keys(), key=lambda x: int(x)):
            paths = file_dict[img_num]

            # 检查正反面图片是否齐全
            if not paths['front']:
                missing_front.append(img_num)
                continue
            if not paths['back']:
                missing_back.append(img_num)
                continue

            try:
                # 加载原始图像（PIL格式）
                front_original = Image.open(paths['front']).convert('RGB')
                back_original = Image.open(paths['back']).convert('RGB')

                # 应用增强（返回增强后的PIL Image）
                front_augmented = self._apply_augmentations(front_original.copy())
                back_augmented = self._apply_augmentations(back_original.copy())

                # -------------------------- 保存增强后的图片 --------------------------
                # 构造保存文件名（编号+正反面+增强标识，确保唯一）
                front_save_name = f"{img_num}_front_augmented.jpg"
                back_save_name = f"{img_num}_back_augmented.jpg"
                front_save_path = os.path.join(self.save_augmented_dir, front_save_name)
                back_save_path = os.path.join(self.save_augmented_dir, back_save_name)

                # 保存图片（支持JPG/PNG，根据原始格式可调整）
                front_augmented.save(front_save_path)
                back_augmented.save(back_save_path)
                augmentation_log.append(f"已保存增强图片: {front_save_name}, {back_save_name}")

                # 可视化第一组样本（可选，保持原有逻辑）
                if not first_sample_processed:
                    visualization_data = {
                        'front': (front_original, front_augmented),
                        'back': (back_original, back_augmented)
                    }
                    self._visualize_first_sample(visualization_data)
                    first_sample_processed = True

                # 转换为张量并合并正反面
                front_tensor = self.transform(front_augmented)
                back_tensor = self.transform(back_augmented)
                combined_tensor = torch.cat([front_tensor, back_tensor], dim=0)
                data_array.append(combined_tensor.numpy())

                # 缓存图像用于Mosaic增强
                if self.mosaic_prob > 0:
                    self.cached_images.extend([front_original, back_original])

            except Exception as e:
                print(f"合并编号 {img_num} 的图片时出错: {str(e)}，已跳过")

        # 处理Mosaic增强并保存
        if (self.mosaic_prob > 0 and len(self.cached_images) >= 4 and
                random.random() < self.mosaic_prob):
            try:
                mosaic_img = ImageAugmentor.mosaic(self.cached_images)  # 生成Mosaic图片
                # 保存Mosaic增强图片（用时间戳或序号确保唯一）
                mosaic_save_name = f"mosaic_{len(data_array)}_augmented.jpg"
                mosaic_save_path = os.path.join(self.save_augmented_dir, mosaic_save_name)
                mosaic_img.save(mosaic_save_path)
                augmentation_log.append(f"已保存Mosaic增强图片: {mosaic_save_name}")

                # 转换为张量并添加到数据数组
                mosaic_tensor = self.transform(mosaic_img)
                data_array.append(np.stack([mosaic_tensor.numpy()] * 2))

            except Exception as e:
                print(f"Mosaic增强失败: {str(e)}")

        # 转换为numpy数组
        data_array = np.stack(data_array) if len(data_array) > 0 else np.array([])

        # 输出处理信息
        actual_samples = len(data_array)
        print(f"预处理完成 - 实际样本数: {actual_samples}, 目标样本数: {self.target_samples}")
        print(f"增强图片保存目录: {os.path.abspath(self.save_augmented_dir)}")
        print(f"输出形状: {data_array.shape} (灰度图模式: {self.grayscale})")

        if missing_front:
            print(f"缺少正面图像的编号: {', '.join(missing_front)} (共{len(missing_front)}个)")
        if missing_back:
            print(f"缺少背面图像的编号: {', '.join(missing_back)} (共{len(missing_back)}个)")

        if augmentation_log:
            print("\n增强日志:")
            for log in augmentation_log[:5]:  # 只显示前5条，避免过长
                print(f"  - {log}")
            if len(augmentation_log) > 5:
                print(f"  - ... 共{len(augmentation_log)}条记录")

        if actual_samples != self.target_samples:
            print(f"警告: 样本数与目标不符 ({actual_samples} != {self.target_samples})")

        return data_array

    def _visualize_first_sample(self, vis_data):
        """可视化第一组样本的增强效果（保持原有逻辑）"""
        plt.figure(figsize=(12, 8))

        # 正面对比
        plt.subplot(2, 2, 1)
        plt.imshow(vis_data['front'][0])
        plt.title("Front - Original")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(vis_data['front'][1])
        plt.title("Front - Augmented")
        plt.axis('off')

        # 背面对比
        plt.subplot(2, 2, 3)
        plt.imshow(vis_data['back'][0])
        plt.title("Back - Original")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(vis_data['back'][1])
        plt.title("Back - Augmented")
        plt.axis('off')

        plt.tight_layout()
        # plt.show()