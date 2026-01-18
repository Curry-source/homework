import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from Processsing.image_advance import ImageAugmentor


class WeldingPreprocessor_IR:
    def __init__(self, img_size=(32, 32), target_samples=493,
                 grayscale=True, augment=False, mosaic_prob=0.0):
        """
        初始化焊接热成像图片预处理器（适配单组热成像图片输入）
        """
        self.img_size = img_size
        self.target_samples = target_samples  # 目标样本数（含可能的增强样本）
        self.grayscale = grayscale
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        self.cached_images = []

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def _apply_augmentations(self, image):
        if not self.augment:
            return image
        return ImageAugmentor.apply_augmentations(image)

    def _load_and_preprocess(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_augmented = self._apply_augmentations(img)
        return self.transform(img_augmented)

    def preprocess_images(self, image_folder):
        data_array = []
        augmentation_log = []
        first_sample_processed = False
        processed_files = []

        # 遍历并加载所有图像
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(root, file)
                    if img_path in processed_files:
                        continue
                    processed_files.append(img_path)

                    try:
                        img_original = Image.open(img_path).convert('RGB')
                        img_augmented = self._apply_augmentations(img_original.copy())

                        if not first_sample_processed:
                            visualization_data = {
                                'original': img_original,
                                'augmented': img_augmented
                            }
                            first_sample_processed = True

                        img_tensor = self.transform(img_augmented)
                        data_array.append(img_tensor.numpy())

                        if self.mosaic_prob > 0:
                            self.cached_images.append(img_augmented.copy())

                    except Exception as e:
                        print(f"处理文件 {file} 时出错: {str(e)}，已跳过")
                        continue

        # 可视化第一个样本
        # if first_sample_processed:
        #     self._visualize_first_sample(visualization_data)

        # Mosaic增强逻辑（添加控制条件，确保总样本数不超过目标）
        current_count = len(data_array)
        need_mosaic = (self.mosaic_prob > 0
                       and len(self.cached_images) >= 4
                       and random.random() < self.mosaic_prob
                       # 关键修复：只有当前样本数小于目标时才添加Mosaic样本
                       and current_count < self.target_samples)

        if need_mosaic:
            try:
                mosaic_img = ImageAugmentor.mosaic(self.cached_images)
                mosaic_tensor = self.transform(mosaic_img)
                data_array.append(mosaic_tensor.numpy())
                augmentation_log.append("Applied Mosaic augmentation")
                print("Mosaic增强已成功应用，新增1个增强样本")
            except Exception as e:
                print(f"Mosaic增强失败: {str(e)}，已跳过")

        # 格式转换
        if len(data_array) > 0:
            data_array = np.stack(data_array)
        else:
            data_array = np.array([])
            print("警告: 未成功处理任何图像，返回空数组")

        # 输出处理结果
        actual_samples = len(data_array)
        print(f"\n预处理完成 - 实际样本数: {actual_samples}, 目标样本数: {self.target_samples}")
        print(f"输出数据形状: {data_array.shape}")
        print(f"已处理的原始图像文件数: {len(processed_files)}")

        if augmentation_log:
            print("\n应用的图像增强:")
            for log in augmentation_log:
                print(f"  - {log}")

        if actual_samples != self.target_samples:
            print(f"警告: 实际样本数与目标样本数不符 ({actual_samples} != {self.target_samples})")

        return data_array

    def _visualize_first_sample(self, vis_data):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(vis_data['original'])
        plt.title("Thermal Image - Original", fontsize=12)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(vis_data['augmented'])
        plt.title("Thermal Image - Augmented", fontsize=12)
        plt.axis('off')

        plt.tight_layout()
        plt.show()
