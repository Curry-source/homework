import cv2
import numpy as np
import random
from PIL import Image
from matplotlib import pyplot as plt

import cv2
import numpy as np
from PIL import Image
import random  # 明确导入random
import matplotlib.pyplot as plt


class ImageAugmentor:
    @staticmethod
    def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
        """CLAHE 对比度受限自适应直方图均衡化"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 2 or image.shape[2] == 1:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            enhanced = clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            l = clahe.apply(l)
            enhanced = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced)

    @staticmethod
    def guided_filter(image, radius=5, eps=0.01):
        """引导滤波（保边平滑）"""
        try:
            if not hasattr(cv2, 'ximgproc'):
                raise ImportError("OpenCV-contrib module not installed")

            if isinstance(image, Image.Image):
                image = np.array(image)

            if len(image.shape) == 2 or image.shape[2] == 1:
                guided = cv2.ximgproc.guidedFilter(
                    guide=image,
                    src=image,
                    radius=radius,
                    eps=eps
                )
            else:
                guided = np.zeros_like(image)
                for i in range(3):
                    guided[:, :, i] = cv2.ximgproc.guidedFilter(
                        guide=image[:, :, i],
                        src=image[:, :, i],
                        radius=radius,
                        eps=eps
                    )
            return Image.fromarray(guided)

        except ImportError:
            print("Warning: cv2.ximgproc not available, using GaussianBlur instead")
            blurred = cv2.GaussianBlur(np.array(image), (5, 5), 0)
            return Image.fromarray(blurred)

    @staticmethod
    def sharpen(image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
        """锐化增强"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)

        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)

        return Image.fromarray(sharpened)

    @staticmethod
    def mosaic(images, grid_size=(2, 2)):
        """Mosaic 随机拼接"""
        if len(images) < grid_size[0] * grid_size[1]:
            raise ValueError("Not enough images for mosaic")

        selected = random.sample(images, grid_size[0] * grid_size[1])
        tile_h = selected[0].size[1] // grid_size[0]
        tile_w = selected[0].size[0] // grid_size[1]

        mosaic_img = Image.new('RGB', (selected[0].size[0], selected[0].size[1]))

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                idx = i * grid_size[1] + j
                img = selected[idx].resize((tile_w, tile_h))
                mosaic_img.paste(img, (j * tile_w, i * tile_h))

        return mosaic_img

    @staticmethod
    def apply_augmentations(image):
        """应用增强组合"""

        image = ImageAugmentor.apply_clahe(image)
        image = ImageAugmentor.guided_filter(image)
        image = ImageAugmentor.sharpen(image)

        return image
