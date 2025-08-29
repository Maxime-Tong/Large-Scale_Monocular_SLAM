import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

def visualize_depth_map(depth_map, save_path, colormap='viridis', vmin=None, vmax=None):
    """
    可视化深度图并保存到指定路径
    
    参数:
    depth_map: numpy数组，深度图数据
    save_path: str，保存路径（包括文件名和扩展名）
    colormap: str，matplotlib colormap名称，可选 'viridis', 'plasma', 'inferno', 'magma', 'jet', 'gray' 等
    vmin, vmax: 颜色映射的最小值和最大值，如果为None则自动计算
    """
    try:
        # 确保深度图是numpy数组
        if not isinstance(depth_map, np.ndarray):
            depth_map = np.array(depth_map)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 设置颜色映射的范围
        if vmin is None:
            vmin = np.nanmin(depth_map)
        if vmax is None:
            vmax = np.nanmax(depth_map)
        
        # 显示深度图
        im = plt.imshow(depth_map, cmap=colormap, vmin=vmin, vmax=vmax)
        
        # 添加颜色条
        plt.colorbar(im, label='depth value')
        plt.title('depth map')
        plt.axis('off')  # 隐藏坐标轴
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
        print(f"深度图已保存到: {save_path}")
        
    except Exception as e:
        print(f"保存深度图时出错: {e}")
        raise

def visualize_depth_map_cv2(depth_map, save_path, normalize=True, colormap=cv2.COLORMAP_JET):
    """
    使用OpenCV可视化深度图（适合实时应用）
    
    参数:
    depth_map: numpy数组，深度图数据
    save_path: str，保存路径
    normalize: bool，是否归一化到0-255
    colormap: OpenCV colormap，如 cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_INFERNO 等
    """
    try:
        # 确保深度图是numpy数组
        if not isinstance(depth_map, np.ndarray):
            depth_map = np.array(depth_map)
        
        # 归一化到0-255（如果需要）
        if normalize:
            # 处理无效值
            valid_depth = depth_map[np.isfinite(depth_map)]
            if len(valid_depth) > 0:
                min_val = np.min(valid_depth)
                max_val = np.max(valid_depth)
                if max_val > min_val:
                    depth_normalized = ((depth_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
            else:
                depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            depth_normalized = depth_map.astype(np.uint8)
        
        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_normalized, colormap)
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图像
        cv2.imwrite(save_path, depth_colored)
        print(f"深度图已保存到: {save_path}")
        
    except Exception as e:
        print(f"保存深度图时出错: {e}")
        raise

# 示例用法
if __name__ == "__main__":
    # 创建一个示例深度图（模拟数据）
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)
    depth_example = np.sin(np.sqrt(x**2 + y**2))  # 创建一个波状深度图
    
    # 使用不同的可视化方法
    visualize_depth_map(depth_example, "../test_output/depth_map_viridis.png", colormap='viridis')
    visualize_depth_map(depth_example, "../test_output/depth_map_jet.png", colormap='jet')
    visualize_depth_map(depth_example, "../test_output/depth_map_gray.png", colormap='gray')
    
    # 使用OpenCV版本
    visualize_depth_map_cv2(depth_example, "../test_output/depth_map_cv2_jet.jpg", colormap=cv2.COLORMAP_JET)
    visualize_depth_map_cv2(depth_example, "../test_output/depth_map_cv2_inferno.jpg", colormap=cv2.COLORMAP_INFERNO)
    