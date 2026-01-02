import cv2
import numpy as np
import math

def generate_top_view_test_video(output_path="test_serve.mp4", fps=30, duration=1.6):
    """
    生成一个俯视+轻微斜角的乒乓球发球视频：
    - 近大远小（弱透视）
    - 消失点在无穷远（使用仿射变换）
    - 桌面为梯形，可见两个边
    - 球有明显抛物线轨迹
    """
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # === 1. 定义真实桌面（mm） ===
    table_real = np.array([
        [0, 0],           # TL
        [2740, 0],        # TR
        [2740, 1525],     # BR
        [0, 1525]         # BL
    ], dtype=np.float32)

    # === 2. 映射到图像平面（使用仿射：保持平行线平行）===
    # 俯视 + 轻微向右倾斜，近处（底部）宽，远处（顶部）略窄
    img_h = 300  # 桌面在图像中的高度（像素）
    base_y = 500  # 桌面底部 y 坐标

    bottom_width_px = 900   # 近处（底部）宽度
    top_width_px = 820      # 远处（顶部）宽度（略窄，体现深度）

    center_x = width // 2

    table_px = np.array([
        [center_x - top_width_px//2, base_y - img_h],    # TL (远左)
        [center_x + top_width_px//2, base_y - img_h],    # TR (远右)
        [center_x + bottom_width_px//2, base_y],         # BR (近右)
        [center_x - bottom_width_px//2, base_y],         # BL (近左)
    ], dtype=np.float32)

    # 计算仿射变换（取前3个点）
    M_affine = cv2.getAffineTransform(table_real[:3], table_px[:3])

    # === 3. 生成真实世界中的球轨迹（抛物线）===
    t_max = 0.8  # 秒
    total_frames_traj = int(fps * t_max)
    t_vals = np.linspace(0, t_max, total_frames_traj)

    v0x, v0z = 3.0, 3.0  # m/s
    g = 9.8  # m/s²

    x_real_m = v0x * t_vals               # 向前飞行
    z_real_m = v0z * t_vals - 0.5 * g * t_vals**2  # 上升后下落
    z_real_m[z_real_m < 0] = 0

    # 转为 mm
    x_real = x_real_m * 1000              # mm
    y_real = np.full_like(x_real, 762)    # 沿中线（1525/2）
    z_real = z_real_m * 1000              # mm

    # 合并为 Nx2（xy 平面）
    traj_real_xy = np.stack([x_real, y_real], axis=1).astype(np.float32)

    # 仿射变换到图像坐标
    traj_px = cv2.transform(traj_real_xy.reshape(-1, 1, 2), M_affine).reshape(-1, 2)

    # === 4. 渲染视频 ===
    total_frames = int(fps * duration)
    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), 80, dtype=np.uint8)  # 深灰背景

        # 画桌面（四边形）
        pts = table_px.astype(np.int32)
        cv2.fillPoly(frame, [pts], (210, 210, 210))  # 浅灰桌面
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

        # 画中线
        mid_back = tuple(pts[0] + (pts[1] - pts[0]) // 2)
        mid_front = tuple(pts[3] + (pts[2] - pts[3]) // 2)
        cv2.line(frame, mid_back, mid_front, (0, 0, 0), 2)

        # 画球
        if frame_idx < len(traj_px):
            px_x, px_y = traj_px[frame_idx]
            # 根据高度 z 向上偏移（球越高，y 越小）
            visual_y = int(px_y - z_real[frame_idx] / 25.0)  # 25 mm ≈ 1 pixel
            if 0 <= px_x < width and 0 <= visual_y < height:
                cv2.circle(frame, (int(px_x), visual_y), 12, (0, 165, 255), -1)  # 橙色球

        out.write(frame)

    out.release()
    print(f"✅ 生成成功: {output_path}")
    print("视角说明：俯视+轻微斜角，桌面为梯形，球有明显弧线。")


# ====== 运行生成 ======
if __name__ == "__main__":
    generate_top_view_test_video("test_serve.mp4")