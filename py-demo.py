import cv2
import numpy as np
from scipy.optimize import curve_fit
import math
import os

class PingPongAngleAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        # 球桌真实尺寸（毫米）
        self.table_real_xy = np.float32([[0, 0], [2740, 0], [2740, 1525], [0, 1525]])
        self.H = None          # 单应性矩阵
        self.scale_z = 1.0     # Z轴缩放因子（mm/像素）
        self.use_3d = False    # 是否启用3D模式

    def calculate_apex_angle(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, base_frame = cap.read()
        if not ret:
            print("❌ 无法读取视频")
            return None

        # === 尝试标定 ===
        corners = self.detect_table_corners_auto(base_frame)
        if corners and len(corners) >= 4:
            self.table_corners_px = corners[:4]
            self.H, _ = cv2.findHomography(np.float32(corners[:4]), self.table_real_xy)
            table_pixel_height = abs(corners[2][1] - corners[0][1])
            self.scale_z = max(0.5, min(5.0, 2000.0 / table_pixel_height)) if table_pixel_height > 20 else 1.5
            self.use_3d = True
            print("🟢 启用3D物理坐标模式")
        else:
            self.use_3d = False
            print("🟡 未检测到完整桌角，启用2D像素坐标模式（结果仅供参考）")

        # === 跟踪轨迹 ===
        ball_traj_px = []
        prev_center = None
        while True:
            ret, frame = cap.read()
            if not ret: break
            center = self.detect_ball_robust(frame, prev_center)
            if center:
                ball_traj_px.append(center)
                prev_center = center
        cap.release()

        if len(ball_traj_px) < 3:
            print("❌ 轨迹点过少（<3），无法可视化")
            return None
        print(f"🎯 共检测到 {len(ball_traj_px)} 个球位置")

        # === 提取关键点（允许失败）===
        apex_idx, end_idx = self._find_key_indices(ball_traj_px)
        has_valid_triangle = (apex_idx is not None and end_idx is not None)

        if not has_valid_triangle:
            print("⚠️ 未找到有效 End 点（未落回起始高度），仅绘制轨迹")
            apex_idx = self._find_apex_only(ball_traj_px)
            if apex_idx is None:
                apex_idx = len(ball_traj_px) // 2  # 最后 fallback
            end_idx = None

        # === 可视化 ===
        img = base_frame.copy()
        # 绘制整个轨迹
        for pt in ball_traj_px:
            cv2.circle(img, pt, 2, (0, 255, 255), -1)

        # 起点（始终存在）
        start_pt = ball_traj_px[0]
        cv2.circle(img, start_pt, 6, (255, 0, 0), -1)
        cv2.putText(img, 'Start', (start_pt[0]+10, start_pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Apex（始终尝试标出）
        apex_pt = ball_traj_px[apex_idx]
        cv2.circle(img, apex_pt, 6, (0, 255, 0), -1)
        cv2.putText(img, 'Apex', (apex_pt[0]+10, apex_pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        angle = None
        mode_str = "3D" if self.use_3d else "2D"
        if has_valid_triangle:
            end_pt = ball_traj_px[end_idx]
            cv2.circle(img, end_pt, 6, (255, 0, 255), -1)
            cv2.putText(img, 'End', (end_pt[0]+10, end_pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.line(img, start_pt, apex_pt, (0, 0, 255), 2)
            cv2.line(img, apex_pt, end_pt, (0, 0, 255), 2)
            try:
                P0 = np.array(ball_traj_px[0])
                Pm = np.array(ball_traj_px[apex_idx])
                Pc = np.array(ball_traj_px[end_idx])

                if self.use_3d:
                    traj_px_array = np.array(ball_traj_px, dtype=np.float32).reshape(-1, 1, 2)
                    xy_mm = cv2.perspectiveTransform(traj_px_array, self.H).reshape(-1, 2)
                    table_y_ref = np.mean([pt[1] for pt in self.table_corners_px])
                    z_mm = (table_y_ref - np.array([p[1] for p in ball_traj_px])) * self.scale_z
                    traj_3d_original = np.column_stack((xy_mm, z_mm))
                    v1 = traj_3d_original[0] - traj_3d_original[apex_idx]
                    v2 = traj_3d_original[end_idx] - traj_3d_original[apex_idx]
                else:
                    v1 = P0 - Pm
                    v2 = Pc - Pm

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 1e-6 and norm2 > 1e-6:
                    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle = math.degrees(math.acos(cos_theta))
            except Exception as e:
                print(f"⚠️ 角度计算出错: {e}")
                angle = None

        text = f"Angle: {angle:.1f}° ({mode_str})" if angle is not None else "Angle: N/A (incomplete trajectory)"
        color = (0, 165, 255) if mode_str == "2D" else (0, 255, 0)
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        output_path = "output_analysis.jpg"
        success = cv2.imwrite(output_path, img)
        print(f"📸 图像已保存: {'成功' if success else '失败'}")

        if angle is not None:
            confidence = "高精度" if self.use_3d else "低置信度"
            print(f"🎯 顶角: {angle:.1f}° ({confidence})")
        else:
            print("🎯 未形成有效发球轨迹（缺少落点）")

        return angle

    def _find_key_indices(self, traj_px):
        n = len(traj_px)
        if n < 5:
            return None, None

        y_vals = np.array([p[1] for p in traj_px])

        start_idx = max(2, n // 5)
        end_idx_search = min(n - 3, int(n * 0.8))
        if start_idx >= end_idx_search:
            return None, None

        segment_y = y_vals[start_idx:end_idx_search]
        apex_in_seg = int(np.argmin(segment_y))
        apex_idx = start_idx + apex_in_seg

        start_y = y_vals[0]
        end_idx = n - 1
        for i in range(apex_idx + 1, n):
            if abs(y_vals[i] - start_y) < 25:
                end_idx = i
                break

        if apex_idx <= 0 or apex_idx >= n - 1:
            return None, None

        return apex_idx, end_idx

    def _find_apex_only(self, traj_px):
        """仅找最高点（Y最小），用于 fallback"""
        y_vals = [p[1] for p in traj_px]
        apex_idx = int(np.argmin(y_vals))
        return apex_idx if 0 < apex_idx < len(traj_px) - 1 else None

    # 剩余函数保持不变：detect_table_corners_auto, detect_ball_robust, fit_parabola_and_get_3d_traj

    def detect_table_corners_auto(self, frame):
        """自动检测球桌四角（鲁棒版）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=60, maxLineGap=20)
        if lines is None or len(lines) < 4:
            return None

        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 50: continue
            if angle < 15 or angle > 165:
                h_lines.append((x1, y1, x2, y2))
            elif 75 < angle < 105:
                v_lines.append((x1, y1, x2, y2))

        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        # 找上下左右边界
        h_mid_y = [(min(l[1], l[3]) + max(l[1], l[3])) / 2 for l in h_lines]
        top_line = h_lines[np.argmin(h_mid_y)]
        bottom_line = h_lines[np.argmax(h_mid_y)]

        v_mid_x = [(min(l[0], l[2]) + max(l[0], l[2])) / 2 for l in v_lines]
        left_line = v_lines[np.argmin(v_mid_x)]
        right_line = v_lines[np.argmax(v_mid_x)]

        def intersect(l1, l2):
            x1, y1, x2, y2 = l1
            x3, y3, x4, y4 = l2
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-5: return None
            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
            return (int(px), int(py))

        corners = [
            intersect(top_line, left_line),
            intersect(top_line, right_line),
            intersect(bottom_line, right_line),
            intersect(bottom_line, left_line)
        ]
        valid = [pt for pt in corners if pt and 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]]
        return valid if len(valid) >= 4 else None

    def detect_ball_robust(self, frame, prev_center=None):
        """鲁棒球体检测（橙色+圆形）"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 覆盖橙色和浅黄色（抗光照变化）
        mask1 = cv2.inRange(hsv, np.array([0, 70, 120]), np.array([25, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 70, 120]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 1000: continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
            if circularity < 0.5: continue
            if prev_center is not None:
                if np.linalg.norm(np.array(center) - np.array(prev_center)) > 150: continue
            candidates.append((center, area))

        return max(candidates, key=lambda x: x[1])[0] if candidates else None

    def fit_parabola_and_get_3d_traj(self, traj_px):
        """拟合抛物线并生成3D轨迹（仅在use_3d=True时调用）"""
        if len(traj_px) < 5: return None
        x = np.array([p[0] for p in traj_px])
        y = np.array([p[1] for p in traj_px])

        def parabola(x, a, b, c): return a*x**2 + b*x + c
        try:
            popt, _ = curve_fit(parabola, x, y, maxfev=5000)
        except Exception:
            return None

        x_dense = np.linspace(x.min(), x.max(), 100)
        y_dense = parabola(x_dense, *popt)
        traj_2d_dense = np.column_stack((x_dense, y_dense))

        src_pts = traj_2d_dense.reshape(-1, 1, 2).astype(np.float32)
        xy_mm = cv2.perspectiveTransform(src_pts, self.H).reshape(-1, 2)

        table_y_ref = np.mean([pt[1] for pt in self.table_corners_px])
        z_mm = (table_y_ref - y_dense) * self.scale_z

        traj_3d = np.column_stack((xy_mm, z_mm))
        return traj_3d, traj_2d_dense

    def _find_key_indices(self, traj_px):
        n = len(traj_px)
        if n < 5:
            return None, None

        y_vals = np.array([p[1] for p in traj_px])

        # 如果轨迹很短（<15帧），允许 apex 在更宽范围内
        if n < 15:
            # 直接找全局最小 Y（最顶部）
            apex_idx = int(np.argmin(y_vals))
            # end 点设为最后一帧（即使没回到起始高度）
            end_idx = n - 1
            # 但至少保证 apex 不是第一个或最后一个
            if apex_idx == 0 or apex_idx == n - 1:
                print("⚠️ 轨迹太短且顶点在端点，无法构成三角形")
                return None, None
            return apex_idx, end_idx

        # 原有逻辑：适用于较长轨迹
        start_idx = max(2, n // 5)
        end_idx_search = min(n - 3, int(n * 0.8))
        if start_idx >= end_idx_search:
            return None, None

        segment_y = y_vals[start_idx:end_idx_search]
        apex_in_seg = int(np.argmin(segment_y))
        apex_idx = start_idx + apex_in_seg

        # 找回落点
        start_y = y_vals[0]
        end_idx = n - 1
        for i in range(apex_idx + 1, n):
            if abs(y_vals[i] - start_y) < 25:
                end_idx = i
                break

        # 放宽检查：只要 apex 不在最前/最后 1 帧即可
        if apex_idx <= 0 or apex_idx >= n - 1:
            return None, None

        return apex_idx, end_idx

    def visualize_result(self, frame, traj_px, apex_idx, end_idx, angle, mode="2D"):
        img = frame.copy()
        # 绘制轨迹
        for i in range(end_idx + 1):
            if i < len(traj_px):
                cv2.circle(img, traj_px[i], 3, (0, 255, 255), -1)
        # 标注关键点
        colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255)]
        labels = ['Start', 'Apex', 'End']
        pts = [traj_px[0], traj_px[apex_idx], traj_px[end_idx]]
        for pt, col, label in zip(pts, colors, labels):
            cv2.circle(img, pt, 6, col, -1)
            cv2.putText(img, label, (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        # 连接三角形
        cv2.line(img, pts[0], pts[1], (0, 0, 255), 2)
        cv2.line(img, pts[1], pts[2], (0, 0, 255), 2)
        # 标注角度和模式
        text = f"Angle: {angle:.1f}° ({mode})"
        color = (0, 165, 255) if mode == "2D" else (0, 255, 0)
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        return img

    def visualize_result_flexible(self, frame, traj_px, apex_idx, end_idx, angle, mode="2D"):
        img = frame.copy()
        n = len(traj_px)

        # 绘制整个轨迹（黄色小点）
        for i, pt in enumerate(traj_px):
            cv2.circle(img, pt, 2, (0, 255, 255), -1)

        # 标注起点
        start_pt = traj_px[0]
        cv2.circle(img, start_pt, 6, (255, 0, 0), -1)
        cv2.putText(img, 'Start', (start_pt[0]+10, start_pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 标注 apex（绿色）
        apex_pt = traj_px[apex_idx]
        cv2.circle(img, apex_pt, 6, (0, 255, 0), -1)
        cv2.putText(img, 'Apex', (apex_pt[0]+10, apex_pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 标注 end（品红）
        end_pt = traj_px[end_idx]
        cv2.circle(img, end_pt, 6, (255, 0, 255), -1)
        cv2.putText(img, 'End', (end_pt[0]+10, end_pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 连接三角形（如果三点不重合）
        if not np.array_equal(start_pt, apex_pt) and not np.array_equal(apex_pt, end_pt):
            cv2.line(img, start_pt, apex_pt, (0, 0, 255), 2)
            cv2.line(img, apex_pt, end_pt, (0, 0, 255), 2)

        # 显示角度或提示
        if angle is not None:
            text = f"Angle: {angle:.1f}° ({mode})"
        else:
            text = f"Angle: N/A ({mode})"

        color = (0, 165, 255) if mode == "2D" else (0, 255, 0)
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        return img

    # def calculate_apex_angle(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, base_frame = cap.read()
        if not ret:
            print("❌ 无法读取视频")
            return None

        # === 尝试标定 ===
        corners = self.detect_table_corners_auto(base_frame)
        if corners and len(corners) >= 4:
            self.table_corners_px = corners[:4]
            self.H, _ = cv2.findHomography(np.float32(corners[:4]), self.table_real_xy)
            table_pixel_height = abs(corners[2][1] - corners[0][1])
            self.scale_z = max(0.5, min(5.0, 2000.0 / table_pixel_height)) if table_pixel_height > 20 else 1.5
            self.use_3d = True
            print("🟢 启用3D物理坐标模式")
        else:
            self.use_3d = False
            print("🟡 未检测到完整桌角，启用2D像素坐标模式（结果仅供参考）")

        # === 跟踪轨迹 ===
        ball_traj_px = []
        prev_center = None
        while True:
            ret, frame = cap.read()
            if not ret: break
            center = self.detect_ball_robust(frame, prev_center)
            if center:
                ball_traj_px.append(center)
                prev_center = center
        cap.release()

        if len(ball_traj_px) < 3:
            print("❌ 轨迹点过少（<3），无法可视化")
            return None
        print(f"🎯 共检测到 {len(ball_traj_px)} 个球位置")

        # === 提取关键点（允许失败）===
        apex_idx, end_idx = self._find_key_indices(ball_traj_px)
        print(f"🔍 关键点索引: apex={apex_idx}, end={end_idx}")

        # --- Fallback: 如果 apex 无效，用全局最高点 ---
        if apex_idx is None:
            y_vals = [p[1] for p in ball_traj_px]
            apex_idx = int(np.argmin(y_vals))  # 最高点（Y最小）
            print("⚠️ 使用全局最高点作为 apex")

        # --- Fallback: 如果 end 无效，用最后一帧 ---
        if end_idx is None or end_idx <= apex_idx:
            end_idx = len(ball_traj_px) - 1
            print("⚠️ 使用轨迹终点作为 end 点")

        # 确保索引合法
        apex_idx = max(0, min(apex_idx, len(ball_traj_px) - 1))
        end_idx = max(0, min(end_idx, len(ball_traj_px) - 1))

        # === 尝试计算角度（失败则设为 None）===
        angle = None
        mode_str = "3D" if self.use_3d else "2D"
        try:
            P0 = np.array(ball_traj_px[0])
            Pm = np.array(ball_traj_px[apex_idx])
            Pc = np.array(ball_traj_px[end_idx])

            if self.use_3d:
                traj_px_array = np.array(ball_traj_px, dtype=np.float32).reshape(-1, 1, 2)
                try:
                    xy_mm = cv2.perspectiveTransform(traj_px_array, self.H).reshape(-1, 2)
                    table_y_ref = np.mean([pt[1] for pt in self.table_corners_px])
                    z_mm = (table_y_ref - np.array([p[1] for p in ball_traj_px])) * self.scale_z
                    traj_3d_original = np.column_stack((xy_mm, z_mm))
                    v1 = traj_3d_original[0] - traj_3d_original[apex_idx]
                    v2 = traj_3d_original[end_idx] - traj_3d_original[apex_idx]
                    print("✅ 使用原始轨迹的3D坐标计算角度")
                except Exception as e:
                    print(f"⚠️ 3D变换失败，回退到2D: {e}")
                    v1 = P0 - Pm
                    v2 = Pc - Pm
                    mode_str = "2D"
            else:
                v1 = P0 - Pm
                v2 = Pc - Pm

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = math.degrees(math.acos(cos_theta))
            else:
                print("⚠️ 向量长度为零，无法计算角度")
        except Exception as e:
            print(f"⚠️ 角度计算异常: {e}")

        # === 可视化（始终执行）===
        vis_img = self.visualize_result_flexible(
            base_frame, ball_traj_px, apex_idx, end_idx, angle, mode_str
        )
        output_path = "output_analysis.jpg"
        success = cv2.imwrite(output_path, vis_img)
        print(f"📸 图像保存结果: {'成功' if success else '失败'}")

        if angle is not None:
            confidence = "高精度" if self.use_3d else "低置信度（仅几何）"
            print(f"🎯 顶角: {angle:.1f}° ({confidence})")
        else:
            print("🎯 顶角: N/A（轨迹不完整）")

        return angle


# =================== 使用示例 ===================
if __name__ == "__main__":
    VIDEO_PATH = "2025-12-31 23-12-47.mkv"  # 替换为你的视频路径
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 视频文件不存在: {VIDEO_PATH}")
    else:
        analyzer = PingPongAngleAnalyzer(VIDEO_PATH)
        angle = analyzer.calculate_apex_angle()
