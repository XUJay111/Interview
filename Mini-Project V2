import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, Scrollbar, Frame
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import os
import traceback

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class MultiViewStereoApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("多视角与双视角图像工具")
        self.root.geometry("750x700")

        self.image_paths = []
        self.image_cv_originals = []
        self.image_cv_grays = []
        self.keypoints_list = []
        self.descriptors_list = []
        # 假设的相机内参矩阵K。对于真实场景，这需要标定。
        # 这里我们使用一个基于图像尺寸的合理猜测。
        self.K_matrices = []

        # --- Left Frame for Image Loading and Selection ---
        left_frame = Frame(self.root, bd=2, relief=tk.SUNKEN)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        tk.Button(left_frame, text="添加图像", command=self.add_images).pack(pady=5, fill=tk.X)
        tk.Label(left_frame, text="已加载图像:").pack(pady=(10,0))
        self.loaded_images_listbox = Listbox(left_frame, selectmode=tk.EXTENDED, exportselection=False, width=30, height=10)
        self.loaded_images_listbox.pack(pady=5, fill=tk.BOTH, expand=True)
        scrollbar_loaded = Scrollbar(self.loaded_images_listbox, orient=tk.VERTICAL, command=self.loaded_images_listbox.yview)
        self.loaded_images_listbox.config(yscrollcommand=scrollbar_loaded.set)
        scrollbar_loaded.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(left_frame, text="选择图像 A (参考):").pack(pady=(10,0))
        self.select_img_a_listbox = Listbox(left_frame, exportselection=False, width=30, height=5)
        self.select_img_a_listbox.pack(pady=5, fill=tk.X)
        tk.Label(left_frame, text="选择图像 B (待处理):").pack(pady=(5,0))
        self.select_img_b_listbox = Listbox(left_frame, exportselection=False, width=30, height=5)
        self.select_img_b_listbox.pack(pady=5, fill=tk.X)
        self.btn_clear_images = tk.Button(left_frame, text="清空所有图像", command=self.clear_all_images)
        self.btn_clear_images.pack(pady=10, fill=tk.X)

        # --- Right Frame for Function Buttons and Previews ---
        right_frame = Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        func_frame = tk.LabelFrame(right_frame, text="功能选择", padx=10, pady=10)
        func_frame.pack(fill="x", pady=10)

        # 功能1: 计算并可视化双机相对位姿
        tk.Button(func_frame, text="1. 计算并可视化双机相对位姿 (A->B)", command=self.run_relative_camera_pose_visualization, width=40).pack(pady=5)
        
        # 功能2: 双视几何
        two_view_frame = tk.LabelFrame(func_frame, text="双视几何", padx=5, pady=5)
        two_view_frame.pack(fill="x", pady=(10, 5), padx=5)
        tk.Button(two_view_frame, text="2A. 显示几何概念图 (示意图)", command=self.show_epipolar_geometry_concept).pack(pady=3, fill=tk.X)
        tk.Button(two_view_frame, text="2B. 绘制图像对极线 (选定两张)", command=self.run_epipolar_lines_for_selected).pack(pady=3, fill=tk.X)
        
        # 功能3: 立体校正
        tk.Button(func_frame, text="3. 执行双目立体校正 (选定两张)", command=self.run_stereo_rectification, width=40).pack(pady=(15, 5))

        # --- Thumbnail Previews ---
        thumbnail_frame = tk.LabelFrame(right_frame, text="选定图像预览 (A 和 B)", padx=10, pady=10)
        thumbnail_frame.pack(fill="x", pady=10)
        self.canvas_img_a_thumb = tk.Canvas(thumbnail_frame, width=150, height=100, bg="lightgrey")
        self.canvas_img_a_thumb.pack(side=tk.LEFT, padx=5)
        self.canvas_img_b_thumb = tk.Canvas(thumbnail_frame, width=150, height=100, bg="lightgrey")
        self.canvas_img_b_thumb.pack(side=tk.RIGHT, padx=5)
        self.select_img_a_listbox.bind('<<ListboxSelect>>', lambda e: self.update_thumbnail_preview('A'))
        self.select_img_b_listbox.bind('<<ListboxSelect>>', lambda e: self.update_thumbnail_preview('B'))

        # --- Status Bar ---
        self.status_label = tk.Label(self.root, text="准备就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def _cv_to_tkimage_resized(self, cv_image, target_size=(150, 100)):
        # ... existing code ...
        if cv_image is None: return None
        try:
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail(target_size, Image.LANCZOS)
            return ImageTk.PhotoImage(image=img_pil)
        except Exception as e:
            print(f"Error converting CV to TK image: {e}")
            return None

    def update_thumbnail_preview(self, canvas_id):
        # ... existing code ...
        listbox = self.select_img_a_listbox if canvas_id == 'A' else self.select_img_b_listbox
        canvas = self.canvas_img_a_thumb if canvas_id == 'A' else self.canvas_img_b_thumb
        selected_indices = listbox.curselection()
        if not selected_indices:
            canvas.delete("all")
            canvas.create_text(75, 50, text="未选择", fill="grey")
            return
        idx = selected_indices[0]
        if 0 <= idx < len(self.image_cv_originals):
            img_cv = self.image_cv_originals[idx]
            img_tk = self._cv_to_tkimage_resized(img_cv)
            if img_tk:
                canvas.delete("all")
                canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                canvas.image = img_tk # Keep a reference
            else:
                canvas.delete("all"); canvas.create_text(75, 50, text="预览失败", fill="red")
        else:
            canvas.delete("all"); canvas.create_text(75, 50, text="索引无效", fill="red")

    def add_images(self):
        paths = filedialog.askopenfilenames(title="选择一张或多张图像", filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not paths: return
        new_images_loaded = 0
        for path in paths:
            if path in self.image_paths:
                self._update_status(f"图像 {os.path.basename(path)} 已加载，跳过。")
                continue
            try:
                img_cv = cv2.imread(path)
                if img_cv is None: raise ValueError(f"无法读取图像文件: {os.path.basename(path)}")
                
                h, w = img_cv.shape[:2]
                focal_length = max(w, h) # A common heuristic
                K = np.array([[focal_length, 0, w/2],
                              [0, focal_length, h/2],
                              [0, 0, 1]], dtype=np.float32)
                self.K_matrices.append(K)

                self.image_paths.append(path)
                self.image_cv_originals.append(img_cv)
                gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                self.image_cv_grays.append(gray_img)
                self._update_status(f"正在为 {os.path.basename(path)} 计算特征点...")
                kp, des = self._detect_features_for_image(gray_img)
                self.keypoints_list.append(kp)
                self.descriptors_list.append(des)
                filename = os.path.basename(path)
                self.loaded_images_listbox.insert(tk.END, filename)
                self.select_img_a_listbox.insert(tk.END, filename)
                self.select_img_b_listbox.insert(tk.END, filename)
                new_images_loaded += 1
                self._update_status(f"图像 {filename} 加载并处理完毕。")
            except Exception as e:
                messagebox.showerror("加载错误", f"加载或处理图像 {os.path.basename(path)} 失败: {e}")
                self._update_status(f"加载图像 {os.path.basename(path)} 失败。")
                traceback.print_exc()
        if new_images_loaded > 0: self._update_status(f"成功加载 {new_images_loaded} 张新图像。")
        self.update_thumbnail_preview('A'); self.update_thumbnail_preview('B')


    def clear_all_images(self):
        # ... existing code ...
        if not self.image_paths:
            messagebox.showinfo("提示", "没有图像可清除。")
            return
        if messagebox.askyesno("确认", "确定要清空所有已加载的图像和数据吗？"):
            self.image_paths.clear(); self.image_cv_originals.clear(); self.image_cv_grays.clear()
            self.keypoints_list.clear(); self.descriptors_list.clear(); self.K_matrices.clear()
            self.loaded_images_listbox.delete(0, tk.END)
            self.select_img_a_listbox.delete(0, tk.END); self.select_img_b_listbox.delete(0, tk.END)
            for canvas in [self.canvas_img_a_thumb, self.canvas_img_b_thumb]:
                canvas.delete("all"); canvas.create_text(75, 50, text="未选择", fill="grey")
            self._update_status("所有图像已清空。")

    def _detect_features_for_image(self, gray_img):
        # ... existing code ...
        kp, des = None, None
        try:
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(gray_img, None)
        except (AttributeError, cv2.error) as e_sift:
            print(f"SIFT创建失败或不可用: {e_sift}. 尝试ORB.")
            try:
                orb = cv2.ORB_create(nfeatures=2000)
                kp, des = orb.detectAndCompute(gray_img, None)
            except Exception as e_orb:
                print(f"ORB也失败: {e_orb}")
                self._update_status("特征检测器 (SIFT/ORB) 均失败。")
                return None, None
        if kp is None or len(kp) == 0:
             self._update_status("未检测到特征点。")
             return None, None
        return kp, des

    def _get_common_inliers(self, idx_a, idx_b):
        """Helper function to find Fundamental matrix and inliers between two images."""
        kp_a, des_a = self.keypoints_list[idx_a], self.descriptors_list[idx_a]
        kp_b, des_b = self.keypoints_list[idx_b], self.descriptors_list[idx_b]
        if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
            raise ValueError(f"图像 {os.path.basename(self.image_paths[idx_a])} 或 {os.path.basename(self.image_paths[idx_b])} 的描述子不足。")

        matcher_norm = cv2.NORM_HAMMING if des_a.dtype == np.uint8 else cv2.NORM_L2
        bf = cv2.BFMatcher(matcher_norm, crossCheck=False)
        matches_knn = bf.knnMatch(des_a, des_b, k=2)
        
        good_matches = []
        if matches_knn and all(len(m_pair) == 2 for m_pair in matches_knn if m_pair):
            for m, n in matches_knn:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        else: # Fallback for some descriptor types
            bf_single = cv2.BFMatcher(matcher_norm, crossCheck=True)
            good_matches = bf_single.match(des_a, des_b)

        if len(good_matches) < 8:
            raise ValueError(f"优质匹配点不足 ({len(good_matches)})，至少需要8个。")

        pts_a_matched = np.float32([kp_a[m.queryIdx].pt for m in good_matches])
        pts_b_matched = np.float32([kp_b[m.trainIdx].pt for m in good_matches])

        F, mask_fundamental = cv2.findFundamentalMat(pts_a_matched, pts_b_matched, cv2.FM_RANSAC, 3.0, 0.99)
        if F is None:
            raise ValueError("无法计算基础矩阵F。")

        inlier_mask = mask_fundamental.ravel() == 1
        pts_a_inliers = pts_a_matched[inlier_mask]
        pts_b_inliers = pts_b_matched[inlier_mask]
        
        if len(pts_a_inliers) < 8:
            raise ValueError(f"基础矩阵计算后内点数量不足 ({len(pts_a_inliers)})。")
            
        return F, pts_a_inliers, pts_b_inliers

    # --- FUNCTION 1: Relative Camera Pose Visualization ---
    def run_relative_camera_pose_visualization(self):
        idx_a, idx_b = self._get_selected_pair_indices()
        if idx_a is None: return

        self._update_status("正在计算相对位姿...")
        try:
            F, pts_a_inliers, pts_b_inliers = self._get_common_inliers(idx_a, idx_b)
            K_a = self.K_matrices[idx_a]
            K_b = self.K_matrices[idx_b]
            
            # 从F和K计算本质矩阵E
            E = K_b.T @ F @ K_a
            
            # 分解E得到R和t
            # cv2.recoverPose返回可能的旋转矩阵和平移向量
            points, R, t, mask = cv2.recoverPose(E, pts_a_inliers, pts_b_inliers, cameraMatrix=K_a)
            
            # --- 可视化 ---
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            # 相机A在原点，朝向-Z轴
            pose_A = np.eye(4)
            
            # 相机B的位姿 (从A坐标系看B)
            pose_B = np.eye(4)
            pose_B[:3, :3] = R.T # R是B到A的旋转，所以A看B的旋转是R.T
            pose_B[:3, 3] = -R.T @ t.ravel() # t是B在A坐标系下的位置，但为了得到B的中心，需要这样计算
            
            def draw_camera(ax, pose, scale, color, label):
                origin = pose[:3, 3]
                x_axis = pose[:3, 0] * scale
                y_axis = pose[:3, 1] * scale
                z_axis = pose[:3, 2] * scale
                ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='red', length=scale)
                ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='green', length=scale)
                ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='blue', length=scale)
                ax.text(origin[0], origin[1], origin[2], label, color=color)

            draw_camera(ax, pose_A, 0.5, 'black', '相机 A (参考)')
            draw_camera(ax, pose_B, 0.5, 'purple', '相机 B')

            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title('计算出的双机相对位姿 (A->B)')
            
            # 自动调整坐标轴范围
            all_points = np.vstack([pose_A[:3, 3], pose_B[:3, 3]])
            max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                                  all_points[:,1].max()-all_points[:,1].min(), 
                                  all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0
            mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
            mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
            mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_aspect('equal')

            plot_window = tk.Toplevel(self.root)
            plot_window.title("计算出的双机相对位姿")
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas.draw()
            self._update_status("相对位姿计算与可视化完成。")
            
        except Exception as e:
            messagebox.showerror("位姿计算错误", f"计算相对位姿失败: {e}")
            self._update_status(f"相对位姿计算失败: {e}")
            traceback.print_exc()

    # --- FUNCTION 2A: Epipolar Geometry Conceptual Diagram ---
    def show_epipolar_geometry_concept(self):
        # ... existing code ...
        self._update_status("正在显示双视几何概念图...")
        try:
            fig_concept = plt.figure(figsize=(8, 5))
            ax_concept = fig_concept.add_subplot(111)

            # Camera centers and image planes (simplified 2D representation)
            O1, O2 = np.array([0, 0.5]), np.array([3, 0.5]) # Camera centers
            img_plane1_y, img_plane2_y = 0, 0 # Image planes as lines
            ax_concept.plot([-1, 1], [img_plane1_y, img_plane1_y], 'k-', label='图像平面1')
            ax_concept.plot([2, 4], [img_plane2_y, img_plane2_y], 'k-', label='图像平面2')

            # 3D Point P and its projections
            P = np.array([1.5, 2]) # A point in 3D space (shown in 2D)
            ax_concept.plot(P[0], P[1], 'ko', markersize=8, label='3D点 P')
            p, p_prime = np.array([0.75, img_plane1_y]), np.array([2.25, img_plane2_y]) # Projections
            ax_concept.plot(p[0], p[1], 'bo', markersize=6, label='p (P在图像1的投影)')
            ax_concept.plot(p_prime[0], p_prime[1], 'ro', markersize=6, label="p' (P在图像2的投影)")

            # Projection lines from camera centers to P and to projections p, p'
            ax_concept.plot([O1[0], P[0]], [O1[1], P[1]], 'b--') # O1 to P
            ax_concept.plot([O1[0], p[0]], [O1[1], p[1]], 'b-')  # O1 to p
            ax_concept.plot([O2[0], P[0]], [O2[1], P[1]], 'r--') # O2 to P
            ax_concept.plot([O2[0], p_prime[0]], [O2[1], p_prime[1]], 'r-') # O2 to p'
            
            # Epipoles e1, e2 (projection of one camera center onto the other image plane)
            e1, e2 = np.array([0.25, img_plane1_y]), np.array([2.75, img_plane2_y]) # Simplified positions
            ax_concept.plot(e1[0], e1[1], 'gx', markersize=8, mew=2, label='e1 (极点 O2->I1)')
            ax_concept.plot(e2[0], e2[1], 'mx', markersize=8, mew=2, label='e2 (极点 O1->I2)')
            
            # Epipolar lines l, l' (lines connecting epipole and projection of P)
            # Extend lines slightly for better visualization
            ax_concept.plot([e1[0], p[0]*1.2 - e1[0]*0.2], [e1[1], p[1]], 'g-', label='对极线 l (在图像1)')
            ax_concept.plot([e2[0], p_prime[0]*1.2 - e2[0]*0.2], [e2[1], p_prime[1]], 'm-', label="对极线 l' (在图像2)")
            
            # Baseline (line connecting camera centers)
            ax_concept.plot([O1[0], O2[0]], [O1[1], O2[1]], 'k:', label='基线 (O1-O2)')

            # Mark camera centers
            ax_concept.plot(O1[0], O1[1], 'bs', markersize=10, label='O1 (相机中心1)')
            ax_concept.plot(O2[0], O2[1], 'rs', markersize=10, label='O2 (相机中心2)')

            ax_concept.set_xlim(-1.5, 4.5); ax_concept.set_ylim(-0.5, 3)
            ax_concept.set_aspect('equal', adjustable='box')
            ax_concept.set_title("双视几何基本概念 (示意图)")
            ax_concept.legend(fontsize='small', loc='upper right')
            ax_concept.axis('off') # Clean look

            concept_window = tk.Toplevel(self.root)
            concept_window.title("双视几何概念")
            canvas = FigureCanvasTkAgg(fig_concept, master=concept_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas.draw()
            self._update_status("双视几何概念图显示完成。")

        except Exception as e:
            messagebox.showerror("绘图错误", f"显示概念图失败: {e}")
            self._update_status(f"显示概念图失败: {e}")
            traceback.print_exc()


    def _get_selected_pair_indices(self):
        # ... existing code ...
        idx_a_tuple = self.select_img_a_listbox.curselection()
        idx_b_tuple = self.select_img_b_listbox.curselection()
        if not idx_a_tuple or not idx_b_tuple:
            messagebox.showerror("选择错误", "请为该功能选择两张图像 (图像A和图像B)。")
            return None, None
        idx_a, idx_b = idx_a_tuple[0], idx_b_tuple[0]
        if idx_a == idx_b:
            messagebox.showerror("选择错误", "请选择两张不同的图像。")
            return None, None
        if not (0 <= idx_a < len(self.image_paths) and 0 <= idx_b < len(self.image_paths)):
            messagebox.showerror("错误", "选择的图像索引无效。请重新加载图像。")
            return None, None
        return idx_a, idx_b

    def _draw_epilines(self, img_to_draw_on, lines, points_to_draw, draw_points=True):
        r, c = img_to_draw_on.shape[:2]
        img_color = cv2.cvtColor(img_to_draw_on, cv2.COLOR_GRAY2BGR) if len(img_to_draw_on.shape) == 2 else img_to_draw_on.copy()

        max_points_to_draw = 30
        num_pts = len(points_to_draw)
        if num_pts == 0: return img_color

        indices_to_draw = np.arange(num_pts)
        if num_pts > max_points_to_draw:
            indices_to_draw = np.random.choice(num_pts, max_points_to_draw, replace=False)

        for i in indices_to_draw:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            line_params = lines[i]
            pt = points_to_draw[i]

            x0, y0 = map(int, [0, -line_params[2] / line_params[1]])
            x1, y1 = map(int, [c, -(line_params[2] + line_params[0] * c) / line_params[1]])
            
            cv2.line(img_color, (x0, y0), (x1, y1), color, 1)
            if draw_points:
                 cv2.circle(img_color, tuple(map(int, pt)), 5, color, -1)
        return img_color

    # --- FUNCTION 2B: Epipolar Lines on Selected Images ---
    def run_epipolar_lines_for_selected(self):
        idx_a, idx_b = self._get_selected_pair_indices()
        if idx_a is None: return

        self._update_status("正在计算基础矩阵并绘制对极线...")
        try:
            F, pts_a_inliers, pts_b_inliers = self._get_common_inliers(idx_a, idx_b)
            self._update_status(f"找到 {len(pts_a_inliers)} 个内点。正在绘制对极线...")
            
            img_a_orig = self.image_cv_originals[idx_a]
            img_b_orig = self.image_cv_originals[idx_b]

            # 在图像A上绘制对应于图像B中点的对极线
            lines_on_a = cv2.computeCorrespondEpilines(pts_b_inliers.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
            img_a_with_lines = self._draw_epilines(img_a_orig, lines_on_a, pts_a_inliers)

            # 在图像B上绘制对应于图像A中点的对极线
            lines_on_b = cv2.computeCorrespondEpilines(pts_a_inliers.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
            img_b_with_lines = self._draw_epilines(img_b_orig, lines_on_b, pts_b_inliers)
            
            result_window = tk.Toplevel(self.root)
            result_window.title(f"对极线: {os.path.basename(self.image_paths[idx_a])} vs {os.path.basename(self.image_paths[idx_b])}")
            fig = plt.figure(figsize=(14, 7))
            plt.subplot(121); plt.imshow(cv2.cvtColor(img_a_with_lines, cv2.COLOR_BGR2RGB))
            plt.title(f'图像A上的对极线和对应点'); plt.axis('off')
            plt.subplot(122); plt.imshow(cv2.cvtColor(img_b_with_lines, cv2.COLOR_BGR2RGB))
            plt.title(f'图像B上的对极线和对应点'); plt.axis('off')
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=result_window)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas.draw()
            self._update_status("对极线显示完成。")

        except Exception as e:
            messagebox.showerror("错误", f"处理对极线时发生错误: {e}")
            self._update_status(f"对极线处理错误: {e}")
            traceback.print_exc()

    # --- FUNCTION 3: Stereo Rectification ---
    def run_stereo_rectification(self):
        idx_a, idx_b = self._get_selected_pair_indices()
        if idx_a is None: return

        self._update_status("正在执行立体校正...")
        try:
            F, pts_a_inliers, pts_b_inliers = self._get_common_inliers(idx_a, idx_b)
            self._update_status(f"找到 {len(pts_a_inliers)} 个内点。正在计算校正变换...")

            img_a_orig = self.image_cv_originals[idx_a]
            img_b_orig = self.image_cv_originals[idx_b]
            h_a, w_a = img_a_orig.shape[:2]
            h_b, w_b = img_b_orig.shape[:2]

            # 计算立体校正的变换矩阵 H1 和 H2
            # 注意：cv2.stereoRectifyUncalibrated 返回值 success, H1, H2
            _, H1, H2 = cv2.stereoRectifyUncalibrated(pts_a_inliers, pts_b_inliers, F, imgSize=(w_a, h_a))

            # 应用变换
            img_a_rectified = cv2.warpPerspective(img_a_orig, H1, (w_a, h_a))
            img_b_rectified = cv2.warpPerspective(img_b_orig, H2, (w_b, h_b))

            # --- 显示结果 ---
            result_window = tk.Toplevel(self.root)
            result_window.title(f"立体校正结果")
            fig = plt.figure(figsize=(12, 10))

            # 原始图像
            plt.subplot(221); plt.imshow(cv2.cvtColor(img_a_orig, cv2.COLOR_BGR2RGB));
            plt.title('原始图像 A'); plt.axis('off')
            plt.subplot(222); plt.imshow(cv2.cvtColor(img_b_orig, cv2.COLOR_BGR2RGB));
            plt.title('原始图像 B'); plt.axis('off')

            # 校正后图像
            ax1 = plt.subplot(223); plt.imshow(cv2.cvtColor(img_a_rectified, cv2.COLOR_BGR2RGB));
            plt.title('校正后的图像 A'); plt.axis('off')
            ax2 = plt.subplot(224); plt.imshow(cv2.cvtColor(img_b_rectified, cv2.COLOR_BGR2RGB));
            plt.title('校正后的图像 B'); plt.axis('off')
            
            # 在校正后的图像上绘制水平线以验证
            for i in range(20, h_a, 40): ax1.axhline(i, color='yellow', linestyle='--', alpha=0.5)
            for i in range(20, h_b, 40): ax2.axhline(i, color='yellow', linestyle='--', alpha=0.5)

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=result_window)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas.draw()
            self._update_status("立体校正完成。")

        except Exception as e:
            messagebox.showerror("错误", f"立体校正时发生错误: {e}")
            self._update_status(f"立体校正错误: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiViewStereoApp(root)
    root.mainloop()
