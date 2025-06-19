import sys
import json
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QListWidget, QSplitter, 
                             QMessageBox, QComboBox, QGridLayout, QSlider, QGroupBox)
from PyQt5.QtCore import Qt, QDir
from PyQt5.QtGui import QImage, QPixmap

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像对极几何与视角矫正工具")
        self.setGeometry(100, 100, 1400, 900)
        
        # 初始化变量
        self.image_folder = ""
        self.transform_data = None
        self.images = []
        self.image_list = []
        self.K = None  # 相机内参矩阵
        self.alpha = 0.5  # 立体校正参数
        self.focal_scale = 1.0  # 焦距缩放因子
        
        # 创建UI
        self.init_ui()
        
    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # 左侧面板 - 图片列表和按钮
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 图片列表
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        left_layout.addWidget(QLabel("图片列表:"))
        left_layout.addWidget(self.list_widget)
        
        # 按钮
        self.load_button = QPushButton("加载图片文件夹")
        self.load_button.clicked.connect(self.load_image_folder)
        left_layout.addWidget(self.load_button)
        
        # 选项
        options_group = QGroupBox("处理选项")
        options_layout = QGridLayout(options_group)
        
        # 特征点匹配方法
        options_layout.addWidget(QLabel("特征点检测方法:"), 0, 0)
        self.feature_method = QComboBox()
        self.feature_method.addItems(["SIFT", "ORB", "AKAZE"])
        options_layout.addWidget(self.feature_method, 0, 1)
        
        # 匹配点数量
        options_layout.addWidget(QLabel("匹配点数量:"), 1, 0)
        self.match_count = QComboBox()
        self.match_count.addItems(["10", "20", "50", "100"])
        self.match_count.setCurrentIndex(1)  # 默认20个点
        options_layout.addWidget(self.match_count, 1, 1)
        
        # 立体校正参数
        options_layout.addWidget(QLabel("立体校正参数 (alpha):"), 2, 0)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 10)
        self.alpha_slider.setValue(5)  # 默认0.5
        self.alpha_slider.valueChanged.connect(self.update_alpha)
        options_layout.addWidget(self.alpha_slider, 2, 1)
        self.alpha_label = QLabel("0.5")
        options_layout.addWidget(self.alpha_label, 2, 2)
        
        # 焦距缩放
        options_layout.addWidget(QLabel("焦距缩放因子:"), 3, 0)
        self.focal_slider = QSlider(Qt.Horizontal)
        self.focal_slider.setRange(5, 20)
        self.focal_slider.setValue(10)  # 默认1.0
        self.focal_slider.valueChanged.connect(self.update_focal_scale)
        options_layout.addWidget(self.focal_slider, 3, 1)
        self.focal_label = QLabel("1.0")
        options_layout.addWidget(self.focal_label, 3, 2)
        
        # 功能按钮
        self.epipolar_button = QPushButton("显示对极线")
        self.epipolar_button.clicked.connect(self.show_epipolar_lines)
        self.epipolar_button.setEnabled(False)
        options_layout.addWidget(self.epipolar_button, 4, 0, 1, 2)
        
        self.rectify_button = QPushButton("视角矫正")
        self.rectify_button.clicked.connect(self.rectify_images)
        self.rectify_button.setEnabled(False)
        options_layout.addWidget(self.rectify_button, 5, 0, 1, 2)
        
        left_layout.addWidget(options_group)
        
        # 右侧面板 - 图片显示
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 图片显示区域
        image_grid = QGridLayout()
        
        self.image_label1 = QLabel()
        self.image_label1.setAlignment(Qt.AlignCenter)
        self.image_label1.setMinimumSize(400, 300)
        self.image_label1.setText("图片1将显示在这里")
        self.image_label1.setStyleSheet("border: 1px solid gray;")
        
        self.image_label2 = QLabel()
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setMinimumSize(400, 300)
        self.image_label2.setText("图片2将显示在这里")
        self.image_label2.setStyleSheet("border: 1px solid gray;")
        
        self.image_label1_rect = QLabel()
        self.image_label1_rect.setAlignment(Qt.AlignCenter)
        self.image_label1_rect.setMinimumSize(400, 300)
        self.image_label1_rect.setText("矫正后的图片1")
        self.image_label1_rect.setStyleSheet("border: 1px solid gray;")
        
        self.image_label2_rect = QLabel()
        self.image_label2_rect.setAlignment(Qt.AlignCenter)
        self.image_label2_rect.setMinimumSize(400, 300)
        self.image_label2_rect.setText("矫正后的图片2")
        self.image_label2_rect.setStyleSheet("border: 1px solid gray;")
        
        image_grid.addWidget(QLabel("原始图片1"), 0, 0)
        image_grid.addWidget(QLabel("原始图片2"), 0, 1)
        image_grid.addWidget(self.image_label1, 1, 0)
        image_grid.addWidget(self.image_label2, 1, 1)
        image_grid.addWidget(QLabel("矫正图片1"), 2, 0)
        image_grid.addWidget(QLabel("矫正图片2"), 2, 1)
        image_grid.addWidget(self.image_label1_rect, 3, 0)
        image_grid.addWidget(self.image_label2_rect, 3, 1)
        
        right_layout.addLayout(image_grid)
        
        # 添加左右面板到主布局
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)
        
        self.setCentralWidget(main_widget)
    
    def update_alpha(self, value):
        self.alpha = value / 10.0
        self.alpha_label.setText(f"{self.alpha:.1f}")
        
    def update_focal_scale(self, value):
        self.focal_scale = value / 10.0
        self.focal_label.setText(f"{self.focal_scale:.1f}")
    
    def load_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.image_folder = folder
            try:
                print(f"选择的文件夹: {folder}")
                
                # 尝试加载transform.json或transforms.json
                json_files = ["transform.json", "transforms.json"]
                json_path = None
                
                for json_file in json_files:
                    path = os.path.join(folder, json_file)
                    if os.path.exists(path):
                        json_path = path
                        print(f"找到JSON文件: {json_path}")
                        break
                
                if json_path is None:
                    raise FileNotFoundError("找不到transform.json或transforms.json文件")
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.transform_data = json.load(f)
                
                # 加载图片列表
                self.images = []
                self.image_list = []
                self.list_widget.clear()
                
                # 计算相机内参矩阵
                self.calculate_intrinsic_matrix()
                
                # 获取文件夹中所有图片文件
                image_files = []
                for file in os.listdir(folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(file)
                
                print(f"文件夹中找到的图片文件: {image_files}")
                
                # 匹配JSON中的帧和实际图片文件
                for frame in self.transform_data["frames"]:
                    # 提取文件名（不含路径）
                    file_name = os.path.basename(frame["file_path"])
                    
                    # 尝试匹配实际文件
                    found = False
                    for img_file in image_files:
                        if img_file.lower() == file_name.lower():
                            img_path = os.path.join(folder, img_file)
                            self.images.append(img_path)
                            self.image_list.append(img_file)
                            self.list_widget.addItem(img_file)
                            found = True
                            break
                    
                    if not found:
                        print(f"警告: 找不到图片文件 {file_name}")
                
                if not self.images:
                    raise ValueError("没有找到有效的图片文件")
                
                print(f"成功加载 {len(self.images)} 张图片")
                
                self.epipolar_button.setEnabled(True)
                self.rectify_button.setEnabled(True)
                QMessageBox.information(self, "成功", f"已加载 {len(self.images)} 张图片和相机参数")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载图片或相机参数文件: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def calculate_intrinsic_matrix(self):
        """从camera_angle_x计算相机内参矩阵"""
        # 假设所有图片尺寸相同，使用第一张图片确定尺寸
        if not self.transform_data["frames"]:
            raise ValueError("JSON中没有找到图片帧数据")
        
        # 获取文件夹中第一张图片
        img_path = None
        for file in os.listdir(self.image_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.image_folder, file)
                break
        
        if img_path is None:
            raise FileNotFoundError("文件夹中没有找到图片文件")
        
        print(f"使用图片计算内参: {img_path}")
        
        # 使用OpenCV读取图片
        img = self.robust_imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
        
        h, w = img.shape[:2]
        
        # 从camera_angle_x计算焦距
        camera_angle_x = self.transform_data["camera_angle_x"]
        
        # 方法1: 使用原始公式
        focal_length = 0.5 * w / np.tan(0.5 * camera_angle_x)
        
        # 方法2: 使用对角线视角
        # diag = np.sqrt(w**2 + h**2)
        # focal_length = diag / (2 * np.tan(0.5 * camera_angle_x))
        
        # 应用缩放因子
        focal_length *= self.focal_scale
        
        # 创建内参矩阵
        self.K = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ])
        
        print(f"计算相机内参矩阵: 图像尺寸={w}x{h}, 焦距={focal_length:.2f}")
        print(self.K)
    
    def robust_imread(self, path):
        """更健壮的图片读取方法，处理路径中的非ASCII字符"""
        # 方法1: 使用numpy从文件读取
        try:
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except:
            pass
        
        # 方法2: 尝试使用Qt加载
        try:
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                img = self.pixmap_to_cv(pixmap)
                if img is not None:
                    return img
        except:
            pass
        
        # 方法3: 尝试直接读取
        try:
            img = cv2.imread(path)
            if img is not None:
                return img
        except:
            pass
        
        print(f"所有方法都无法读取图片: {path}")
        return None
    
    def pixmap_to_cv(self, pixmap):
        """将QPixmap转换为OpenCV图像"""
        image = pixmap.toImage()
        width = image.width()
        height = image.height()
        
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # 4 for RGBA
        
        # 转换为BGR格式
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return bgr
    
    def extract_pose(self, transform_matrix):
        """从4x4变换矩阵中提取旋转矩阵和平移向量"""
        # 提取旋转部分 (3x3)
        R = np.array(transform_matrix)[:3, :3]
        
        # 提取平移部分
        t = np.array(transform_matrix)[:3, 3]
        
        # 确保旋转矩阵是正交的
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        # 如果行列式为负，则调整
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = U @ Vt
        
        return R, t
    
    def get_selected_images(self):
        selected_items = self.list_widget.selectedItems()
        if len(selected_items) != 2:
            QMessageBox.warning(self, "警告", "请选择两张图片")
            return None, None
        
        idx1 = self.list_widget.row(selected_items[0])
        idx2 = self.list_widget.row(selected_items[1])
        return idx1, idx2
    
    def show_epipolar_lines(self):
        idx1, idx2 = self.get_selected_images()
        if idx1 is None or idx2 is None:
            return
        
        # 加载图片
        img1 = self.robust_imread(self.images[idx1])
        img2 = self.robust_imread(self.images[idx2])
        
        if img1 is None or img2 is None:
            QMessageBox.critical(self, "错误", "无法加载图片")
            return
        
        print(f"处理图片: {os.path.basename(self.images[idx1])} 和 {os.path.basename(self.images[idx2])}")
        
        try:
            # 获取相机位姿
            R1, t1 = self.extract_pose(self.transform_data["frames"][idx1]["transform_matrix"])
            R2, t2 = self.extract_pose(self.transform_data["frames"][idx2]["transform_matrix"])
            
            print("R1:\n", R1)
            print("t1:", t1)
            print("R2:\n", R2)
            print("t2:", t2)
            
            # 计算相对位姿
            R_rel = R2 @ R1.T
            t_rel = t2 - R_rel @ t1
            
            print("相对旋转:\n", R_rel)
            print("相对平移:", t_rel)
            
            # 计算本质矩阵
            T = np.array([
                [0, -t_rel[2], t_rel[1]],
                [t_rel[2], 0, -t_rel[0]],
                [-t_rel[1], t_rel[0], 0]
            ])
            E = T @ R_rel
            
            # 计算基础矩阵
            F = np.linalg.inv(self.K).T @ E @ np.linalg.inv(self.K)
            print("基础矩阵:\n", F)
            
            # 根据选择创建特征检测器
            method = self.feature_method.currentText()
            if method == "SIFT":
                detector = cv2.SIFT_create()
            elif method == "ORB":
                detector = cv2.ORB_create()
            elif method == "AKAZE":
                detector = cv2.AKAZE_create()
            else:
                detector = cv2.SIFT_create()
            
            # 检测特征点
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                QMessageBox.warning(self, "警告", "无法检测到足够的特征点")
                return
            
            print(f"检测到特征点: 图片1={len(kp1)}, 图片2={len(kp2)}")
            
            # 匹配特征点
            if method == "SIFT":
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                
            matches = bf.match(des1, des2)
            
            if not matches:
                QMessageBox.warning(self, "警告", "无法匹配特征点")
                return
            
            print(f"找到匹配点: {len(matches)}")
            
            # 按距离排序并选择前N个
            matches = sorted(matches, key=lambda x: x.distance)
            num_matches = int(self.match_count.currentText())
            matches = matches[:min(num_matches, len(matches))]
            
            # 绘制对极线
            img1_lines = img1.copy()
            img2_lines = img2.copy()
            
            # 绘制匹配点
            for match in matches:
                pt1 = kp1[match.queryIdx].pt
                pt2 = kp2[match.trainIdx].pt
                
                # 在img1上绘制点
                cv2.circle(img1_lines, (int(pt1[0]), int(pt1[1])), 8, (0, 0, 255), -1)
                
                # 在img2上绘制点
                cv2.circle(img2_lines, (int(pt2[0]), int(pt2[1])), 8, (0, 0, 255), -1)
                
                # 计算img1上的点在img2上的对极线
                line = F @ np.array([pt1[0], pt1[1], 1])
                a, b, c = line
                h, w = img2.shape[:2]
                
                # 计算线段的两个端点
                if abs(b) > 1e-5:  # 避免除以零
                    x0 = 0
                    y0 = int(-c / b)
                    x1 = w
                    y1 = int(-(c + a * w) / b)
                    
                    # 确保坐标在图像范围内
                    if y0 < 0 or y0 > h:
                        if a != 0:
                            y0 = 0
                            x0 = int(-(b * y0 + c) / a)
                            if x0 < 0 or x0 > w:
                                y0 = h
                                x0 = int(-(b * y0 + c) / a)
                    
                    if y1 < 0 or y1 > h:
                        if a != 0:
                            y1 = 0
                            x1 = int(-(b * y1 + c) / a)
                            if x1 < 0 or x1 > w:
                                y1 = h
                                x1 = int(-(b * y1 + c) / a)
                    
                    # 裁剪坐标到图像边界
                    x0 = max(0, min(w-1, x0))
                    y0 = max(0, min(h-1, y0))
                    x1 = max(0, min(w-1, x1))
                    y1 = max(0, min(h-1, y1))
                    
                    cv2.line(img2_lines, (x0, y0), (x1, y1), (0, 255, 0), 2)
                
                # 计算img2上的点在img1上的对极线
                line = F.T @ np.array([pt2[0], pt2[1], 1])
                a, b, c = line
                h, w = img1.shape[:2]
                
                # 计算线段的两个端点
                if abs(b) > 1e-5:  # 避免除以零
                    x0 = 0
                    y0 = int(-c / b)
                    x1 = w
                    y1 = int(-(c + a * w) / b)
                    
                    # 确保坐标在图像范围内
                    if y0 < 0 or y0 > h:
                        if a != 0:
                            y0 = 0
                            x0 = int(-(b * y0 + c) / a)
                            if x0 < 0 or x0 > w:
                                y0 = h
                                x0 = int(-(b * y0 + c) / a)
                    
                    if y1 < 0 or y1 > h:
                        if a != 0:
                            y1 = 0
                            x1 = int(-(b * y1 + c) / a)
                            if x1 < 0 or x1 > w:
                                y1 = h
                                x1 = int(-(b * y1 + c) / a)
                    
                    # 裁剪坐标到图像边界
                    x0 = max(0, min(w-1, x0))
                    y0 = max(0, min(h-1, y0))
                    x1 = max(0, min(w-1, x1))
                    y1 = max(0, min(h-1, y1))
                    
                    cv2.line(img1_lines, (x0, y0), (x1, y1), (0, 255, 0), 2)
            
            # 显示图片
            self.display_image(img1, self.image_label1)
            self.display_image(img2, self.image_label2)
            self.display_image(img1_lines, self.image_label1_rect)
            self.display_image(img2_lines, self.image_label2_rect)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def rectify_images(self):
        idx1, idx2 = self.get_selected_images()
        if idx1 is None or idx2 is None:
            return
        
        # 加载图片
        img1 = self.robust_imread(self.images[idx1])
        img2 = self.robust_imread(self.images[idx2])
        
        if img1 is None or img2 is None:
            QMessageBox.critical(self, "错误", "无法加载图片")
            return
        
        print(f"矫正图片: {os.path.basename(self.images[idx1])} 和 {os.path.basename(self.images[idx2])}")
        
        try:
            # 获取相机位姿
            T1 = np.array(self.transform_data["frames"][idx1]["transform_matrix"])
            T2 = np.array(self.transform_data["frames"][idx2]["transform_matrix"])
            
            # 方法1: 直接提取旋转和平移
            R1, t1 = self.extract_pose(T1)
            R2, t2 = self.extract_pose(T2)
            
            # 方法2: 计算相对变换
            T_rel = T2 @ np.linalg.inv(T1)
            R_rel = T_rel[:3, :3]
            t_rel = T_rel[:3, 3]
            
            # 使用第二种方法计算相对位姿
            # R_rel = R2 @ R1.T
            # t_rel = t2 - R_rel @ t1
            
            print("相对旋转:\n", R_rel)
            print("相对平移:", t_rel)
            
            # 立体校正
            R1_rect, R2_rect, P1, P2, Q, _, _ = cv2.stereoRectify(
                cameraMatrix1=self.K, distCoeffs1=None,
                cameraMatrix2=self.K, distCoeffs2=None,
                imageSize=img1.shape[:2],
                R=R_rel, T=t_rel,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=self.alpha
            )
            
            print("校正旋转1:\n", R1_rect)
            print("校正旋转2:\n", R2_rect)
            
            # 计算映射矩阵
            map1x, map1y = cv2.initUndistortRectifyMap(
                self.K, None, R1_rect, P1, img1.shape[:2], cv2.CV_32FC1
            )
            map2x, map2y = cv2.initUndistortRectifyMap(
                self.K, None, R2_rect, P2, img2.shape[:2], cv2.CV_32FC1
            )
            
            # 应用校正
            img1_rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
            img2_rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
            
            # 绘制水平线以验证校正
            img1_rect_lines = img1_rect.copy()
            img2_rect_lines = img2_rect.copy()
            
            h, w = img1_rect.shape[:2]
            for y in range(0, h, 50):
                cv2.line(img1_rect_lines, (0, y), (w, y), (0, 255, 0), 1)
                cv2.line(img2_rect_lines, (0, y), (w, y), (0, 255, 0), 1)
            
            # 显示结果
            self.display_image(img1, self.image_label1)
            self.display_image(img2, self.image_label2)
            self.display_image(img1_rect_lines, self.image_label1_rect)
            self.display_image(img2_rect_lines, self.image_label2_rect)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"视角矫正过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_image(self, img, label):
        """将OpenCV图像显示在QLabel上"""
        if img is None:
            return
        
        # 调整图像大小以适应标签
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        
        # 缩放图像以适应标签大小
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(label.width(), label.height(), 
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        
        # 更新标签尺寸以保持宽高比
        label.setFixedSize(pixmap.size())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
