**整体项目目录解析：**

--selected paper 选取的五篇论文

--Paper Research pdf形式的阅读报告

--Mini-project V1 第一版的项目 代码解读与效果展示pdf

--Mini-project V2 第二版的项目 代码解读与效果展示pdf

--学习资料 一些学习的笔记

**1.选取的论文：**(Lerf由于文件太大没有上传)

[1]Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3d gaussian splatting for  real-time radiance field rendering. ACM Trans. Graph., 42(4), 139-1.

[2]Kerr, J., Kim, C. M., Goldberg, K., Kanazawa, A., & Tancik, M. (2023). Lerf: Language  embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on  Computer Vision (pp. 19729-19739). 

[3]Qin, M., Li, W., Zhou, J., Wang, H., & Pfister, H. (2024). Langsplat: 3d language gaussian  splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern  Recognition (pp. 20051-20060).

[4]Ji, Y., Zhu, H., Tang, J., Liu, W., Zhang, Z., Tan, X., & Xie, Y. (2025, April). Fastlgs: Speeding  up language embedded gaussians with feature grid mapping. In Proceedings of the AAAI  Conference on Artificial Intelligence (Vol. 39, No. 4, pp. 3922-3930).

[5]Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., & Novotny, D. (2025). Vggt:  Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern  Recognition Conference (pp. 5294-5306).

**2.关于论文报告：**

我整体的写作风格可能是：学术化的文字，加上很多我个人对学术和该领域的初体验(一些体验与感受)

具体可以看目录，首先是对三维重建与分解领域的整体认识，具体到每一篇文章里：

首先是文章的动机，接着是方法，随后的是该工作整体的优缺点，

有几篇文章，我还进行了简单的复现，还有最后的个人小结

**3.项目思路的推荐阅读顺序**

我现在对老师要做的这个东西有三种理解，我都做了：
1.进行简单的三维重建(论文报告里，复现了3DGS和VGGT)

2.进行一次SFM，这里借助了COLMAP实现了，在V2的文件夹里

3.通过opencv库等实现计算得到相机内参矩阵、基础矩阵和本质矩阵，实现视角纠正与画出双级线(软件的V1与V2实现的都是这个，V2使用了COLMAP处理得到后的数据)

所以我的心路历程是，先复现了两个三维重建的方法，接着用图形学知识实现视角纠正与画出双级线(带有一些近似的假设)，
后面用COLMAP实现了一次SFM,再用COLMAP得到的数据去实现更好的视角纠正与画出双级线，即软件V2
