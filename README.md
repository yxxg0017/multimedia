# 智能纹饰提取与艺术展馆项目 (Intelligent Ornament Extraction)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20PyTorch-orange)](https://pytorch.org/)
[![Project URL](https://img.shields.io/badge/GitHub-Repository-green)](https://github.com/yxxg0017/multimedia)

> **项目地址**: [https://github.com/yxxg0017/multimedia](https://github.com/yxxg0017/multimedia)

## 1. 项目概述

本项目是一个集成了深度学习与传统计算机视觉技术的智能图像处理系统，旨在从输入的器物图片中自动识别、提取核心的装饰性纹样，并将其通过一个动态的 Web 界面进行展示。

系统不仅能够批量处理图片，还提供了一个交互式的 Web 应用，让用户可以上传自己的图片进行实时的纹饰提取体验。

## 2. 核心功能

* **🏛️ 数据驱动的艺术展馆**
    基于 `tags.csv`和 `story.csv` 文件，动态生成一个按朝代分类的纹饰展馆网页，展示已提取的纹饰并附带相关介绍。

* **⚙️ 自动化批处理**
    提供一个强大的命令行工具 (`pipeline.py`)，能够对指定目录下的所有图片进行全自动、端到端的处理，包括显著性检测、纹饰提取和背景透明化。

* **⚡ 单图实时处理**
    用户可通过 Web 界面上传单张图片，后端会异步处理该图片，并实时反馈处理进度和最终结果。

* **🎨 核心纹饰提取**
    项目的核心算法能够从复杂的背景中精准分离出物体，并进一步从物体本身提取出最具代表性的核心装饰图案。

## 3. 技术架构与算法详解

项目的工作流程分为两大阶段：**显著性物体检测** 和 **核心纹饰提取**。

### 3.1. 技术栈

| 领域 | 技术/库 |
| :--- | :--- |
| **后端框架** | Flask |
| **深度学习** | PyTorch, Torchvision |
| **计算机视觉** | OpenCV, Pillow (PIL), Scikit-image |
| **数值计算** | NumPy, SciPy |
| **Web 前端** | HTML, CSS, JavaScript (通过 Flask 模板渲染) |

### 3.2. 核心算法

#### 阶段一：显著性物体检测 (U-2-Net)

* **模型**: 本项目采用 **U-2-Net**。这是一种先进的深度学习模型，专门用于从图像中精确地分割出最显著的物体。它的双层嵌套 U 型结构使其能够在不增加过多计算量的情况下，捕捉丰富的上下文信息，从而生成高质量的物体蒙版（Saliency Mask）。
* **流程**: 输入一张原始图片 $I_{original}$，U-2-Net 模型会输出一张灰度图 $R$ (Saliency Mask)，其中高亮区域代表识别出的主要物体。

#### 阶段二：核心纹饰提取 (传统 CV 算法)

这是本项目的创新核心。在获得 U-2-Net 生成的物体蒙版后，通过一系列精巧的图像处理步骤来提取纹饰：

1.  **图像差分 (Image Differencing)**
    将原始图像的灰度图 $I_G$ 与 U-2-Net 生成的蒙版 $R_G$ 进行相减。这一步利用了蒙版相对平滑、而原图纹理丰富的特点，差分操作能够有效地放大和凸显出物体表面的精细纹路。

    差分图像 $X$ 的计算公式为：
    ```math
    X = R_G - I_G
    ```
    其中，$X$ 的值越高，表示该区域越有可能是纹饰细节。

2.  **动态阈值分割 (Dynamic Thresholding)**
    为了将纹饰图案与物体主体分离开，我们对差分图像 $X$ 计算一个动态阈值 $T$。该阈值根据图像内容自适应计算，使其对不同光照和对比度的图片具有更好的鲁棒性。

    计算公式为：
    ```math
    T = \frac{\mu(X_{roi}) + C}{2}
    ```
    * $\mu(X_{roi})$: 差分图像 $X$ 在显著性区域内（即蒙版 $R > 0$）的像素均值。
    * $C$: 经验常数（代码中根据差分值的正负取 50 或 150）。

    随后，通过该阈值生成二值化蒙版 $Y$：
    ```math
    Y(i, j) = \begin{cases} 255 & \text{if } X(i, j) \ge T \\ 0 & \text{otherwise} \end{cases}
    ```

3.  **形态学处理 (Morphological Operations)**
    使用 **形态学闭运算 (Morphological Closing)** 来填充二值化后纹饰图案内部的微小空洞，并连接邻近的断裂部分，使纹饰的蒙版更加完整和连贯。

4.  **连通组件分析 (Connected-Component Analysis)**
    识别出所有独立的纹饰区域 $C_1, C_2, ..., C_n$。这一步可以过滤掉面积过小的噪点，并为后续筛选核心纹饰做准备。

5.  **高斯加权定位 (Gaussian-Weighted Localization)**
    为了找到“核心”部分，算法为每个组件 $C_k$ 计算总权重 $W_k$。该权重基于组件内像素到图像中心的距离 $d$，由高斯函数加权（假设中心区域为设计主体）。

    ```math
    w(d) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(d-\mu)^2}{2\sigma^2}}
    ```

    组件的总权重为：
    ```math
    W_k = \sum_{(x,y) \in C_k} w\left(\sqrt{(x - x_c)^2 + (y - y_c)^2}\right)
    ```
    最终，总权重 $W_k$ 最大的组件被选为核心纹饰。

6.  **背景透明化与输出**
    最后，将提取出的核心纹饰的黑色背景转换为透明的 Alpha 通道，并保存为 PNG 格式。至此，一张背景干净、主体突出的纹饰图片即处理完成，可直接用于展馆展示。

## 4. 如何运行项目

### 4.1. 环境配置

请按照以下步骤配置 Python 环境及依赖：

1.  **克隆代码仓库**
    ```bash
    git clone [https://github.com/yxxg0017/multimedia.git](https://github.com/yxxg0017/multimedia.git)
    cd multimedia
    ```

2.  **创建并激活虚拟环境**
    建议使用 Conda 管理环境以避免版本冲突：
    ```bash
    conda create -n art-env python=3.8
    conda activate art-env
    ```

3.  **安装项目依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置预训练模型**
    本项目依赖 U-2-Net 模型进行显著性检测：
    * **下载**: 请下载 `u2net.pth` 权重文件（[下载链接](https://github.com/xuebinqin/U-2-Net)）。
    * **放置**: 将文件移动至项目的 `saved_models/u2net/` 目录下。

### 4.2. 运行 Web 应用

启动 Flask Web 服务器，访问纹饰展馆和图片上传功能。

1.  **数据准备**: 确保项目根目录下存在 `tags.csv` 和 `story.csv` 文件（用于生成展馆数据）。
2.  **启动服务**:
    ```bash
    python app.py
    ```
3.  **访问应用**: 打开浏览器访问 `http://localhost:5000`。

### 4.3. 运行批处理流水线

若需对本地大量图片进行自动化处理（提取纹饰并去背），可使用 CLI 工具 `pipeline.py`。

**命令示例:**

```bash
python pipeline.py --images ./my_images --output ./my_results
````

**参数详解:**

| 参数 | 说明 |
| :--- | :--- |
| `--images` | **输入目录**：包含待处理原始图片的文件夹路径。 |
| `--output` | **输出目录**：处理结果的保存路径。脚本会自动创建该目录。 |

> **提示**: 处理完成后的最终纹饰图片将保存在输出目录下的 `final_output/` 子文件夹中。

