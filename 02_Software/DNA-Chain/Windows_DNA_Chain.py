import sys
import os
import cv2
import numpy as np
import heapq
import time
import hashlib
from collections import defaultdict

from reedsolo import RSCodec

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, \
    QLineEdit, QTreeWidget, QTreeWidgetItem, QMessageBox, QPlainTextEdit, QScrollArea
from PyQt5.QtGui import QPixmap, QImage, QPainter  # 导入QPixmap用于加载图片显示

# ============================================================
# RS 纠错码参数（按论文设计：3.2.5节）
#   - 载荷分块: 150 nt
#   - 地址索引: 3 nt (5-bit, 支持最多32条寡核苷酸)
#   - 文件ID:   6 nt (10-bit, MD5-based, 支持最多1024个文件)
#   - 信息块:   159 nt (= 150 + 3 + 6)
#   - RS校验:   10 字节 (reedsolo nsym=10, GF(2^8))
#               可纠正 5 个字节级错误 (nsym/2=5)
#   - 寡核苷酸总长: 159 nt + RS校验 nt
# ============================================================
PAYLOAD_SEGMENT_NT = 150       # 每条寡核苷酸的载荷长度
INDEX_NT = 6                   # 地址索引碱基数 (10 bits, 支持最多1024条寡核苷酸)
FILE_ID_NT = 6                 # 文件ID碱基数 (10 bits)
INFO_BLOCK_NT = PAYLOAD_SEGMENT_NT + INDEX_NT + FILE_ID_NT  # 162
RS_NSYM = 10                   # reedsolo 纠错字节数, 可纠正 5 个错误

# 定义碱基与二进制编码的对应关系（前两组四进制模型）
base_mapping = {
    "00": "A",
    "01": "T",
    "10": "C",
    "11": "G"
}

# 定义R和Y的取值规则（示例中假设A和G为0，T和C为1）
def define_RY(binary_digit, second_base, current_GC):
    if binary_digit == "0":
        R, Y = "A", "G"
    else:
        R, Y = "T", "C"
    return R, Y

# 将5位二进制转换为3个碱基的函数（编码函数）
def binary_to_three_bases(binary_string, previous_sequence=''):
    # 前两组二进制转换为碱基
    first_base = base_mapping[binary_string[:2]]
    second_base = base_mapping[binary_string[2:4]]
    # 确定R和Y的取值，这里暂不考虑GC含量调整简化处理
    R, Y = define_RY(binary_string[4], second_base, 0)
    # 根据规则确定第三位碱基（确保不与第二位碱基相同且考虑R、Y取值）
    if second_base == R:
        third_base = Y
    else:
        third_base = R
    return first_base + second_base + third_base

# 辅助函数，根据碱基查找对应的二进制编码（反向查找base_mapping字典）
def base_to_binary(base):
    for binary, b in base_mapping.items():
        if b == base:
            return binary
    return ""

# 解码函数，将碱基序列解码回二进制字符串，并去除编码时补的0
def decode_bases(encoded_data):
    original_length, bases = encoded_data
    binary_result = ""
    for i in range(0, len(bases), 3):
        group = bases[i:i + 3]
        if len(group) == 3:
            first_binary = base_to_binary(group[0])
            second_binary = base_to_binary(group[1])
            # 根据编码时确定第三位碱基的规则来反推第5位二进制
            if group[1] == group[2]:
                fifth_binary = "0" if group[1] in ["A", "G"] else "1"
            else:
                fifth_binary = "0" if group[2] in ["A", "G"] else "1"
            binary_result += first_binary + second_binary + fifth_binary
    # 根据原始长度截取得到正确的二进制数据
    return binary_result[:original_length]

def image_to_chaincode_with_start(image_path):
    """
    提取图像的链码并记录每个轮廓的起始点。
    :param image_path: 输入图像路径。
    :return: (链码列表, 起始点列表)。
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    chaincodes = []
    start_points = []
    for contour in contours:
        start_points.append(tuple(contour[0][0]))
        chaincode = []
        for i in range(1, len(contour)):
            dx = contour[i][0][0] - contour[i - 1][0][0]
            dy = contour[i][0][1] - contour[i - 1][0][1]
            if dx == 1 and dy == 0:
                chaincode.append(0)
            elif dx == 1 and dy == -1:
                chaincode.append(1)
            elif dx == 0 and dy == -1:
                chaincode.append(2)
            elif dx == -1 and dy == -1:
                chaincode.append(3)
            elif dx == -1 and dy == 0:
                chaincode.append(4)
            elif dx == -1 and dy == 1:
                chaincode.append(5)
            elif dx == 0 and dy == 1:
                chaincode.append(6)
            elif dx == 1 and dy == 1:
                chaincode.append(7)
        chaincodes.append(chaincode)
    return chaincodes, start_points

def chaincode_to_image_with_start(chaincodes, start_points, image_size):
    """
    根据链码和起始点重建原二值图像。

    链码描述的是物体区域的*边界*；标准重建方式是沿链码 trace 出闭合轮廓，
    再填充轮廓所包围的区域（与编码端 cv2.findContours / THRESH_BINARY 的
    极性一致：物体区域 = 255，背景 = 0）。仅绘制边界像素（描边）会丢失
    区域内部信息，像素级重合率仅有约 34%；填充重建可逐像素还原阈值化
    图像（在链码精确恢复时重合率 = 100%）。
    :param chaincodes: 链码列表。
    :param start_points: 每条链码的起始点列表。
    :param image_size: 图像尺寸 (height, width)。
    :return: 重建的二值图像。
    """
    directions = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
    image = np.zeros(image_size, dtype=np.uint8)  # 背景 0；物体区域填充为 255

    # 收集所有闭合轮廓多边形，*一次性* fillPoly：OpenCV 对多个多边形采用
    # even-odd 规则，嵌套轮廓（物体内部的孔洞）会被正确留空；若逐条累加填充，
    # 内层孔洞会被错误填实，像素重合率下降。
    polys = []
    for chaincode, start_point in zip(chaincodes, start_points):
        x, y = start_point
        pts = [(x, y)]
        for code in chaincode:
            dx, dy = directions[code]
            x += dx
            y += dy
            pts.append((x, y))
        polys.append(np.array(pts, dtype=np.int32))
    if polys:
        cv2.fillPoly(image, polys, 255)
    return image

def encode_chaincodes_with_huffman(chaincodes):
    """
    对链码进行霍夫曼编码。
    :param chaincodes: 链码列表。
    :return: (编码后的二进制数据, 霍夫曼编码表)。
    """
    # 扁平化链码
    flat_chaincodes = [code for chaincode in chaincodes for code in chaincode]

    # 统计频率
    freq = defaultdict(int)
    for code in flat_chaincodes:
        freq[code] += 1

    # 构造最小堆
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)

    # 构造霍夫曼树
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)

        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]

        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # 提取霍夫曼编码表
    huffman_codes = {}
    for pair in heap[0][1:]:
        symbol, code = pair
        huffman_codes[symbol] = code

    # 编码链码
    encoded_data = "".join(huffman_codes[code] for code in flat_chaincodes)

    return encoded_data, huffman_codes

def decode_huffman_to_chaincodes(encoded_data, huffman_codes):
    """
    解码霍夫曼编码为链码。
    :param encoded_data: 编码后的二进制数据。
    :param huffman_codes: 霍夫曼编码表。
    :return: 解码后的链码。
    """
    reverse_huffman_codes = {code: symbol for symbol, code in huffman_codes.items()}

    current_code = ""
    decoded_chaincodes = []
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_huffman_codes:
            decoded_chaincodes.append(reverse_huffman_codes[current_code])
            current_code = ""

    return decoded_chaincodes

# ============================================================
# RS 纠错码编解码函数（按论文设计：3.2.5节）
# ============================================================

def _bits_to_bytes(bits):
    """将二进制字符串转为字节，不足8位时末尾补0。"""
    pad = (8 - len(bits) % 8) % 8
    bits = bits + '0' * pad
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8)), pad


def _bytes_to_bits(data, n_bits=None):
    """将字节转回二进制字符串，可指定截取长度。"""
    bits = ''.join(format(b, '08b') for b in data)
    return bits[:n_bits] if n_bits else bits


def _generate_file_id(image_path):
    """根据图像MD5生成10位文件ID (0-1023)。"""
    h = hashlib.md5()
    with open(image_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return int(h.hexdigest()[:3], 16) % 1024  # 10 bits


def _int_to_bases(value, n_bits, n_nt):
    """将整数转为指定碱基数：int→二进制(补零)→5bit→3base映射。"""
    bits = format(value, f'0{n_bits}b')
    # 补齐到5的倍数
    pad = (5 - len(bits) % 5) % 5
    bits = bits + '0' * pad
    bases = ""
    for i in range(0, len(bits), 5):
        group = bits[i:i+5]
        if len(group) == 5:
            bases += binary_to_three_bases(group)
    return bases[:n_nt]


def _bases_to_int(bases, n_bits):
    """从碱基序列还原整数值。"""
    bits = ""
    for i in range(0, len(bases), 3):
        group = bases[i:i+3]
        if len(group) == 3:
            b0 = base_to_binary(group[0])
            b1 = base_to_binary(group[1])
            if group[1] == group[2]:
                b2 = "0" if group[1] in ["A", "G"] else "1"
            else:
                b2 = "0" if group[2] in ["A", "G"] else "1"
            bits += b0 + b1 + b2
    return int(bits[:n_bits], 2) if bits else 0


def encode_with_rs(encoded_bases, image_path):
    """
    对已映射的碱基序列添加分块索引、文件ID和RS纠错码。
    按论文设计：
      1. 将碱基序列分为150nt载荷段
      2. 每段末尾追加3nt地址索引 + 6nt文件ID = 159nt信息块
      3. 将159nt信息块转为比特(逆向5b→3b)→字节，用reedsolo生成RS校验
      4. RS校验字节转为碱基，置于信息块5'端
    返回: (oligos列表, file_id, n_payload_segments)
    """
    file_id = _generate_file_id(image_path)

    # 1. 分块（用有效三联体 "AAG" 填充末段，对应 bits "00000"）
    PAD_TRIPLET = binary_to_three_bases("00000")  # valid 5b→3b output
    segments = []
    for i in range(0, len(encoded_bases), PAYLOAD_SEGMENT_NT):
        seg = encoded_bases[i:i + PAYLOAD_SEGMENT_NT]
        if len(seg) < PAYLOAD_SEGMENT_NT:
            pad_len = PAYLOAD_SEGMENT_NT - len(seg)
            # 确保填充到3的倍数
            seg = seg + PAD_TRIPLET * (pad_len // 3 + 1)
            seg = seg[:PAYLOAD_SEGMENT_NT]
        segments.append(seg)

    n_segments = len(segments)
    if n_segments > 1024:
        raise ValueError(f"段数{n_segments}超过地址索引容量(1024)")

    oligos = []
    rsc = RSCodec(nsym=RS_NSYM)

    for idx, seg in enumerate(segments):
        # 2. 追加地址索引(3nt)和文件ID(6nt)
        index_bases = _int_to_bases(idx, 10, INDEX_NT)
        fileid_bases = _int_to_bases(file_id, 10, FILE_ID_NT)
        info_block = seg + index_bases + fileid_bases  # 162 nt

        # 3. 信息块→比特→字节→RS编码
        # 逆向5b→3b: 每3个碱基→5bit
        info_bits = ""
        for j in range(0, len(info_block), 3):
            group = info_block[j:j + 3]
            if len(group) == 3:
                b0 = base_to_binary(group[0])
                b1 = base_to_binary(group[1])
                if group[1] == group[2]:
                    b2 = "0" if group[1] in ["A", "G"] else "1"
                else:
                    b2 = "0" if group[2] in ["A", "G"] else "1"
                info_bits += b0 + b1 + b2

        info_bytes, pad_bits = _bits_to_bytes(info_bits)
        # RS编码: info_bytes → info_bytes + parity_bytes
        encoded_full = rsc.encode(info_bytes)
        parity_bytes = encoded_full[len(info_bytes):]

        # 4. RS校验字节→比特→碱基(5b→3b)
        parity_bits = ''.join(format(b, '08b') for b in parity_bytes)
        # 补齐到5的倍数
        pad_p = (5 - len(parity_bits) % 5) % 5
        parity_bits_padded = parity_bits + '0' * pad_p
        parity_bases = ""
        for j in range(0, len(parity_bits_padded), 5):
            group = parity_bits_padded[j:j + 5]
            if len(group) == 5:
                parity_bases += binary_to_three_bases(group)

        # 组装: [RS校验碱基] + [载荷150nt] + [索引3nt] + [文件ID 6nt]
        oligo = parity_bases + info_block
        oligos.append(oligo)

    return oligos, file_id, n_segments


def decode_with_rs(oligos, payload_bits, file_id):
    """
    对带RS纠错码的寡核苷酸列表进行解码。
    流程:
      1. 对每条寡核苷酸: 分离RS校验碱基和信息块
      2. 信息块→比特→字节, RS校验碱基→比特→字节
      3. reedsolo解码纠错
      4. 纠错后的信息块→比特→还原载荷碱基
      5. 按地址索引排序, 拼接载荷
    返回: (corrected_bases, n_corrected)
    """
    rsc = RSCodec(nsym=RS_NSYM)
    segments_by_idx = {}

    for oligo in oligos:
        # 计算RS校验碱基数: parity_bytes=10, 10*8=80bits, 补到5的倍数=80, 80/5*3=48nt
        parity_nt = (RS_NSYM * 8 + 4) // 5 * 3  # 48 nt
        parity_bases = oligo[:parity_nt]
        info_block = oligo[parity_nt:]  # 159 nt

        # 信息块→比特→字节
        info_bits = ""
        for j in range(0, len(info_block), 3):
            group = info_block[j:j + 3]
            if len(group) == 3:
                b0 = base_to_binary(group[0])
                b1 = base_to_binary(group[1])
                if group[1] == group[2]:
                    b2 = "0" if group[1] in ["A", "G"] else "1"
                else:
                    b2 = "0" if group[2] in ["A", "G"] else "1"
                info_bits += b0 + b1 + b2

        info_bytes, pad_bits = _bits_to_bytes(info_bits)

        # RS校验碱基→比特→字节
        parity_bits = ""
        for j in range(0, len(parity_bases), 3):
            group = parity_bases[j:j + 3]
            if len(group) == 3:
                b0 = base_to_binary(group[0])
                b1 = base_to_binary(group[1])
                if group[1] == group[2]:
                    b2 = "0" if group[1] in ["A", "G"] else "1"
                else:
                    b2 = "0" if group[2] in ["A", "G"] else "1"
                parity_bits += b0 + b1 + b2

        parity_bytes = bytes(int(parity_bits[i:i+8], 2)
                             for i in range(0, len(parity_bits) - 7, 8))

        # RS解码纠错
        full_data = info_bytes + parity_bytes
        try:
            corrected, _, _ = rsc.decode(full_data)
            corrected_bytes = bytes(corrected)
        except Exception:
            # 纠错失败，使用原始信息
            corrected_bytes = info_bytes

        # 纠错后字节→比特
        corrected_bits = _bytes_to_bits(corrected_bytes, len(info_bits))

        # 还原信息块碱基
        corrected_info_bases = ""
        for j in range(0, len(corrected_bits), 5):
            group = corrected_bits[j:j + 5]
            if len(group) == 5:
                corrected_info_bases += binary_to_three_bases(group)
        corrected_info_bases = corrected_info_bases[:INFO_BLOCK_NT]

        # 提取地址索引和载荷
        payload_seg = corrected_info_bases[:PAYLOAD_SEGMENT_NT]
        index_bases = corrected_info_bases[PAYLOAD_SEGMENT_NT:PAYLOAD_SEGMENT_NT + INDEX_NT]
        idx = _bases_to_int(index_bases, 10)
        segments_by_idx[idx] = payload_seg

    # 按索引排序拼接
    sorted_payloads = [segments_by_idx[i] for i in sorted(segments_by_idx.keys())]
    corrected_bases = "".join(sorted_payloads)

    return corrected_bases


def process_image_wrapper(image_path, image_size):
    """
    封装后的函数，整合整个图像处理、编码解码流程，用于按钮点击调用。
    :param image_path: 输入图像路径。
    :param image_size: 图像尺寸。
    :return: 还原后的图像（以QImage格式返回以便在PyQt5中展示）、编码后的碱基序列、解码后的碱基序列、编码时间、解码时间。
    """
    # 提取链码和起始点
    chaincodes, start_points = image_to_chaincode_with_start(image_path)
    
    # 开始计时编码过程
    encode_start_time = time.time()
    # 霍夫曼编码链码
    encoded_chaincodes, huffman_codes = encode_chaincodes_with_huffman(chaincodes)
    
    # 将编码后的二进制数据转换为碱基序列（编码碱基序列）
    encoded_bases = ""
    for i in range(0, len(encoded_chaincodes), 5):
        group = encoded_chaincodes[i:i + 5]
        if len(group) == 5:
            encoded_bases += binary_to_three_bases(group)
    # 结束计时编码过程
    encode_end_time = time.time()
    encode_time = encode_end_time - encode_start_time
    
    # 开始计时解码过程
    decode_start_time = time.time()
    # 霍夫曼解码
    decoded_chaincodes = decode_huffman_to_chaincodes(encoded_chaincodes, huffman_codes)
    
    # 将解码后的链码还原为分组链码
    grouped_chaincodes = []
    index = 0
    for chaincode in chaincodes:
        grouped_chaincodes.append(decoded_chaincodes[index:index + len(chaincode)])
        index += len(chaincode)
    
    # 生成还原图像
    restored_image = chaincode_to_image_with_start(grouped_chaincodes, start_points, image_size)
    
    # 将分组链码对应的二进制数据拼接起来（去除编码时多余补的0）
    binary_result = ""
    for chaincode in grouped_chaincodes:
        for code in chaincode:
            binary_result += bin(code)[2:].zfill(3)
    # 还原碱基序列（解码碱基序列）
    decoded_bases = decode_bases((len(binary_result), binary_result))
    # 结束计时解码过程
    decode_end_time = time.time()
    decode_time = decode_end_time - decode_start_time

    # 将OpenCV的图像格式转换为PyQt5能展示的QImage格式
    height, width = restored_image.shape[:2]
    bytes_per_line = width * 1
    q_image = QImage(restored_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    return q_image, encoded_bases, decoded_bases, encode_time, decode_time

class InteractiveImageLabel(QLabel):
    """支持鼠标拖拽和滚轮缩放的图片标签"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)  # 不自动缩放，由我们自己控制
        self.pixmap = QPixmap()
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.drag_start = QPoint(0, 0)
        self.dragging = False
        self.original_size = QPoint(0, 0)

    def setPixmap(self, pixmap):
        """设置图片并记录原始尺寸"""
        super().setPixmap(pixmap)
        self.pixmap = pixmap
        self.original_size = QPoint(pixmap.width(), pixmap.height())
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.update()

    def paintEvent(self, event):
        """重绘事件，根据缩放和偏移量绘制图片"""
        if self.pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        scaled_width = int(self.original_size.x() * self.scale_factor)
        scaled_height = int(self.original_size.y() * self.scale_factor)

        # 计算绘制位置，确保图片居中并应用偏移
        x = (self.width() - scaled_width) // 2 + self.offset.x()
        y = (self.height() - scaled_height) // 2 + self.offset.y()

        # 绘制缩放后的图片
        painter.drawPixmap(x, y, scaled_width, scaled_height, self.pixmap)

    def wheelEvent(self, event):
        """滚轮事件处理 - 实现缩放功能"""
        if not self.pixmap.isNull():
            # 计算滚轮滚动量
            delta = event.angleDelta().y()
            if delta > 0:
                self.scale_factor *= 1.1
            else:
                self.scale_factor /= 1.1

            # 限制缩放范围
            self.scale_factor = max(0.5, min(self.scale_factor, 5.0))
            self.update()
        event.accept()

    def mousePressEvent(self, event):
        """鼠标按下事件 - 开始拖拽"""
        if event.button() == Qt.LeftButton and not self.pixmap.isNull():
            self.dragging = True
            self.drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 执行拖拽"""
        if self.dragging and not self.pixmap.isNull():
            # 计算偏移量
            delta = event.pos() - self.drag_start
            self.offset += delta
            self.drag_start = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放事件 - 结束拖拽"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
        super().mouseReleaseEvent(event)

    def reset_view(self):
        """重置视图到原始状态"""
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.update()

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(2000, 1200)
        self.setWindowTitle("DNA—Chain-Window")

        # 提取通用的QLabel样式，设置字体大小为20px以及白色背景、内边距等样式
        common_label_style = """
                                    QLabel {
                                        background-color: #D4C8B9;  /* 浅米色背景 */
                                        padding: 10px;
                                        font-size: 20px;
                                        border: none;
                                    }
                                """
        # 提取通用的QLineEdit样式，设置字体大小为20px
        common_line_edit_style = """
                                QLineEdit {
                                    font-size: 20px;
                                }
                            """
        picture_style = """
        QLabel {
            border: 2px dashed gray;
            background-color: white;
            padding: 10px;
        }
            """
        # 提取通用的QPushButton样式，设置字体大小为20px
        common_button_style = """
                                QPushButton {
                                    font-size: 20px;
                                }
                            """

        # 原Box1，应用通用标签样式
        self.Box1 = QLabel(self)
        self.Box1.setStyleSheet(common_label_style)
        self.Box1.resize(600, 1000)
        self.Box1.move(100, 100)

        # 整体布局使用垂直布局来管理Box1内的元素
        vbox = QVBoxLayout(self.Box1)

        # 在Box1中添加新标签，应用通用标签样式
        self.inner_label1 = QLabel("Picture Select", self.Box1)
        self.inner_label1.setStyleSheet(common_label_style)
        vbox.addWidget(self.inner_label1)

        # 用于放置Folder相关元素的水平布局
        hbox_folder = QHBoxLayout()
        self.inner_label2 = QLabel("Folder：", self.Box1)
        self.inner_label2.setStyleSheet(common_label_style)
        hbox_folder.addWidget(self.inner_label2)
        self.text_edit = QLineEdit(self.Box1)
        self.text_edit.setStyleSheet(common_line_edit_style)
        hbox_folder.addWidget(self.text_edit)
        self.button = QPushButton("Browse", self.Box1)
        self.button.setStyleSheet(common_button_style)
        self.button.clicked.connect(self.select_folder)
        hbox_folder.addWidget(self.button)
        vbox.addLayout(hbox_folder)

        # 添加树形组件相关布局及设置，设置QTreeWidget整体样式，包含字体大小20px
        self.tree_widget = QTreeWidget(self.Box1)
        self.tree_widget.setStyleSheet("QTreeWidget{font-size:20px;}")
        self.tree_widget.setHeaderLabels(["file name"])
        self.tree_widget.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        vbox.addWidget(self.tree_widget)

        # 在QTreeWidget下面添加三个标签及相关部件
        # 处理label_1及对应的文本框和按钮
        hbox_label1 = QHBoxLayout()
        self.label_1 = QLabel("Image Path", self.Box1)
        self.label_1.setStyleSheet(common_label_style)
        hbox_label1.addWidget(self.label_1)
        self.text_edit_label1 = QLineEdit(self.Box1)
        self.text_edit_label1.setStyleSheet(common_line_edit_style)
        hbox_label1.addWidget(self.text_edit_label1)
        self.button_label1 = QPushButton("Encode", self.Box1)
        self.button_label1.clicked.connect(self.handle_button_click)
        self.button_label1.setStyleSheet(common_button_style)
        hbox_label1.addWidget(self.button_label1)
        vbox.addLayout(hbox_label1)

        # 处理label_2及对应的文本框和按钮
        hbox_label2 = QHBoxLayout()
        self.label_2 = QLabel("Encode Path", self.Box1)
        self.label_2.setStyleSheet(common_label_style)
        hbox_label2.addWidget(self.label_2)
        self.text_edit_label2 = QLineEdit(self.Box1)
        self.text_edit_label2.setStyleSheet(common_line_edit_style)
        hbox_label2.addWidget(self.text_edit_label2)
        self.button_label2 = QPushButton("Save", self.Box1)
        self.button_label2.clicked.connect(self.save_text_to_file)
        self.button_label2.setStyleSheet(common_button_style)
        hbox_label2.addWidget(self.button_label2)
        vbox.addLayout(hbox_label2)

        # 添加label_3
        #self.label_3 = QLabel("标签3内容", self.Box1)
        #self.label_3.setStyleSheet(common_label_style)
        #vbox.addWidget(self.label_3)

        # Box2相关布局设置
        # Box2相关布局设置 - 修改为使用自定义标签
        self.Box2 = QLabel(self)
        self.Box2.setStyleSheet(common_label_style)
        self.Box2.resize(600, 1000)
        self.Box2.move(705, 100)

        # 为Box2创建内部垂直布局用于放置组件
        vbox_box2 = QVBoxLayout(self.Box2)

        self.inner_label2 = QLabel("Picture Preview", self.Box2)
        self.inner_label2.setStyleSheet(common_label_style)
        self.inner_label2.setFixedSize(600, 50)

        # 创建两个自定义图片标签
        self.label_box2_1 = InteractiveImageLabel(self.Box2)
        self.label_box2_1.setStyleSheet("border: 2px dashed gray; background-color: white; padding: 10px;")
        self.label_box2_2 = InteractiveImageLabel(self.Box2)
        self.label_box2_2.setStyleSheet("border: 2px dashed gray; background-color: white; padding: 10px;")

        # 将两个标签添加到垂直布局中
        vbox_box2.addWidget(self.inner_label2)
        vbox_box2.addWidget(self.label_box2_1)
        vbox_box2.addWidget(self.label_box2_2)

        vbox_box2.setStretchFactor(self.label_box2_1, 1)
        vbox_box2.setStretchFactor(self.label_box2_2, 1)

        # Box3相关设置
        self.inner_label3 = QLabel("DNA Sequance Info", self.Box2)
        self.inner_label3.setStyleSheet(common_label_style)
        self.inner_label3.setFixedSize(600, 50)

        self.inner_label4 = QLabel("CG‘s Content", self.Box2)
        self.inner_label4.setStyleSheet(common_label_style)
        self.inner_label4.setFixedSize(600, 50)

        self.Box3 = QLabel(self)
        self.Box3.setStyleSheet(common_label_style)
        self.Box3.resize(600, 1000)
        self.Box3.move(1310, 100)


        vbox_box3 = QVBoxLayout(self.Box3)
        self.plain_text_edit = QPlainTextEdit()

        vbox_box3.addWidget(self.inner_label3)
        vbox_box3.addWidget(self.plain_text_edit)
        vbox_box3.addWidget(self.inner_label4)


        self.show()

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.text_edit.setText(folder_path)
            self.display_images_in_tree(folder_path)

    def display_images_in_tree(self, folder_path):
        self.tree_widget.clear()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # 常见图片后缀，可按需扩展
        for root, dirs, files in os.walk(folder_path):
            root_item = QTreeWidgetItem(self.tree_widget)
            root_item.setText(0, root)
            for file in files:
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension in image_extensions:
                    file_item = QTreeWidgetItem(root_item)
                    file_item.setText(0, file)
                    file_item.setText(1, "图片文件")
                    file_size = os.path.getsize(os.path.join(root, file))
                    file_item.setText(2, str(file_size) + "字节")
                    file_modify_time = os.path.getmtime(os.path.join(root, file))
                    file_item.setText(3, str(file_modify_time))
                    file_create_time = os.path.getctime(os.path.join(root, file))
                    file_item.setText(4, str(file_create_time))
                else:
                    continue
            for dir_name in dirs:
                dir_item = QTreeWidgetItem(root_item)
                dir_item.setText(0, dir_name)
                dir_item.setText(1, "文件夹")

                #dir_item.setText(2, "")
                #dir_item.setText(3, "")
                #dir_item.setText(4, "")

    def on_tree_item_double_clicked(self, item, column):
        # 判断点击的节点是否是文件（这里根据之前设置的"属性"列文本判断）
        if item.text(1) == "图片文件":
            file_path = os.path.join(self.text_edit.text(), item.text(0))
            self.text_edit_label1.setText(file_path)
            # 尝试加载图片并显示在label_box2_1中
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                self.label_box2_1.setText("无法加载图片")
            else:
                # 获取label_box2_1的当前尺寸
                label_width = self.label_box2_1.width()
                label_height = self.label_box2_1.height()
                # 获取图片原始尺寸
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                # 计算宽高比例，以决定按哪个方向进行等比例缩放（取较小的缩放比例）
                width_ratio = label_width / pixmap_width
                height_ratio = label_height / pixmap_height
                scale_ratio = min(width_ratio, height_ratio)
                # 根据缩放比例计算缩放后的尺寸
                scaled_width = int(pixmap_width * scale_ratio)
                scaled_height = int(pixmap_height * scale_ratio)
                # 缩放图片
                self.label_box2_1.setAlignment(Qt.AlignCenter)
                scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, aspectRatioMode=1)
                self.label_box2_1.setPixmap(scaled_pixmap)

    def handle_button_click(self):
        #image_path = 'DNA-Chain.jpg'  # 这里替换为实际的图像路径
        image_path = self.text_edit_label1.text().strip()
        if not image_path:
            QMessageBox.warning(self, "提示", "图像路径不能为空，请输入正确的图像路径！")
            return
        image_size = (500, 500)  # 这里替换为实际期望的图像尺寸
        q_image, encoded_bases, decoded_bases, encode_time, decode_time = process_image_wrapper(image_path, image_size)

        print("Encoded Bases Sequence:", encoded_bases)
        print(f"编码时间: {encode_time:.6f} 秒")
        print(f"解码时间: {decode_time:.6f} 秒")
        
        # 在文本框中添加时间信息
        time_info = f"编码时间: {encode_time:.6f} 秒\n解码时间: {decode_time:.6f} 秒\n\n"
        self.plain_text_edit.setPlainText(time_info + encoded_bases)
        
        # 显示时间信息的消息框
        QMessageBox.information(self, "处理时间统计", 
                               f"编码时间: {encode_time:.6f} 秒\n" 
                               f"解码时间: {decode_time:.6f} 秒\n" 
                               f"总时间: {(encode_time + decode_time):.6f} 秒")

        # 获取label_box2_2的当前尺寸，用于等比例缩放图片
        label_width = self.label_box2_2.width()
        label_height = self.label_box2_2.height()

        # 获取图片原始尺寸（从返回的QImage获取宽高信息）
        pixmap_width = q_image.width()
        pixmap_height = q_image.height()

        # 计算宽高比例，以决定按哪个方向进行等比例缩放（取较小的缩放比例）
        width_ratio = label_width / pixmap_width
        height_ratio = label_height / pixmap_height
        scale_ratio = min(width_ratio, height_ratio)

        # 根据缩放比例计算缩放后的尺寸
        scaled_width = int(pixmap_width * scale_ratio)
        scaled_height = int(pixmap_height * scale_ratio)

        # 缩放图片
        scaled_q_image = q_image.scaled(scaled_width, scaled_height, aspectRatioMode=1)
        pixmap = QPixmap.fromImage(scaled_q_image)

        self.label_box2_2.setAlignment(Qt.AlignCenter)
        self.label_box2_2.setPixmap(pixmap)

    def save_text_to_file(self):
        # 获取QPlainTextEdit中的文本内容
        text_content = self.plain_text_edit.toPlainText()
        if not text_content:
            QMessageBox.warning(self, "提示", "文本内容不能为空，请先确保有内容可保存！")
            return

        # 尝试从图片路径相关部件获取文件名（这里假设text_edit_label1中存放着图片路径，可按实际情况调整）
        file_path_in_label = self.text_edit_label1.text().strip()
        if not file_path_in_label:
            QMessageBox.warning(self, "提示", "无法获取图片路径信息，无法生成文件名，请先正确选择图片！")
            return
        file_name = os.path.basename(file_path_in_label)
        file_name = os.path.splitext(file_name)[0]  # 去除文件扩展名，只保留文件名部分

        # 定义文件夹名称
        folder_name = "Chain-DNA-Sequance"
        # 检查文件夹是否存在，不存在则创建
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # 拼接完整的文件路径，按照文件名 +.txt的格式
        file_path = os.path.join(folder_name, file_name + ".txt")

        try:
            # 打开文件，使用'w'模式表示写入，如果文件不存在则创建，如果存在则覆盖
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text_content)
            print(f"文本已成功保存到 {file_path}")

            # 弹出消息框提示用户文件保存成功
            QMessageBox.information(self, "保存成功", f"文本已成功保存到 {file_path}")

            # 将保存的文件路径赋值给text_edit_label2，方便用户知晓保存位置
            self.text_edit_label2.setText(file_path)

            # 新增功能：读取保存的文件内容并计算GC含量占比
            gc_content = self.calculate_gc_content(file_path)
            gc_content = round(gc_content, 2)
            print(f"该文件的GC含量占比为：{gc_content}%")
            self.inner_label4.setText(f"GC's Content: {gc_content}%")
            QMessageBox.information(self, "GC含量统计", f"该文件的GC含量占比为：{gc_content}%")
        except Exception as e:
            print(f"保存文件时出现错误: {e}")

    def calculate_gc_content(self, file_path):
        gc_count = 0
        total_count = 0
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                for char in line:
                    if char in "GCgc":
                        gc_count += 1
                    total_count += 1
        if total_count == 0:
            return 0
        return (gc_count / total_count) * 100

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Window()
    sys.exit(app.exec_())
