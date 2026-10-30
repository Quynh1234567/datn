"""Golden Model tiền xử lý số nguyên trên ESP32-S3 – đặc tả A1, mục 2.2–2.3.

Khớp từng bit với firmware/esp32_preproc/main/preproc.c:
    Gauss 5×5, σ = 1, k = [56, 250, 412, 250, 56] / 1024, tách được,
    giữ nguyên tổng 2D rồi làm tròn MỘT lần: G = (Σ k_i·k_j·I + 2¹⁹) >> 20  (0..255)
    biên "reflect101" (mặc định cv2.GaussianBlur) hoặc "replicate"
Khung tối: Σ G < 15 × 19 200 trên ảnh đã làm mượt (giống features/pipeline.py).

Không phụ thuộc gói ocsvm_fpga – nạp trực tiếp bằng đường dẫn cũng được.
"""
from __future__ import annotations

import zlib

import numpy as np

W, H = 160, 120
K = np.array([56, 250, 412, 250, 56], np.int64)
SHIFT = 20
BORDERS = {"reflect101": "reflect", "replicate": "edge"}   # tên → chế độ np.pad
BORDER_ID = {"reflect101": 0, "replicate": 1}
PATTERN_NAMES = ["zeros", "ones255", "hramp", "vramp", "checker1", "noise12345", "noise777", "impulses",
                 "blocks8", "xor"]


def gauss5_q(img: np.ndarray, border: str = "reflect101") -> np.ndarray:
    """Ảnh uint8 (H, W) → ảnh làm mượt uint8, đúng từng bit như preproc.c."""
    if img.dtype != np.uint8:
        raise TypeError("ảnh đầu vào phải là uint8")
    h, w = img.shape
    p = np.pad(img.astype(np.int64), 2, mode=BORDERS[border])
    t = sum(K[i] * p[:, i:i + w] for i in range(5))
    s = sum(K[j] * t[j:j + h, :] for j in range(5))
    return ((s + (1 << (SHIFT - 1))) >> SHIFT).astype(np.uint8)


def frame_sum(smoothed: np.ndarray) -> int:
    return int(smoothed.astype(np.int64).sum())


def is_dark(smoothed: np.ndarray, mean_thr: int = 15) -> bool:
    return frame_sum(smoothed) < mean_thr * smoothed.size


def crc32(buf) -> int:
    return zlib.crc32(np.ascontiguousarray(buf, np.uint8).tobytes()) & 0xFFFFFFFF


def test_pattern(pid: int) -> np.ndarray:
    """Sinh lại y hệt pp_test_pattern() trong preproc.c."""
    y, x = np.indices((H, W))
    img = np.zeros((H, W), np.uint8)
    if pid == 1:
        img[:] = 255
    elif pid == 2:
        img = (x * 255 // (W - 1)).astype(np.uint8)
    elif pid == 3:
        img = (y * 255 // (H - 1)).astype(np.uint8)
    elif pid == 4:
        img = (((x + y) & 1) * 255).astype(np.uint8)
    elif pid in (5, 6):
        s = 12345 if pid == 5 else 777
        out = np.empty(H * W, np.uint8)
        for i in range(H * W):
            s = (s * 1103515245 + 12345) & 0xFFFFFFFF
            out[i] = s >> 24
        img = out.reshape(H, W)
    elif pid == 7:
        for r, c in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1), (1, 1), (H - 2, W - 2), (H // 2, W // 2),
                     (0, 80), (60, 0)]:
            img[r, c] = 255
    elif pid == 8:
        img = np.where(((x // 8) + (y // 8)) & 1, 200, 30).astype(np.uint8)
    elif pid == 9:
        img = ((x ^ y) & 0xFF).astype(np.uint8)
    return img


def expected_crcs() -> dict:
    """{(tên mẫu, biên): (crc ảnh làm mượt, tổng)} – nhúng vào firmware để tự kiểm tra trên chip."""
    out = {}
    for pid, name in enumerate(PATTERN_NAMES):
        img = test_pattern(pid)
        for b in BORDERS:
            g = gauss5_q(img, b)
            out[(name, b)] = (crc32(g), frame_sum(g))
    return out


def vectors_header() -> str:
    """Nội dung main/selftest_vectors.h."""
    L = ["/* SINH TỰ ĐỘNG bởi src/ocsvm_fpga/hwmodel/preproc_q.py – vectors_header(). KHÔNG SỬA TAY. */",
         "#pragma once", "#include <stdint.h>", "",
         "typedef struct {", "    uint8_t pattern;", "    uint8_t border;   /* 0 = reflect101, 1 = replicate */",
         "    uint32_t crc;     /* CRC-32 ảnh đã làm mượt */", "    uint32_t sum;     /* Σ ảnh đã làm mượt */",
         "} pp_vector_t;", "", "static const pp_vector_t PP_VECTORS[] = {"]
    for (name, b), (crc, s) in expected_crcs().items():
        L.append(f"    {{{PATTERN_NAMES.index(name)}, {BORDER_ID[b]}, 0x{crc:08X}u, {s}u}},   /* {name}, {b} */")
    L += ["};", "#define PP_N_VECTORS (sizeof(PP_VECTORS) / sizeof(PP_VECTORS[0]))", ""]
    return "\n".join(L)
