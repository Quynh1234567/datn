"""Golden Model số nguyên của bộ trích đặc trưng trên FPGA – đặc tả A1, mục 3–8.

    ảnh I1, I2 (uint8, đã Gauss trên ESP32)
      → tầng 1  đạo hàm 4·Ix, 4·Iy, 4·It           (11 bit, chính xác tuyệt đối)
      → tầng 2  a = 4Ix / D16, b = 4Iy / D16          (15 bit, 18 bit lẻ; tính 1 lần / cặp khung)
      → tầng 3  64 vòng Jacobi, u, v Q6.6 (13 bit)    (ū = (2·cạnh + góc)·10923 >> 17)
      → tầng 4  |v| = isqrt(u² + v²), trung bình 16 ô (×6991 >> 20 → Q5.10)
      → tầng 5  nén log c·sign·ln(1 + |x|/c) bằng bảng tra 2 vùng (Q5.10)
      → tầng 6  z = (g − mean)·inv_std (Q5.10)        → đầu vào OC-SVM (giữ nguyên đặc tả cũ)

Mọi phép tính là số nguyên, dịch phải là dịch số học (floor) giống `>>>` của Verilog.
Kết quả của mỗi hàm ở đây là CHUẨN mà RTL phải khớp từng bit.

Độ rộng u, v (wu, fu) và cách tính |v| được tham số hóa để quét ở A3; mặc định = đặc tả.
Không phụ thuộc gói ocsvm_fpga (chỉ numpy) để dùng được ở mọi nơi.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SPEC_VERSION = "0.4"   # phải trùng claude/hs_fixed_point_spec.md và scripts/20_hs_golden.py

H, W = 120, 160
GRID_R, GRID_C = 4, 4
CELL_H, CELL_W = H // GRID_R, W // GRID_C          # 30 × 40 = 1 200 điểm
CELL_N = CELL_H * CELL_W
DIM = GRID_R * GRID_C * 3                          # 48


# ------------------------------------------------------------------------- tham số

@dataclass(frozen=True)
class HSParams:
    alpha2_16: int = 64          # 16·α², α = 2
    n_iter: int = 64
    coef_frac: int = 18          # a, b: 18 bit lẻ
    coef_w: int = 15             # a, b: 15 bit có dấu
    wu: int = 13                 # u, v: tổng số bit (có dấu)
    fu: int = 6                  # u, v: số bit lẻ → Q6.6 (±64 px; bản 0.4 – IITB có |u| tới 52 px)
    avg_mul: int = 10923         # ≈ 2¹⁷/12
    avg_shift: int = 17
    mag_mode: str = "round"      # |v| = round(√·) (mặc định, sai số nhỏ hơn) | "floor"
    data_frac: int = 10          # đặc trưng, z: Q5.10
    data_w: int = 16
    cell_shift: int = 20
    log_c: float = 0.5           # grid_log_scale
    log_a_bits: int = 10         # vùng A: 1024 phần tử, |x_q| < 2048, bước 2 LSB
    log_a_step_sh: int = 1
    log_b_step_sh: int = 6       # vùng B: bước 64 LSB = 1/16
    log_out_frac: int = 10
    log_out_w: int = 12          # Q1.10 không dấu
    inv_w: int = 16              # 1/σ không dấu, số bit lẻ tự chọn

    @property
    def cell_mul(self) -> int:   # Σ(Q·fu) · cell_mul >> 20 = trung bình ở Q·data_frac
        return int(round(2 ** (self.cell_shift + self.data_frac - self.fu) / CELL_N))

    @property
    def log_a_limit(self) -> int:     # |x_q| < limit → vùng A
        return (1 << self.log_a_bits) << self.log_a_step_sh

    @property
    def log_b_size(self) -> int:
        return ((1 << (self.data_w - 1)) - self.log_a_limit) >> self.log_b_step_sh


DEFAULT = HSParams()


def rnd(x, s: int):
    """Làm tròn về gần nhất (nửa lên) bằng dịch số học: (x + 2^(s−1)) >> s."""
    return (x + (1 << (s - 1))) >> s


def sat(x, w: int):
    return np.clip(x, -(1 << (w - 1)), (1 << (w - 1)) - 1)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """a[y+dy, x+dx], biên replicate (dy, dx ∈ {0, 1})."""
    p = np.pad(a, ((0, dy), (0, dx)), mode="edge")
    return p[dy:dy + a.shape[0], dx:dx + a.shape[1]]


# ---------------------------------------------------------------- tầng 1–2

def derivatives4(i1: np.ndarray, i2: np.ndarray):
    """4·Ix, 4·Iy, 4·It trên khối 2×2×2 (đúng công thức image_derivatives của repo, ×4)."""
    a = i1.astype(np.int64)
    b = i2.astype(np.int64)
    a00, a01, a10, a11 = a, _shift(a, 0, 1), _shift(a, 1, 0), _shift(a, 1, 1)
    b00, b01, b10, b11 = b, _shift(b, 0, 1), _shift(b, 1, 0), _shift(b, 1, 1)
    ix = (a01 - a00) + (a11 - a10) + (b01 - b00) + (b11 - b10)
    iy = (a10 - a00) + (a11 - a01) + (b10 - b00) + (b11 - b01)
    it = (b00 - a00) + (b01 - a01) + (b10 - a10) + (b11 - a11)
    return ix, iy, it


def coeffs(ix4, iy4, p: HSParams = DEFAULT):
    """D16 = 64 + Ix4² + Iy4²; a = Ix4/D16, b = Iy4/D16 ở 18 bit lẻ (chia trên trị tuyệt đối, làm tròn)."""
    d16 = p.alpha2_16 + ix4 * ix4 + iy4 * iy4

    def div(n):
        q = ((np.abs(n) << p.coef_frac) + (d16 >> 1)) // d16
        return sat(np.sign(n) * q, p.coef_w)

    return d16, div(ix4), div(iy4)


# ------------------------------------------------------------------- tầng 3

def avg12(q: np.ndarray, p: HSParams = DEFAULT) -> np.ndarray:
    """ū ≈ (2·Σcạnh + Σgóc)/12, biên replicate."""
    z = np.pad(q, 1, mode="edge")
    e = z[:-2, 1:-1] + z[2:, 1:-1] + z[1:-1, :-2] + z[1:-1, 2:]
    c = z[:-2, :-2] + z[:-2, 2:] + z[2:, :-2] + z[2:, 2:]
    return rnd((2 * e + c) * p.avg_mul, p.avg_shift)


@dataclass
class HSTrace:
    ix4: np.ndarray
    iy4: np.ndarray
    it4: np.ndarray
    d16: np.ndarray
    a: np.ndarray
    b: np.ndarray
    u: np.ndarray                      # Q·fu (Q6.6) sau n_iter vòng
    v: np.ndarray
    u1: np.ndarray | None = None       # sau vòng 1 (gỡ lỗi RTL)
    v1: np.ndarray | None = None
    n_sat: int = 0                     # số lần u/v bị bão hòa (cộng mọi vòng)
    max_abs_n: int = 0                 # |N| lớn nhất (kiểm tra 25 bit)


def horn_schunck_q(i1: np.ndarray, i2: np.ndarray, p: HSParams = DEFAULT, keep_iter1: bool = False) -> HSTrace:
    """Bản nhanh (bộ đệm cấp phát một lần, u và v xử lý chung, số 32 bit khi không thể tràn).
    Kết quả GIỐNG TỪNG BIT horn_schunck_q_ref – kiểm tra trong scripts/20_hs_golden.py --selftest."""
    ix, iy, it = derivatives4(i1, i2)
    d16, a, b = coeffs(ix, iy, p)
    # int32 an toàn khi (2·cạnh + góc)·AVG_MUL < 2³¹  ⇔  12·2^(wu−1)·AVG_MUL < 2³¹  (wu ≤ 15)
    dt = np.int32 if 12 * (1 << (p.wu - 1)) * p.avg_mul + (1 << p.avg_shift) < (1 << 31) else np.int64
    h, w = ix.shape
    uv = np.zeros((2, h, w), dt)
    pad = np.zeros((2, h + 2, w + 2), dt)
    ixd, iyd = ix.astype(dt), iy.astype(dt)
    itd = (it << p.fu).astype(dt)
    ab = np.stack([a, b]).astype(np.int64)
    lo, hi = -(1 << (p.wu - 1)), (1 << (p.wu - 1)) - 1
    half_avg, half_c = dt(1 << (p.avg_shift - 1)), np.int64(1 << (p.coef_frac - 1))
    tr = HSTrace(ix, iy, it, d16, a, b, None, None)
    max_n, n_sat = 0, 0
    for k in range(p.n_iter):
        pad[:, 1:-1, 1:-1] = uv                     # biên replicate
        pad[:, 0, 1:-1] = uv[:, 0]
        pad[:, -1, 1:-1] = uv[:, -1]
        pad[:, :, 0] = pad[:, :, 1]
        pad[:, :, -1] = pad[:, :, -2]
        e = pad[:, :-2, 1:-1] + pad[:, 2:, 1:-1] + pad[:, 1:-1, :-2] + pad[:, 1:-1, 2:]
        c = pad[:, :-2, :-2] + pad[:, :-2, 2:] + pad[:, 2:, :-2] + pad[:, 2:, 2:]
        avg = ((2 * e + c) * dt(p.avg_mul) + half_avg) >> p.avg_shift
        n = ixd * avg[0] + iyd * avg[1] + itd
        max_n = max(max_n, int(n.max()), -int(n.min()))
        new = avg - ((ab * n.astype(np.int64) + half_c) >> p.coef_frac)
        bad = (new < lo) | (new > hi)
        if bad.any():
            n_sat += int(bad.sum())
            np.clip(new, lo, hi, out=new)
        uv = new.astype(dt)
        if k == 0 and keep_iter1:
            tr.u1, tr.v1 = uv[0].astype(np.int64), uv[1].astype(np.int64)
    tr.u, tr.v = uv[0].astype(np.int64), uv[1].astype(np.int64)
    tr.n_sat, tr.max_abs_n = n_sat, max_n
    return tr


def horn_schunck_q_ref(i1: np.ndarray, i2: np.ndarray, p: HSParams = DEFAULT, keep_iter1: bool = False) -> HSTrace:
    """Bản tham chiếu viết thẳng theo đặc tả (chậm hơn) – dùng để kiểm tra bản nhanh."""
    ix, iy, it = derivatives4(i1, i2)
    d16, a, b = coeffs(ix, iy, p)
    u = np.zeros_like(ix)
    v = np.zeros_like(ix)
    lo, hi = -(1 << (p.wu - 1)), (1 << (p.wu - 1)) - 1
    tr = HSTrace(ix, iy, it, d16, a, b, u, v)
    for k in range(p.n_iter):
        ua, va = avg12(u, p), avg12(v, p)             # Jacobi: chỉ dùng u, v của vòng trước
        n = ix * ua + iy * va + (it << p.fu)
        tr.max_abs_n = max(tr.max_abs_n, int(np.abs(n).max()))
        un = ua - rnd(a * n, p.coef_frac)
        vn = va - rnd(b * n, p.coef_frac)
        tr.n_sat += int(np.sum((un < lo) | (un > hi)) + np.sum((vn < lo) | (vn > hi)))
        u, v = np.clip(un, lo, hi), np.clip(vn, lo, hi)
        if k == 0 and keep_iter1:
            tr.u1, tr.v1 = u.copy(), v.copy()
    tr.u, tr.v = u, v
    return tr


# ------------------------------------------------------------------- tầng 4

def isqrt(x: np.ndarray) -> np.ndarray:
    """floor(√x) chính xác cho số nguyên không âm (≤ 2⁵²)."""
    r = np.floor(np.sqrt(x.astype(np.float64))).astype(np.int64)
    r -= (r * r > x)                       # sửa sai số làm tròn của float
    r += ((r + 1) * (r + 1) <= x)
    return r


def magnitude(u: np.ndarray, v: np.ndarray, p: HSParams = DEFAULT) -> np.ndarray:
    s = u * u + v * v
    if p.mag_mode == "round":
        return (isqrt(4 * s) + 1) >> 1     # round(√s)
    return isqrt(s)


def cell_sums(u, v, m):
    """Σu, Σv, Σ|v| của 16 ô, shape (16, 3), thứ tự ô theo hàng (row-major)."""
    out = np.empty((GRID_R * GRID_C, 3), np.int64)
    k = 0
    for r in range(GRID_R):
        for c in range(GRID_C):
            sl = (slice(r * CELL_H, (r + 1) * CELL_H), slice(c * CELL_W, (c + 1) * CELL_W))
            out[k] = (u[sl].sum(), v[sl].sum(), m[sl].sum())
            k += 1
    return out


def cell_means_q(sums: np.ndarray, p: HSParams = DEFAULT) -> np.ndarray:
    """(16, 3) tổng Q·fu → 48 trung bình Q5.10: [u₀₀, v₀₀, |v|₀₀, u₀₁, …] (giống grid_flow_descriptor)."""
    return sat(rnd(sums * p.cell_mul, p.cell_shift), p.data_w).reshape(-1)


# ------------------------------------------------------------------- tầng 5

def compress_float(x, c: float = 0.5):
    return c * np.sign(x) * np.log1p(np.abs(x) / c)


def log_lut(p: HSParams = DEFAULT) -> np.ndarray:
    """Bảng tra g(|x|) không dấu Q1.10, 2 vùng; mỗi phần tử = g tại điểm giữa khoảng."""
    one = 1 << p.data_frac
    sa, sb = 1 << p.log_a_step_sh, 1 << p.log_b_step_sh
    ca = (np.arange(1 << p.log_a_bits) * sa + (sa - 1) / 2) / one
    cb = (p.log_a_limit + np.arange(p.log_b_size) * sb + (sb - 1) / 2) / one
    vals = compress_float(np.concatenate([ca, cb]), p.log_c)
    q = np.round(vals * (1 << p.log_out_frac)).astype(np.int64)
    if q.max() >= 1 << p.log_out_w:
        raise ValueError("bảng log tràn độ rộng đầu ra")
    return q


def log_compress_q(x_q: np.ndarray, lut: np.ndarray, p: HSParams = DEFAULT) -> np.ndarray:
    ax = np.abs(x_q)
    addr = np.where(ax < p.log_a_limit, ax >> p.log_a_step_sh,
                    (1 << p.log_a_bits) + ((ax - p.log_a_limit) >> p.log_b_step_sh))
    addr = np.minimum(addr, len(lut) - 1)
    return np.sign(x_q) * lut[addr]


# ------------------------------------------------------------------- tầng 6

@dataclass
class ScalerQ:
    mean_q: np.ndarray            # Q5.10 có dấu, 16 bit
    inv_q: np.ndarray             # 1/σ không dấu, inv_w bit, inv_frac bit lẻ
    inv_frac: int
    n_sat_mean: int = 0


def quantize_scaler(mean: np.ndarray, scale: np.ndarray, p: HSParams = DEFAULT) -> ScalerQ:
    """mean, scale của Standardizer (trên đặc trưng ĐÃ nén log) → số nguyên.
    Số bit lẻ của 1/σ được chọn lớn nhất mà 1/σ lớn nhất vẫn vừa inv_w bit."""
    inv = 1.0 / np.asarray(scale, np.float64)
    frac = int(np.floor(np.log2(((1 << p.inv_w) - 1) / inv.max())))
    frac = max(0, min(frac, 2 * p.inv_w))
    mq_raw = np.round(np.asarray(mean) * (1 << p.data_frac)).astype(np.int64)
    mq = sat(mq_raw, p.data_w)
    return ScalerQ(mq, np.round(inv * (1 << frac)).astype(np.int64), frac, int(np.sum(mq != mq_raw)))


def standardize_q(g_q: np.ndarray, sc: ScalerQ, p: HSParams = DEFAULT) -> np.ndarray:
    return sat(rnd((g_q - sc.mean_q) * sc.inv_q, sc.inv_frac), p.data_w)


# ------------------------------------------------------------------ toàn bộ

@dataclass
class FeatureTrace:
    hs: HSTrace
    mag: np.ndarray
    sums: np.ndarray       # (16, 3)
    mean_q: np.ndarray     # 48, Q5.10 – trung bình ô
    log_q: np.ndarray      # 48, Q5.10 – sau nén log
    z_q: np.ndarray | None = None   # 48, Q5.10 – sau chuẩn hóa = đầu vào OC-SVM
    extra: dict = field(default_factory=dict)


def features_q(i1, i2, p: HSParams = DEFAULT, scaler: ScalerQ | None = None, lut=None,
               keep_iter1: bool = False) -> FeatureTrace:
    hs = horn_schunck_q(i1, i2, p, keep_iter1)
    m = magnitude(hs.u, hs.v, p)
    s = cell_sums(hs.u, hs.v, m)
    mq = cell_means_q(s, p)
    lq = log_compress_q(mq, log_lut(p) if lut is None else lut, p)
    zq = standardize_q(lq, scaler, p) if scaler is not None else None
    return FeatureTrace(hs, m, s, mq, lq, zq)


# ---------------------------------------------------- tham chiếu số thực (pipeline train)

_K = np.array([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], np.float32)


def horn_schunck_float(i1, i2, alpha: float = 2.0, n_iter: int = 64):
    """Đúng công thức features/optical_flow.py (float32, biên replicate)."""
    import cv2
    a, b = i1.astype(np.float32), i2.astype(np.float32)
    a00, a01, a10, a11 = a, _shift(a, 0, 1), _shift(a, 1, 0), _shift(a, 1, 1)
    b00, b01, b10, b11 = b, _shift(b, 0, 1), _shift(b, 1, 0), _shift(b, 1, 1)
    ix = 0.25 * ((a01 - a00) + (a11 - a10) + (b01 - b00) + (b11 - b10))
    iy = 0.25 * ((a10 - a00) + (a11 - a01) + (b10 - b00) + (b11 - b01))
    it = 0.25 * ((b00 - a00) + (b01 - a01) + (b10 - a10) + (b11 - a11))
    u = np.zeros_like(a)
    v = np.zeros_like(a)
    den = (alpha * alpha + ix * ix + iy * iy).astype(np.float32)
    for _ in range(n_iter):
        ua = cv2.filter2D(u, -1, _K, borderType=cv2.BORDER_REPLICATE)
        va = cv2.filter2D(v, -1, _K, borderType=cv2.BORDER_REPLICATE)
        d = (ix * ua + iy * va + it) / den
        u, v = ua - ix * d, va - iy * d
    return u, v


def features_float(i1, i2, log_c: float = 0.5):
    """Pipeline train: HS số thực → lưới 4×4 [mean u, mean v, mean |v|] → nén log (chưa chuẩn hóa)."""
    u, v = horn_schunck_float(i1, i2)
    m = np.hypot(u, v)
    out = []
    for r in range(GRID_R):
        for c in range(GRID_C):
            sl = (slice(r * CELL_H, (r + 1) * CELL_H), slice(c * CELL_W, (c + 1) * CELL_W))
            out += [u[sl].mean(), v[sl].mean(), m[sl].mean()]
    return compress_float(np.asarray(out, np.float64), log_c), (u, v)
