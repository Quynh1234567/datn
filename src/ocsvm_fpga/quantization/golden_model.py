"""Golden Model bit-accurate của bộ tăng tốc OC-SVM trên FPGA.

Mô tả CHÍNH XÁC từng bit datapath mà RTL Verilog phải hiện thực, để testbench
so sánh trực tiếp kết quả mô phỏng với file expected_*.mem:

  ┌──────────── cho mỗi Support Vector i (PE song song / tuần tự) ────────────┐
  │ diff_j   = x_q[j] − sv_q[i][j]                (DATA_W+1 bit, Q·FRAC_X)     │
  │ dist     = Σ_j diff_j²                        (Q·2FRAC_X)  ← khối MAC      │
  │ t        = dist · γ_q                         (Q·(2FRAC_X+FRAC_G))         │
  │ idx      = t >> LUT_SHIFT                     (địa chỉ LUT)                │
  │ k_i      = idx < LUT_SIZE ? EXP_LUT[idx] : 0  (Q0.LUT_W ≈ exp(−γ‖x−s‖²))   │
  │ acc     += α_q[i] · k_i                       (Q·(FRAC_A+LUT_W))           │
  └────────────────────────────────────────────────────────────────────────────┘
  f_q = acc − ρ_q
  MODE 0 (OC-SVM)   : anomaly = (f_q <  THR_q)
  MODE 1 (LS-OCSVM) : anomaly = (|f_q| > THR_q)

LUT lưu exp(−t) tại điểm giữa mỗi ô (i + 0.5)·Δt → sai số lượng tử đối xứng.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ..models.base import AnomalyDetector
from .fixed_point import from_fixed, qrange, to_fixed


@dataclass
class QuantParams:
    data_width: int
    data_frac: int
    coef_width: int
    coef_frac: int
    gamma_width: int
    gamma_frac: int
    lut_addr_bits: int
    lut_t_max: int
    lut_width: int
    acc_width: int
    lut_shift: int
    coef_scale: float        # hệ số nhân α, ρ (max|α_scaled| = 1)
    mode: int                # 0 = lower (OC-SVM), 1 = abs (LS-OC-SVM)

    @property
    def acc_frac(self) -> int:
        return self.coef_frac + self.lut_width

    @property
    def lut_size(self) -> int:
        return 1 << self.lut_addr_bits

    @property
    def lut_frac(self) -> int:
        return self.lut_addr_bits - int(math.log2(self.lut_t_max))


class FixedPointOCSVM:
    """Mô hình lượng tử hóa + suy luận số nguyên bit-accurate."""

    def __init__(self, det: AnomalyDetector, fcfg):
        km = det.model
        self.det = det
        t_max = int(fcfg.exp_lut_t_max)
        if t_max & (t_max - 1):
            raise ValueError("exp_lut_t_max phải là lũy thừa của 2")
        gamma_frac = int(math.floor(math.log2(((1 << fcfg.gamma_width) - 1) / km.gamma)))
        lut_frac = fcfg.exp_lut_addr_bits - int(math.log2(t_max))
        lut_shift = 2 * fcfg.data_frac + gamma_frac - lut_frac
        if lut_shift < 0:
            raise ValueError("LUT_SHIFT âm – giảm exp_lut_addr_bits hoặc tăng data_frac")
        coef_scale = 1.0 / float(np.max(np.abs(km.coefs)))
        self.p = QuantParams(
            data_width=fcfg.data_width, data_frac=fcfg.data_frac,
            coef_width=fcfg.coef_width, coef_frac=fcfg.coef_frac,
            gamma_width=fcfg.gamma_width, gamma_frac=gamma_frac,
            lut_addr_bits=fcfg.exp_lut_addr_bits, lut_t_max=t_max, lut_width=fcfg.exp_lut_width,
            acc_width=fcfg.acc_width, lut_shift=lut_shift, coef_scale=coef_scale,
            mode=1 if km.mode == "abs" else 0,
        )
        p = self.p
        self.sv_q, self.sv_sat = to_fixed(km.support_vectors, p.data_width, p.data_frac, True, True)
        self.coef_q = to_fixed(km.coefs * coef_scale, p.coef_width, p.coef_frac, True)
        self.gamma_q = int(round(km.gamma * (1 << gamma_frac)))
        self.rho_q = int(to_fixed(km.rho * coef_scale, p.acc_width, p.acc_frac, True))
        thr_f = km.decision_threshold_from_score(det.threshold) * coef_scale
        self.thr_q = int(to_fixed(thr_f, p.acc_width, p.acc_frac, True))
        centers = (np.arange(p.lut_size) + 0.5) / (1 << lut_frac)
        lo, hi = qrange(p.lut_width, signed=False)
        self.exp_lut = np.clip(np.round(np.exp(-centers) * (1 << p.lut_width)), lo, hi).astype(np.int64)

    # ------------------------------------------------------------------ đầu vào
    def quantize_input(self, x_raw: np.ndarray) -> tuple[np.ndarray, int]:
        """Đặc trưng thô (log-cov) → chuẩn hóa → Q(DATA). Trả về (x_q, số phần tử bão hòa)."""
        z = self.det.scaler.transform(np.atleast_2d(x_raw))
        return to_fixed(z, self.p.data_width, self.p.data_frac, True, True)

    # ------------------------------------------------------------------ datapath
    def kernel_q(self, x_q: np.ndarray, chunk: int = 512) -> np.ndarray:
        """k_q[n, i] – giá trị LUT exp cho từng mẫu n và SV i (số nguyên)."""
        x_q = np.atleast_2d(x_q).astype(np.int64)
        out = np.empty((len(x_q), len(self.sv_q)), np.int64)
        for a in range(0, len(x_q), chunk):
            diff = x_q[a:a + chunk, None, :] - self.sv_q[None, :, :]
            dist = np.einsum("nsd,nsd->ns", diff, diff)
            t = dist * self.gamma_q
            idx = t >> self.p.lut_shift
            out[a:a + chunk] = np.where(idx < self.p.lut_size,
                                        self.exp_lut[np.minimum(idx, self.p.lut_size - 1)], 0)
        return out

    def decision_q(self, x_q: np.ndarray) -> np.ndarray:
        acc = self.kernel_q(x_q) @ self.coef_q
        return acc - self.rho_q

    def anomaly_q(self, f_q: np.ndarray) -> np.ndarray:
        return (np.abs(f_q) > self.thr_q if self.p.mode == 1 else f_q < self.thr_q).astype(np.uint8)

    def decision_float_equiv(self, f_q: np.ndarray) -> np.ndarray:
        """f_q → giá trị f(x) tương đương mô hình float (để so sánh sai số)."""
        return from_fixed(f_q, self.p.acc_frac) / self.p.coef_scale

    # ------------------------------------------------------------------ báo cáo
    def internal_widths(self, d: int) -> dict:
        p = self.p
        diff_w = p.data_width + 1
        dist_w = 2 * diff_w + int(math.ceil(math.log2(max(d, 2))))
        t_w = dist_w + p.gamma_width
        prod_w = p.coef_width + p.lut_width + 1
        acc_need = prod_w + int(math.ceil(math.log2(max(len(self.coef_q), 2))))
        return {"DIFF_W": diff_w, "DIST_W": dist_w, "T_W": t_w, "PROD_W": prod_w,
                "ACC_W_REQUIRED": acc_need, "ACC_W": p.acc_width}

    def check_ranges(self) -> list[str]:
        warn = []
        lo, hi = qrange(self.p.acc_width)
        if not (lo <= self.rho_q <= hi and lo <= self.thr_q <= hi):
            warn.append("ρ hoặc ngưỡng bị bão hòa trong ACC_W")
        need = self.internal_widths(self.sv_q.shape[1])["ACC_W_REQUIRED"]
        if need > self.p.acc_width:
            warn.append(f"ACC_W={self.p.acc_width} < {need} bit cần thiết (trường hợp xấu nhất)")
        if self.sv_sat:
            warn.append(f"{self.sv_sat} phần tử SV bị bão hòa – tăng số bit phần nguyên")
        return warn

    def summary(self) -> dict:
        d = asdict(self.p)
        d.update(gamma_q=self.gamma_q, rho_q=self.rho_q, thr_q=self.thr_q,
                 n_sv=int(len(self.coef_q)), dim=int(self.sv_q.shape[1]),
                 widths=self.internal_widths(self.sv_q.shape[1]))
        return d
