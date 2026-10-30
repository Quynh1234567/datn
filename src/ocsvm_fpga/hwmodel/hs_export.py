"""Xuất tham số và véc-tơ kiểm thử TỪNG TẦNG của khối trích đặc trưng cho testbench RTL.

outputs/hw_vectors/
  hs_params.vh                 localparam cho RTL (độ rộng bit, hằng số nhân/dịch, bảng log)
  mem/log_lut.mem              bảng nén log, LOG_LUT_SIZE dòng × LOG_OUT_W bit
  mem/scaler_mean_q.mem        48 dòng, Q5.10 có dấu 16 bit           (nếu có mô hình)
  mem/scaler_inv_q.mem         48 dòng, 1/σ không dấu INV_W bit, INV_FRAC bit lẻ
  case_XX/
    img1.mem, img2.mem         19 200 dòng × 8 bit (khung trước / sau, đã Gauss)
    deriv.mem                  19 200 dòng × 64 bit, gói giống SDRAM (đặc tả §4.4):
                                 [10:0] Ix4 · [21:11] Iy4 · [32:22] It4 · [47:33] a · [62:48] b
    uv_iter1.mem               19 200 dòng × 26 bit: [12:0] u · [25:13] v  sau vòng 1
    uv_final.mem               như trên, sau N_ITER vòng
    mag.mem                    19 200 dòng × MAG_W bit, |v| từng điểm
    cell_sums.mem              48 dòng × SUM_W bit: Σu, Σv, Σ|v| của ô 0, ô 1, … (row-major)
    feat_mean.mem              48 dòng × 16 bit, trung bình ô Q5.10
    feat_log.mem               48 dòng × 16 bit, sau nén log
    feat_z.mem                 48 dòng × 16 bit, sau chuẩn hóa = đầu vào OC-SVM (nếu có mô hình)
    expected.txt               f_q và cờ bất thường của OC-SVM (nếu có mô hình)
  index.md                     tóm tắt các ca

Mọi file .mem: số hex, mỗi dòng một giá trị, số âm ở dạng bù hai trong đúng độ rộng
→ đọc bằng $readmemh. Điểm ảnh theo hàng (row-major), y = 0…119, x = 0…159.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from . import hs_q as hq
except ImportError:          # nạp trực tiếp bằng đường dẫn (không qua gói)
    import hs_q as hq  # type: ignore

D4_W, D16_W, N_W, MAG_W, SUM_W = 11, 21, 25, 13, 24


def to_hex_lines(values, width: int) -> list[str]:
    mask = (1 << width) - 1
    nd = (width + 3) // 4
    return [format(int(v) & mask, f"0{nd}X") for v in np.ravel(values)]


def write_mem(path: Path, values, width: int, comment: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = [f"// {c}" for c in comment.splitlines()] if comment else []
    path.write_text("\n".join(head + to_hex_lines(values, width)) + "\n", encoding="ascii")


def read_mem(path: Path, width: int, signed: bool = True) -> np.ndarray:
    vals = [int(ln.strip(), 16) for ln in path.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("//")]
    a = np.array(vals, np.int64)
    if signed:
        a = np.where(a >= 1 << (width - 1), a - (1 << width), a)
    return a


def pack(fields) -> np.ndarray:
    """fields = [(mảng, độ rộng), …] – trường đầu ở bit thấp nhất."""
    word = np.zeros(np.asarray(fields[0][0]).size, dtype=object)
    shift = 0
    for arr, w in fields:
        word = word + ((np.ravel(arr).astype(np.int64) & ((1 << w) - 1)).astype(object) << shift)
        shift += w
    return word


def params_vh(p: hq.HSParams, scaler: hq.ScalerQ | None) -> str:
    lut_size = (1 << p.log_a_bits) + p.log_b_size
    inv = f"localparam integer INV_W       = {p.inv_w};\nlocalparam integer INV_FRAC    = {scaler.inv_frac};" \
        if scaler is not None else "// (chua co mo hinh: INV_FRAC xuat cung scaler_inv_q.mem)"
    return f"""// ============================================================================
// hs_params.vh - sinh tu dong boi src/ocsvm_fpga/hwmodel/hs_export.py
// Khoi trich dac trung tren FPGA: dao ham -> Horn-Schunck -> luoi 4x4 -> nen log -> z-score
// Dac ta: claude/hs_fixed_point_spec.md ban {getattr(hq, 'SPEC_VERSION', '?')} (muc 3-8). File ASCII.
// ============================================================================
`ifndef HS_PARAMS_VH
`define HS_PARAMS_VH

localparam integer UV_FORMAT_CHECK = {p.wu * 100 + p.fu};   // 1306 = Q6.6 13 bit (ban 0.4)
localparam integer IMG_W       = {hq.W};
localparam integer IMG_H       = {hq.H};
// tang 1-2: dao ham x4 va he so a = Ix4/D16, b = Iy4/D16
localparam integer D4_W        = {D4_W};      // Ix4, Iy4, It4 co dau
localparam integer D16_W       = {D16_W};      // D16 = ALPHA2_16 + Ix4^2 + Iy4^2 khong dau
localparam integer ALPHA2_16   = {p.alpha2_16};      // 16*alpha^2, alpha = 2
localparam integer COEF_W      = {p.coef_w};      // a, b co dau
localparam integer COEF_FRAC   = {p.coef_frac};
// tang 3: Jacobi
localparam integer N_ITER      = {p.n_iter};
localparam integer UV_W        = {p.wu};      // u, v co dau, Q{p.wu - p.fu - 1}.{p.fu}
localparam integer UV_FRAC     = {p.fu};
localparam integer AVG_MUL     = {p.avg_mul};   // u_avg = ((2*canh + goc) * AVG_MUL + 2^(AVG_SHIFT-1)) >>> AVG_SHIFT
localparam integer AVG_SHIFT   = {p.avg_shift};
localparam integer N_W         = {N_W};      // N = Ix4*u_avg + Iy4*v_avg + (It4 << UV_FRAC)
// tang 4: |v| va trung binh o
localparam integer MAG_W       = {MAG_W};      // |v| khong dau
localparam integer MAG_ROUND   = {int(p.mag_mode == 'round')};       // 1: |v| = (isqrt(4*(u^2+v^2)) + 1) >> 1 ; 0: isqrt(u^2+v^2)
localparam integer GRID_R      = {hq.GRID_R};
localparam integer GRID_C      = {hq.GRID_C};
localparam integer CELL_H      = {hq.CELL_H};
localparam integer CELL_W      = {hq.CELL_W};
localparam integer SUM_W       = {SUM_W};      // tong moi o co dau
localparam integer CELL_MUL    = {p.cell_mul};    // mean = (SUM * CELL_MUL + 2^(CELL_SHIFT-1)) >>> CELL_SHIFT  -> Q5.10
localparam integer CELL_SHIFT  = {p.cell_shift};
// tang 5: nen log bang bang tra 2 vung
localparam integer LOG_A_BITS  = {p.log_a_bits};      // vung A: |x| < LOG_A_LIMIT, dia chi = |x| >> LOG_A_STEP_SH
localparam integer LOG_A_STEP_SH = {p.log_a_step_sh};
localparam integer LOG_A_LIMIT = {p.log_a_limit};
localparam integer LOG_B_STEP_SH = {p.log_b_step_sh};    // vung B: dia chi = 2^LOG_A_BITS + ((|x| - LOG_A_LIMIT) >> LOG_B_STEP_SH)
localparam integer LOG_LUT_SIZE = {lut_size};
localparam integer LOG_OUT_W   = {p.log_out_w};      // khong dau Q1.{p.log_out_frac}; ket qua = sign(x) * LUT
// tang 6: z = sat16(((g - MEAN) * INV + 2^(INV_FRAC-1)) >>> INV_FRAC)
localparam integer FEAT_W      = {p.data_w};      // dac trung, z: Q5.{p.data_frac} co dau
localparam integer FEAT_FRAC   = {p.data_frac};
localparam integer DIM         = {hq.DIM};
{inv}

`endif
"""


def export_case(d: Path, i1, i2, tr: hq.FeatureTrace, p: hq.HSParams, ocsvm=None):
    hs = tr.hs
    write_mem(d / "img1.mem", i1, 8, "khung truoc (da Gauss), 19200 diem, row-major")
    write_mem(d / "img2.mem", i2, 8, "khung sau (da Gauss)")
    write_mem(d / "deriv.mem", pack([(hs.ix4, D4_W), (hs.iy4, D4_W), (hs.it4, D4_W), (hs.a, p.coef_w),
                                     (hs.b, p.coef_w), (np.zeros_like(hs.a), 1)]), 64,
              "[10:0] Ix4 [21:11] Iy4 [32:22] It4 [47:33] a [62:48] b [63] 0")
    if hs.u1 is not None:
        write_mem(d / "uv_iter1.mem", pack([(hs.u1, p.wu), (hs.v1, p.wu)]), 2 * p.wu,
                  f"[{p.wu - 1}:0] u  [{2 * p.wu - 1}:{p.wu}] v  sau vong 1")
    write_mem(d / "uv_final.mem", pack([(hs.u, p.wu), (hs.v, p.wu)]), 2 * p.wu,
              f"[{p.wu - 1}:0] u  [{2 * p.wu - 1}:{p.wu}] v  sau {p.n_iter} vong")
    write_mem(d / "mag.mem", tr.mag, MAG_W, "|v| tung diem")
    write_mem(d / "cell_sums.mem", tr.sums.reshape(-1), SUM_W, "Su, Sv, S|v| cua o 0, o 1, ... (row-major)")
    write_mem(d / "feat_mean.mem", tr.mean_q, p.data_w, "trung binh o Q5.10")
    write_mem(d / "feat_log.mem", tr.log_q, p.data_w, "sau nen log Q5.10")
    info = {}
    if tr.z_q is not None:
        write_mem(d / "feat_z.mem", tr.z_q, p.data_w, "sau chuan hoa Q5.10 = dau vao OC-SVM")
        if ocsvm is not None:
            fq, flag = ocsvm(tr.z_q)
            (d / "expected.txt").write_text(f"f_q {fq}\nanomaly {flag}\n", encoding="ascii")
            info.update(f_q=fq, anomaly=flag)
    return info


def export_all(pairs, out_dir: Path, p: hq.HSParams = hq.DEFAULT, scaler: hq.ScalerQ | None = None,
               ocsvm=None, names=None) -> list[dict]:
    """pairs: list (i1, i2) uint8 đã Gauss. ocsvm(z_q) → (f_q, cờ) nếu có mô hình."""
    out_dir = Path(out_dir)
    lut = hq.log_lut(p)
    (out_dir / "mem").mkdir(parents=True, exist_ok=True)
    (out_dir / "hs_params.vh").write_text(params_vh(p, scaler), encoding="ascii")
    write_mem(out_dir / "mem" / "log_lut.mem", lut, p.log_out_w, "bang nen log, khong dau Q1.10")
    if scaler is not None:
        write_mem(out_dir / "mem" / "scaler_mean_q.mem", scaler.mean_q, p.data_w, "mean Q5.10")
        write_mem(out_dir / "mem" / "scaler_inv_q.mem", scaler.inv_q, p.inv_w, f"1/sigma, {scaler.inv_frac} bit le")
    rows = []
    for k, (i1, i2) in enumerate(pairs):
        tr = hq.features_q(i1, i2, p, scaler, lut, keep_iter1=True)
        info = export_case(out_dir / f"case_{k:02d}", i1, i2, tr, p, ocsvm)
        rows.append({"case": k, "name": names[k] if names else "", "u_absmax": int(np.abs(tr.hs.u).max()),
                     "v_absmax": int(np.abs(tr.hs.v).max()), "n_sat": tr.hs.n_sat,
                     "max_abs_N": tr.hs.max_abs_n, **info})
    L = ["# Véc-tơ kiểm thử khối trích đặc trưng (RTL)", "",
         "| Ca | Nội dung | max\\|u\\| (LSB) | max\\|v\\| (LSB) | bão hòa u/v | max\\|N\\| | f_q | bất thường |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['case']:02d} | {r['name']} | {r['u_absmax']} | {r['v_absmax']} | {r['n_sat']} | "
                 f"{r['max_abs_N']} | {r.get('f_q', '–')} | {r.get('anomaly', '–')} |")
    (out_dir / "index.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return rows
