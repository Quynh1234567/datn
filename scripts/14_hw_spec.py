#!/usr/bin/env python3
"""BẢNG ĐẶC TẢ PHẦN CỨNG – sinh tự động từ đúng mô hình đã xuất (outputs/fpga).

Đọc ocsvm_params.vh + model_float.json + các báo cáo, rồi tính:
  • thông số mô hình, tiền xử lý, đặc trưng
  • định dạng điểm cố định và độ rộng bit từng tín hiệu trong datapath
  • dung lượng bộ nhớ (bit, byte, ước lượng số BRAM18K của Xilinx)
  • số phép tính mỗi lần suy luận
  • độ trễ / thông lượng FPGA theo mức song song P và tần số xung nhịp
  • thời gian truyền UART theo tốc độ baud và ngân sách thời gian mỗi khung
  • độ chính xác (AUC float/fixed, sai số lượng tử, tỉ lệ trùng nhãn)

    python scripts/14_hw_spec.py
    python scripts/14_hw_spec.py --fclk 50 100 --baud 115200 921600 --pc-svm-us 10.7

Kết quả: outputs/fpga/hw_spec.md (đưa vào báo cáo) và outputs/fpga/hw_spec.json
Các con số độ trễ/tài nguyên là ƯỚC LƯỢNG kiến trúc – số đo thật lấy từ báo cáo
tổng hợp Vivado/Quartus và mô phỏng RTL.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocsvm_fpga.config import load_config, resolve_path  # noqa: E402

# Cấu hình BRAM18K (Xilinx 7-series): (độ sâu, độ rộng)
BRAM18_CFG = [(16384, 1), (8192, 2), (4096, 4), (2048, 9), (1024, 18), (512, 36)]
PIPE_OVERHEAD = 12          # chu kỳ khởi động pipeline (đọc BRAM, nhân, LUT, MAC) – ước lượng
UART_BITS_PER_BYTE = 10     # 1 start + 8 data + 1 stop


def read_vh(path: Path) -> dict:
    out = {}
    for m in re.finditer(r"localparam\s+(?:signed\s+\[[^\]]+\]\s+|integer\s+)?(\w+)\s*=\s*([^;]+);",
                         path.read_text(encoding="utf-8", errors="ignore")):
        expr = m.group(2).strip()
        neg = expr.startswith("-")
        expr = expr.lstrip("-").strip()
        try:
            v = int(expr.split("'sd")[1]) if "'sd" in expr else int(expr)
            out[m.group(1)] = -v if neg else v
        except ValueError:
            pass
    return out


def bram18(depth: int, width: int) -> int:
    """Số khối BRAM18K tối thiểu cho bộ nhớ depth × width."""
    return min(math.ceil(depth / d) * math.ceil(width / w) for d, w in BRAM18_CFG)


def num(v) -> str:
    """Số nguyên có dấu cách phân tách hàng nghìn: 158 496."""
    return f"{int(v):,}".replace(",", " ")


def fmt_bits(bits: int) -> str:
    return f"{num(bits)} bit ({bits / 8 / 1024:.2f} KB)"


def q(width, frac, signed=True):
    i = width - frac - (1 if signed else 0)
    return f"Q{i}.{frac} {'có dấu' if signed else 'không dấu'}, {width} bit"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fclk", type=float, nargs="+", default=[100.0], help="tần số xung nhịp FPGA (MHz)")
    ap.add_argument("--baud", type=int, nargs="+", default=[115200, 921600])
    ap.add_argument("--parallel", type=int, nargs="*", default=None, help="mức song song P (mặc định 1,4,8,16,DIM)")
    ap.add_argument("--video-fps", type=float, default=25.0, help="fps của video huấn luyện (IITB = 25)")
    ap.add_argument("--pc-svm-us", type=float, default=None, help="thời gian OC-SVM C trên PC (µs) – lấy từ log bước 07")
    ap.add_argument("--esp32-svm-us", type=float, default=None, help="thời gian OC-SVM trên ESP32 (µs) nếu đã đo")
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = resolve_path(cfg.project.output_dir)
    fdir = out / "fpga"
    if not (fdir / "ocsvm_params.vh").exists():
        sys.exit("Chưa có outputs/fpga/ocsvm_params.vh – chạy scripts/04_export_fpga.py trước")
    P_ = read_vh(fdir / "ocsvm_params.vh")
    mf = json.loads((fdir / "model_float.json").read_text(encoding="utf-8"))
    qr_path = out / "reports" / "quant_report.json"
    qr = json.loads(qr_path.read_text(encoding="utf-8")) if qr_path.exists() else {}
    ev_path = out / "reports" / "eval_report.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else {}

    D, N = P_["DIM"], P_["N_SV"]
    DW, DF, CW, CF = P_["DATA_W"], P_["DATA_FRAC"], P_["COEF_W"], P_["COEF_FRAC"]
    GW, GF, GQ = P_["GAMMA_W"], P_["GAMMA_FRAC"], P_["GAMMA_Q"]
    LA, LS, LW, LSH = P_["LUT_ADDR_W"], P_["LUT_SIZE"], P_["LUT_W"], P_["LUT_SHIFT"]
    AW, AF = P_["ACC_W"], P_["ACC_FRAC"]
    diff_w, dist_w, t_w, prod_w = P_["DIFF_W"], P_["DIST_W"], P_["T_W"], P_["PROD_W"]
    acc_need = prod_w + math.ceil(math.log2(max(N, 2)))
    lut_tmax = 2 ** (LA - (2 * DF + GF - LSH))
    pp, of, ft = cfg.preprocess, cfg.optical_flow, cfg.features
    kind = ft.get("descriptor", "covariance")
    frame_dt_ms = 1000.0 * pp.frame_step / a.video_fps
    Ps = sorted(set(a.parallel or [1, 4, 8, 16, D]))
    S = {}                                                         # dữ liệu cho JSON
    L = []                                                         # dòng markdown

    def table(title, rows, head=("Thông số", "Giá trị", "Ghi chú")):
        L.extend(["", f"## {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)])
        L.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)

    L.append("# Bảng đặc tả bộ tăng tốc OC-SVM trên FPGA")
    L.append(f"\nSinh tự động từ `{fdir.name}/ocsvm_params.vh` và `model_float.json` – khớp đúng mô hình đã xuất.")

    # 1. Mô hình
    feat_desc = (f"ma trận hiệp biến {ft.feature_set} → vech(logm C)" if kind == "covariance"
                 else f"lưới {ft.grid_rows}×{ft.grid_cols} [{ft.grid_stats}]")
    rows = [
        ("Thuật toán", "One-Class SVM, kernel Gauss RBF", "f(x) = Σ αᵢ·exp(−γ‖x−sᵢ‖²) − ρ"),
        ("Loại đặc trưng", feat_desc, kind),
        ("DIM – số chiều véc-tơ đầu vào", D, "số phần tử gửi qua UART mỗi khung"),
        ("N_SV – số Support Vector", N, "số lần tính kernel mỗi khung"),
        ("γ (gamma)", f"{mf['gamma']:.6g}", f"GAMMA_Q = {GQ}, {GF} bit lẻ"),
        ("ρ (rho)", f"{mf['rho']:.6g}", f"RHO_Q = {P_['RHO_Q']}"),
        ("Ngưỡng quyết định", f"{mf.get('threshold_score', 0):.6g} (miền điểm)", f"THR_Q = {P_['THR_Q']}"),
        ("Quy tắc quyết định", "bất thường nếu f(x) < THR" if P_["DECISION_MODE"] == 0 else "bất thường nếu |f(x)| > THR",
         f"DECISION_MODE = {P_['DECISION_MODE']}"),
        ("Làm mượt sau FPGA", f"{P_['SMOOTH_WIN']} khung", "trung bình trượt trên ESP32"),
    ]
    table("1. Thông số mô hình", rows)
    S["model"] = {"dim": D, "n_sv": N, "gamma": mf["gamma"], "rho": mf["rho"], "descriptor": kind}

    # 2. Tiền xử lý
    rows = [
        ("Độ phân giải xử lý", f"{pp.width}×{pp.height} xám", "OV5640 QQVGA"),
        ("Khoảng cách 2 khung", f"frame_step = {pp.frame_step} → {frame_dt_ms:.1f} ms", f"video gốc {a.video_fps:g} fps"),
        ("Làm mượt", f"Gaussian {pp.gaussian_ksize}×{pp.gaussian_ksize}, σ = {pp.gaussian_sigma}", ""),
        ("Dòng quang học", f"{of.get('method', 'hs')}: α = {of.alpha}, {of.num_iter} vòng lặp", "chạy trên ESP32"),
        ("Bỏ khung tối", f"độ sáng TB < {pp.get('skip_dark_below', 0)}", ""),
        ("Chuẩn hóa đầu vào", "z = (x − mean)·inv_std", "trên ESP32 (ocsvm_params.h) hoặc FPGA (scaler_*.mem)"),
    ]
    table("2. Tiền xử lý và trích đặc trưng (ESP32-S3)", rows)

    # 3. Định dạng điểm cố định
    rows = [
        ("x, SV (đầu vào, support vector)", q(DW, DF), f"phạm vi ±{2 ** (DW - DF - 1)}, bước {2 ** -DF:.2e}"),
        ("diff = x − sv", f"{diff_w} bit có dấu", "1 bộ trừ / làn"),
        ("dist = Σ diff²", f"{dist_w} bit không dấu, Q·{2 * DF}", f"tích lũy {D} số hạng"),
        ("γ", f"{GW} bit không dấu, {GF} bit lẻ", f"GAMMA_Q = {GQ} (γ = GAMMA_Q / 2^{GF})"),
        ("t = dist · γ", f"{t_w} bit, Q·{2 * DF + GF}", "1 bộ nhân / SV"),
        ("Địa chỉ LUT = t ≫ LUT_SHIFT", f"{LA} bit (LUT_SHIFT = {LSH})", f"t ≥ {lut_tmax} → k = 0"),
        ("k = exp(−t) (LUT)", q(LW, LW, False), f"{LS} phần tử, bước Δt = {lut_tmax / LS:.4g}"),
        ("α (hệ số)", q(CW, CF), "đã chuẩn hóa: giá trị tuyệt đối lớn nhất của α = 1"),
        ("α · k", f"{prod_w} bit có dấu", "1 bộ nhân / SV"),
        ("Bộ cộng dồn f", f"{AW} bit có dấu, Q·{AF}", f"cần tối thiểu {acc_need} bit (trường hợp xấu nhất)"),
    ]
    table("3. Định dạng điểm cố định và độ rộng bit", rows, ("Tín hiệu", "Định dạng", "Ghi chú"))
    S["fixed_point"] = {"DATA": [DW, DF], "COEF": [CW, CF], "GAMMA": [GW, GF, GQ], "LUT": [LA, LW, LSH],
                        "ACC": [AW, AF], "DIFF_W": diff_w, "DIST_W": dist_w, "T_W": t_w, "PROD_W": prod_w}

    # 4. Bộ nhớ
    mems = [
        ("SV (sv.mem, 1 phần tử/từ)", N * D, DW, "đọc tuần tự P = 1"),
        (f"SV (sv_packed.mem, cả véc-tơ/từ)", N, D * DW, "đọc song song P = DIM"),
        ("α (alpha.mem)", N, CW, ""),
        ("LUT exp (exp_lut.mem)", LS, LW, "ROM"),
        ("Bộ đệm đầu vào x", D, DW, "thanh ghi hoặc LUTRAM"),
        ("Scaler mean + inv_std (tùy chọn)", 2 * D, 32, "chỉ khi chuẩn hóa trên FPGA"),
    ]
    rows, total_bits = [], 0
    for name, depth, width, note in mems:
        bits = depth * width
        if "tùy chọn" not in name and "sv_packed" not in name:
            total_bits += bits
        rows.append((name, f"{depth} × {width}", fmt_bits(bits), bram18(depth, width), note))
    rows.append(("**Tổng (SV tuần tự + α + LUT + x)**", "", fmt_bits(total_bits),
                 bram18(N * D, DW) + bram18(N, CW) + bram18(LS, LW), ""))
    table("4. Bộ nhớ", rows, ("Bộ nhớ", "Độ sâu × độ rộng", "Dung lượng", "BRAM18K (ước lượng)", "Ghi chú"))
    S["memory_bits_total"] = total_bits

    # 5. Khối lượng tính toán
    rows = [
        ("Phép trừ (x − sv)", num(N * D), ""),
        ("Phép nhân bình phương diff²", num(N * D), f"{diff_w}×{diff_w} bit"),
        ("Phép cộng tích lũy dist", num(N * D), ""),
        ("Phép nhân dist·γ", N, f"{dist_w}×{GW} bit"),
        ("Tra LUT exp", N, ""),
        ("Phép nhân α·k", N, f"{CW}×{LW} bit"),
        ("Phép cộng tích lũy f", N, f"{AW} bit"),
        ("**Tổng phép nhân-cộng (MAC) mỗi khung**", num(N * D + 2 * N), ""),
    ]
    table("5. Khối lượng tính toán mỗi lần suy luận", rows, ("Phép toán", "Số lần", "Độ rộng"))

    # 6. Độ trễ theo P và fclk
    rows = []
    S["latency"] = []
    for f in a.fclk:
        for p in Ps:
            cyc = N * math.ceil(D / p) + PIPE_OVERHEAD
            us = cyc / f
            dsp = p + 2 + 1      # P bộ bình phương + ~2 DSP cho dist·γ + 1 DSP cho α·k
            sv_bram = bram18(N * math.ceil(D / p), p * DW)
            rows.append((f"{f:g} MHz", p, num(cyc), f"{us:.2f} µs", num(1e6 / us),
                         dsp, sv_bram))
            S["latency"].append({"fclk": f, "P": p, "cycles": cyc, "us": us, "dsp": dsp, "sv_bram18": sv_bram})
    table("6. Độ trễ và tài nguyên FPGA theo mức song song P (ước lượng)", rows,
          ("f_clk", "P", "Chu kỳ", "Độ trễ", "Suy luận/giây", "DSP (ước lượng)", "BRAM18K cho SV"))
    L.append(f"\nChu kỳ ≈ N_SV × ⌈DIM/P⌉ + {PIPE_OVERHEAD} (khởi động pipeline). DSP ≈ P (bình phương) "
             "+ 2 (dist·γ) + 1 (α·k), tính theo DSP48 25×18 của Xilinx; LUT exp và α không phụ thuộc P.")

    # 7. UART và ngân sách thời gian
    req = 2 + 2 * D + 1
    resp = 1 + AW // 8 + 1 + 1
    rows = []
    S["uart"] = []
    for b in a.baud:
        t_req = req * UART_BITS_PER_BYTE / b * 1000
        t_resp = resp * UART_BITS_PER_BYTE / b * 1000
        rows.append((num(b), f"{req} B → {t_req:.2f} ms", f"{resp} B → {t_resp:.3f} ms",
                     f"{t_req + t_resp:.2f} ms", f"{(t_req + t_resp) / frame_dt_ms:.1%}"))
        S["uart"].append({"baud": b, "req_bytes": req, "resp_bytes": resp, "ms": t_req + t_resp})
    table(f"7. Truyền UART và ngân sách thời gian (1 khung = {frame_dt_ms:.1f} ms)", rows,
          ("Baud", "ESP32 → FPGA", "FPGA → ESP32", "Tổng truyền", "% ngân sách khung"))
    L.append("\nGói yêu cầu: 0xAA 0x55 | DIM × int16 | XOR. Gói trả lời: 0x5A | f_q "
             f"({AW // 8} B) | cờ | XOR. Thời gian truyền UART lớn hơn thời gian tính trên FPGA hàng trăm lần "
             "→ trong hệ thống thật, UART và trích đặc trưng trên ESP32 mới là phần quyết định độ trễ.")

    # 8. So sánh phần mềm
    best = min(S["latency"], key=lambda r: r["us"])
    seq = next(r for r in S["latency"] if r["P"] == min(Ps) and r["fclk"] == a.fclk[0])
    rows = [("FPGA, P = %d, %g MHz" % (seq["P"], seq["fclk"]), f"{seq['us']:.2f} µs", "ước lượng"),
            ("FPGA, P = %d, %g MHz" % (best["P"], best["fclk"]), f"{best['us']:.2f} µs", "ước lượng")]
    if a.pc_svm_us:
        rows.append(("OC-SVM C số nguyên trên PC", f"{a.pc_svm_us:.2f} µs",
                     f"FPGA nhanh hơn ≈ {a.pc_svm_us / best['us']:.1f}× (P = {best['P']})"))
    if a.esp32_svm_us:
        rows.append(("OC-SVM C số nguyên trên ESP32-S3", f"{a.esp32_svm_us:.2f} µs",
                     f"FPGA nhanh hơn ≈ {a.esp32_svm_us / best['us']:.1f}× (P = {best['P']})"))
    if not (a.pc_svm_us or a.esp32_svm_us):
        rows.append(("OC-SVM phần mềm", "–", "thêm --pc-svm-us / --esp32-svm-us để tính hệ số tăng tốc"))
    table("8. So sánh thời gian suy luận OC-SVM", rows, ("Nền tảng", "Thời gian / khung", "Ghi chú"))

    # 9. Độ chính xác
    rows = []
    if qr:
        rows += [
            ("Sai số tuyệt đối f_fixed − f_float lớn nhất", f"{qr.get('decision_abs_err_max', float('nan')):.3e}", ""),
            ("Sai số trung bình", f"{qr.get('decision_abs_err_mean', float('nan')):.3e}", ""),
            ("Tỉ lệ trùng nhãn float ↔ fixed", f"{qr.get('label_agreement_per_frame', float('nan')):.2%}", "mỗi khung"),
            ("AUC float / fixed (thô)", f"{qr.get('auc_float', float('nan')):.4f} / {qr.get('auc_fixed', float('nan')):.4f}",
             "trên tập kiểm chứng của bước 04"),
        ]
    ts = ev.get("test_smoothed", {})
    if ts:
        rows.append(("AUC mô hình (làm mượt)", f"{ts.get('auc', float('nan')):.4f}", f"{ts.get('n_frames', '')} cặp khung Test"))
        rows.append(("EER", f"{ts.get('eer', float('nan')):.4f}", ""))
    rows.append(("Vector kiểm thử cho testbench", f"{P_['N_TEST']} vector", "test_input.mem / test_expected_score.mem"))
    table("9. Độ chính xác và kiểm chứng", rows)

    # 10. Danh sách file
    rows = [(n, d) for n, d in [
        ("ocsvm_params.vh", "localparam cho RTL (mọi thông số ở trên)"),
        ("mem/sv.mem, sv_packed.mem", "support vector (tuần tự / song song)"),
        ("mem/alpha.mem", "hệ số α"),
        ("mem/exp_lut.mem", "bảng tra exp(−t)"),
        ("mem/test_input.mem, test_expected_score.mem, test_expected_label.mem", "vector kiểm thử + kết quả mong đợi"),
        ("mem/test_kernel_vec0.mem", "giá trị kernel từng SV của vector 0 (gỡ lỗi PE)"),
        ("coe/*.coe", "khởi tạo Block Memory Generator (Vivado)"),
        ("ocsvm_params.h, ocsvm_model_data.h", "firmware ESP32"),
    ]]
    table("10. File bàn giao", rows, ("File", "Nội dung"))

    # Tóm tắt (chèn lên đầu)
    seq_r = next(r for r in S["latency"] if r["P"] == 1 or r["P"] == min(Ps))
    par_r = next(r for r in S["latency"] if r["P"] == max(Ps) and r["fclk"] == a.fclk[0])
    u_fast = min(S["uart"], key=lambda r: r["ms"])
    summ = ["", "## Tóm tắt", "", "| Thông số | Giá trị |", "|---|---|",
            f"| DIM × N_SV | {D} × {N} |",
            f"| Bộ nhớ trên chip (SV + α + LUT + x) | {fmt_bits(total_bits)} |",
            f"| Phép nhân-cộng mỗi khung | {num(N * D + 2 * N)} |",
            f"| Độ trễ FPGA, P = {seq_r['P']} / P = {par_r['P']} @ {a.fclk[0]:g} MHz | {seq_r['us']:.2f} µs / {par_r['us']:.2f} µs |",
            f"| Truyền UART mỗi khung @ {num(u_fast['baud'])} baud | {u_fast['ms']:.2f} ms |",
            f"| Ngân sách thời gian mỗi khung | {frame_dt_ms:.1f} ms |"]
    if ts:
        summ.append(f"| AUC (Test, làm mượt) | {ts.get('auc', float('nan')):.4f} |")
    L[2:2] = summ
    (fdir / "hw_spec.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (fdir / "hw_spec.json").write_text(json.dumps(S, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))
    print(f"\n→ {fdir / 'hw_spec.md'}")


if __name__ == "__main__":
    main()
