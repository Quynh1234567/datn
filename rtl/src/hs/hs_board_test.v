// -----------------------------------------------------------------------------
// hs_board_test.v – TOP NẠP BOARD Tang Nano 20K để chạy thử khối HS + OC-SVM khi CHƯA có SDRAM
//
//  • Ảnh: hs_pixgen (công thức cố định) thay cho camera/SDRAM.
//  • Gói DERIV: thay vì ghi rồi đọc lại SDRAM 64 lần, một hs_deriv thứ hai TÍNH LẠI gói từ ảnh
//    ở mỗi vòng (cùng kết quả, cùng tốc độ ~2 chu kỳ/gói như SDRAM) → không cần 1,2 Mbit bộ nhớ.
//  • Mỗi lần chạy xong gửi 1 dòng UART (115 200 baud, 8N1) rồi chạy lại:
//      HS N=0001 C=0026181F F=B1E9C1AF A=1 S=00000000 Z=1A2B\r\n
//    N số lần chạy · C số chu kỳ khối HS · F f_q (32 bit thấp) · A cờ bất thường · S số lần bão hòa
//    Z tổng 48 giá trị z (16 bit). So với dòng mong đợi in bởi scripts/25_hs_board_test.py.
//  • LED0 đảo mỗi lần chạy xong (nhấp nháy = đang chạy), LED1 sáng khi cờ bất thường = 1.
// Xung nhịp: thạch anh 27 MHz trực tiếp (chưa cần rPLL); thời gian = C / 27 MHz.
// Reset tự sinh khi nạp xong (128 chu kỳ) – không phụ thuộc mức tích cực của nút trên board.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
module hs_board_test #(
    parameter integer CLK_HZ    = 27_000_000,
    parameter integer BAUD      = 115_200,
    parameter         LUT_FILE  = "log_lut.mem",
    parameter         MEAN_FILE = "scaler_mean_q.mem",
    parameter         INV_FILE  = "scaler_inv_q.mem",
    parameter         SV_FILE   = "sv_lanes.mem",
    parameter         ALPHA_FILE = "alpha.mem",
    parameter         EXP_FILE  = "exp_lut.mem"
)(
    input  wire clk,
    output wire uart_tx,
    output reg  led_busy_n,
    output wire led_anomaly_n
);
`undef HS_PARAMS_VH
`include "hs_params.vh"                              // N_ITER: số lượt phát lại gói DERIV

    // ---------------------------------------------------------------- reset khi bật nguồn (không dùng nút)
    reg [7:0] por = 8'd0;
    always @(posedge clk) if (!por[7]) por <= por + 1'b1;
    wire rstn = por[7];

    // ---------------------------------------------------------------- khối HS
    reg         go;
    wire        pv, pr, dwv, drq, drv, drr, x_we, svm_start, busy, done;
    wire [15:0] pd;
    wire [63:0] dwd, drd;
    wire [7:0]  x_addr;
    wire [15:0] x_data;
    wire [1:0]  phase;
    wire [31:0] n_sat, cycles;
    wire        g_we, g_ov, g_rv;
    wire [25:0] g_wd, g_ouv;
    wire [5:0]  g_rd;
    wire [23:0] g_rs;
    hs_pixgen g_img (.clk(clk), .rst_n(rstn), .start(go), .valid(pv), .ready(pr), .data(pd));
    hs_engine #(.LUT_FILE(LUT_FILE), .MEAN_FILE(MEAN_FILE), .INV_FILE(INV_FILE)) u_hs (
        .clk(clk), .rst_n(rstn), .start(go),
        .pix_valid(pv), .pix_ready(pr), .pix_data(pd),
        .dw_valid(dwv), .dw_ready(1'b1), .dw_data(dwd),
        .dr_req(drq), .dr_valid(drv), .dr_ready(drr), .dr_data(drd),
        .x_we(x_we), .x_addr(x_addr), .x_data(x_data), .svm_start(svm_start),
        .busy(busy), .done(done), .phase(phase), .n_sat(n_sat), .cycles(cycles),
        .dbg_we(g_we), .dbg_wd(g_wd), .dbg_ov(g_ov), .dbg_ouv(g_ouv), .dbg_rv(g_rv), .dbg_rd(g_rd), .dbg_rs(g_rs));

    // ---------------------------------------------------------------- "SDRAM" thay thế: tính lại gói
    reg  rp_start;
    reg  [7:0] rp_cnt;
    wire rp_v, rp_r, rp_done, rp_busy;
    wire [15:0] rp_d;
    hs_pixgen g_rep (.clk(clk), .rst_n(rstn), .start(rp_start), .valid(rp_v), .ready(rp_r), .data(rp_d));
    hs_deriv  d_rep (.clk(clk), .rst_n(rstn), .start(rp_start),
                     .in_valid(rp_v), .in_ready(rp_r), .in_data(rp_d),
                     .out_valid(drv), .out_ready(drr), .out_data(drd), .done(rp_done), .busy(rp_busy));
    // trong pha HS_ITER: phát lại đúng N_ITER lượt, lượt sau bắt đầu khi lượt trước xong
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin rp_start <= 1'b0; rp_cnt <= 0; end
        else begin
            rp_start <= 1'b0;
            if (go) rp_cnt <= 0;
            else if (drq && !rp_busy && !rp_start && rp_cnt < N_ITER) begin
                rp_start <= 1'b1; rp_cnt <= rp_cnt + 1'b1;
            end
        end
    end

    // dw: chỉ cộng dồn để logic DERIV không bị tối ưu mất
    reg [63:0] dw_x;
    always @(posedge clk) if (go) dw_x <= 0; else if (dwv) dw_x <= dw_x ^ dwd;

    // ---------------------------------------------------------------- OC-SVM
    wire svm_busy, svm_done, anomaly;
    wire signed [63:0] f_out;
    ocsvm_core #(.P(8), .SV_FILE(SV_FILE), .ALPHA_FILE(ALPHA_FILE), .LUT_FILE(EXP_FILE)) u_svm (
        .clk(clk), .rst_n(rstn), .x_we(x_we), .x_addr(x_addr), .x_data(x_data),
        .start(svm_start), .busy(svm_busy), .done(svm_done), .f_out(f_out), .anomaly(anomaly));
    assign led_anomaly_n = ~anomaly;

    reg [15:0] zsum;
    always @(posedge clk) if (go) zsum <= 0; else if (x_we) zsum <= zsum + x_data;

    // ---------------------------------------------------------------- GỠ LỖI: giữ 48 giá trị z đúng như OC-SVM nhận
    // và đếm số lần ghi (phải = 48). In thành dòng thứ hai: "ZV W=30 z0 z1 … z47" (hex 4 chữ số mỗi giá trị)
    reg [15:0] zbuf [0:47] /* synthesis syn_ramstyle = "registers" */;
    reg [7:0]  wcnt;
    always @(posedge clk) if (x_we && x_addr < 8'd48) zbuf[x_addr[5:0]] <= x_data;
    always @(posedge clk) if (go) wcnt <= 0; else if (x_we) wcnt <= wcnt + 1'b1;

    // ---------------------------------------------------------------- GỠ LỖI TỪNG TẦNG: dòng "DB" (không phụ thuộc mô hình)
    // hash 32 bit h = ((h << 5) + h) ^ x theo đúng thứ tự luồng – scripts/25_hs_board_test.py tính giá trị đúng
    //   H_DW : 19 200 gói DERIV của hs_engine (x = gói[31:0] ^ gói[63:32])
    //   H_DR : 19 200 gói DERIV đầu tiên vào hs_iter (khối DERIV thứ hai)
    //   H_I1 : 19 200 lần ghi uv_ram đầu tiên = u, v sau vòng 1 (x = {v, u} 26 bit)
    //   H_FN : luồng u, v vòng cuối vào FEAT · N_FN số điểm · S_U, S_V tổng u, v vòng cuối · CS 48 tổng ô (24 bit)
    function [31:0] hstep; input [31:0] h, x; hstep = ((h << 5) + h) ^ x; endfunction
    reg [31:0] h_dw, h_dr, h_i1, h_fn, s_u, s_v;
    reg [15:0] n_dw, n_dr, n_i1, n_fn;
    always @(posedge clk) begin
        if (go) begin
            h_dw <= 32'd5381; h_dr <= 32'd5381; h_i1 <= 32'd5381; h_fn <= 32'd5381; s_u <= 0; s_v <= 0;
            n_dw <= 0; n_dr <= 0; n_i1 <= 0; n_fn <= 0;
        end else begin
            if (dwv && n_dw < 16'd19200) begin h_dw <= hstep(h_dw, dwd[31:0] ^ dwd[63:32]); n_dw <= n_dw + 1'b1; end
            if (drv && drr && n_dr < 16'd19200) begin h_dr <= hstep(h_dr, drd[31:0] ^ drd[63:32]); n_dr <= n_dr + 1'b1; end
            if (g_we && n_i1 < 16'd19200) begin h_i1 <= hstep(h_i1, {6'd0, g_wd}); n_i1 <= n_i1 + 1'b1; end
            if (g_ov) begin
                h_fn <= hstep(h_fn, {6'd0, g_ouv}); n_fn <= n_fn + 1'b1;
                s_u  <= s_u + {{19{g_ouv[12]}}, g_ouv[12:0]};
                s_v  <= s_v + {{19{g_ouv[25]}}, g_ouv[25:13]};
            end
        end
    end
    wire [24*48-1:0] cs_flat;
    genvar gc;
    generate for (gc = 0; gc < 48; gc = gc + 1) begin : csg
        reg [23:0] r;
        always @(posedge clk) if (g_rv && g_rd == gc) r <= g_rs;
        assign cs_flat[(47 - gc) * 24 +: 24] = r;                     // ô 0 đứng đầu dòng
    end endgenerate
    wire [32*6+16-1:0] db_head = {h_dw, h_dr, h_i1, h_fn, n_fn, s_u, s_v};
    localparam integer NDB = (32 * 6 + 16 + 24 * 48) / 4;              // 340 chữ số hex
    wire [32*6+16+24*48-1:0] dbv = {db_head, cs_flat};
    reg  [8:0] ci3;
    wire [8:0] k3 = ci3 - 9'd3;
    reg  [7:0] ch3;
    always @(*) begin
        if (ci3 == 0) ch3 = "D"; else if (ci3 == 1) ch3 = "B"; else if (ci3 == 2) ch3 = " ";
        else if (ci3 < NDB + 3) ch3 = hx(dbv[(NDB - 1 - k3) * 4 +: 4]);
        else if (ci3 == NDB + 3) ch3 = 8'h0D; else ch3 = 8'h0A;
    end

    // ---------------------------------------------------------------- điều khiển + UART
    localparam integer NCH = 58;
    reg [31:0] r_c, r_f, r_s;
    reg [15:0] r_z, nrun;
    reg        r_a;
    reg [2:0]  ms;
    localparam [2:0] M_GO = 3'd0, M_RUN = 3'd1, M_TX = 3'd2, M_TXW = 3'd3, M_TX2 = 3'd4, M_TXW2 = 3'd5,
                     M_TX3 = 3'd6, M_TXW3 = 3'd7;
    reg [7:0]  ci2;                                       // ký tự của dòng ZV (0 … 201)
    localparam integer NCH2 = 202;
    reg [5:0]  ci;
    reg        send;
    wire       tx_busy;
    function [7:0] hx; input [3:0] v; hx = (v < 10) ? (8'h30 + v) : (8'h41 + v - 10); endfunction
    reg [7:0] ch;
    always @(*) begin
        case (ci)
        0: ch = "H"; 1: ch = "S"; 2: ch = " "; 3: ch = "N"; 4: ch = "=";
        5: ch = hx(nrun[15:12]); 6: ch = hx(nrun[11:8]); 7: ch = hx(nrun[7:4]); 8: ch = hx(nrun[3:0]);
        9: ch = " "; 10: ch = "C"; 11: ch = "=";
        12: ch = hx(r_c[31:28]); 13: ch = hx(r_c[27:24]); 14: ch = hx(r_c[23:20]); 15: ch = hx(r_c[19:16]);
        16: ch = hx(r_c[15:12]); 17: ch = hx(r_c[11:8]);  18: ch = hx(r_c[7:4]);   19: ch = hx(r_c[3:0]);
        20: ch = " "; 21: ch = "F"; 22: ch = "=";
        23: ch = hx(r_f[31:28]); 24: ch = hx(r_f[27:24]); 25: ch = hx(r_f[23:20]); 26: ch = hx(r_f[19:16]);
        27: ch = hx(r_f[15:12]); 28: ch = hx(r_f[11:8]);  29: ch = hx(r_f[7:4]);   30: ch = hx(r_f[3:0]);
        31: ch = " "; 32: ch = "A"; 33: ch = "="; 34: ch = hx({3'b0, r_a});
        35: ch = " "; 36: ch = "S"; 37: ch = "=";
        38: ch = hx(r_s[31:28]); 39: ch = hx(r_s[27:24]); 40: ch = hx(r_s[23:20]); 41: ch = hx(r_s[19:16]);
        42: ch = hx(r_s[15:12]); 43: ch = hx(r_s[11:8]);  44: ch = hx(r_s[7:4]);   45: ch = hx(r_s[3:0]);
        46: ch = " "; 47: ch = "Z"; 48: ch = "=";
        49: ch = hx(r_z[15:12]); 50: ch = hx(r_z[11:8]); 51: ch = hx(r_z[7:4]); 52: ch = hx(r_z[3:0]);
        53: ch = " "; 54: ch = "D"; 55: ch = hx({3'b0, ^dw_x});
        56: ch = 8'h0D; default: ch = 8'h0A;
        endcase
    end
    reg  [7:0]  ch2;
    wire [7:0]  j2  = ci2 - 8'd8;
    wire [15:0] zv  = zbuf[j2[7:2]];
    always @(*) begin
        if (ci2 == 0) ch2 = "Z"; else if (ci2 == 1) ch2 = "V"; else if (ci2 == 2) ch2 = " ";
        else if (ci2 == 3) ch2 = "W"; else if (ci2 == 4) ch2 = "=";
        else if (ci2 == 5) ch2 = hx(wcnt[7:4]); else if (ci2 == 6) ch2 = hx(wcnt[3:0]);
        else if (ci2 == 7) ch2 = " ";
        else if (ci2 < 200) begin
            case (j2[1:0])
            2'd0: ch2 = hx(zv[15:12]); 2'd1: ch2 = hx(zv[11:8]); 2'd2: ch2 = hx(zv[7:4]); default: ch2 = hx(zv[3:0]);
            endcase
        end
        else if (ci2 == 200) ch2 = 8'h0D; else ch2 = 8'h0A;
    end
    wire [7:0] tx_ch = (ms == M_TX3 || ms == M_TXW3) ? ch3 : (ms == M_TX2 || ms == M_TXW2) ? ch2 : ch;
    uart_tx #(.CLKS_PER_BIT(CLK_HZ / BAUD)) u_tx (.clk(clk), .rst_n(rstn), .data(tx_ch), .send(send),
                                                  .tx(uart_tx), .busy(tx_busy));
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            ms <= M_GO; go <= 1'b0; ci <= 0; ci2 <= 0; send <= 1'b0; nrun <= 0; led_busy_n <= 1'b1;
            r_c <= 0; r_f <= 0; r_s <= 0; r_z <= 0; r_a <= 1'b0;
        end else begin
            go <= 1'b0; send <= 1'b0;
            case (ms)
            M_GO:  begin go <= 1'b1; ms <= M_RUN; end
            M_RUN: if (svm_done) begin
                r_c <= cycles; r_f <= f_out[31:0]; r_a <= anomaly; r_s <= n_sat; r_z <= zsum;
                nrun <= nrun + 1'b1; led_busy_n <= ~led_busy_n; ci <= 0; ms <= M_TX;
            end
            M_TX:  if (!tx_busy && !send) begin send <= 1'b1; ms <= M_TXW; end
            M_TXW: if (tx_busy) begin
                if (ci == NCH - 1) begin ci2 <= 0; ms <= M_TX2; end
                else begin ci <= ci + 1'b1; ms <= M_TX; end
            end
            M_TX2:  if (!tx_busy && !send) begin send <= 1'b1; ms <= M_TXW2; end
            M_TXW2: if (tx_busy) begin
                if (ci2 == NCH2 - 1) begin ci3 <= 0; ms <= M_TX3; end
                else begin ci2 <= ci2 + 1'b1; ms <= M_TX2; end
            end
            M_TX3:  if (!tx_busy && !send) begin send <= 1'b1; ms <= M_TXW3; end
            M_TXW3: if (tx_busy) begin
                if (ci3 == NDB + 4) ms <= M_GO;
                else begin ci3 <= ci3 + 1'b1; ms <= M_TX3; end
            end
            endcase
        end
    end
endmodule
