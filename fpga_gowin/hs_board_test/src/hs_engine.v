// -----------------------------------------------------------------------------
// hs_engine.v – khối trích đặc trưng hoàn chỉnh: DERIV → HS_ITER (N vòng) → FEAT → ghi 48 z
//
// Các cổng bộ nhớ là LUỒNG valid/ready, chưa gắn SDRAM (bước sau: bộ điều khiển SDRAM +
// phân xử nối vào 3 luồng này):
//   pix_*  : đọc 2 khung từ SDRAM, mỗi nhịp {i2, i1} theo hàng (19 200 nhịp)
//   dw_*   : ghi gói DERIV 64 bit vào vùng DERIV của SDRAM (19 200 gói)
//   dr_*   : đọc lại vùng DERIV, phát N_ITER lần liên tiếp (dr_req = 1 trong suốt pha này)
// Đầu ra x_we / x_addr / x_data / svm_start nối thẳng ocsvm_core (giao diện có sẵn).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
module hs_engine #(
    parameter integer NIT       = 0,                  // 0 = N_ITER trong hs_params.vh
    parameter         LUT_FILE  = "log_lut.mem",
    parameter         MEAN_FILE = "scaler_mean_q.mem",
    parameter         INV_FILE  = "scaler_inv_q.mem"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    // ảnh vào
    input  wire        pix_valid,
    output wire        pix_ready,
    input  wire [15:0] pix_data,
    // gói DERIV ra (ghi SDRAM)
    output wire        dw_valid,
    input  wire        dw_ready,
    output wire [63:0] dw_data,
    // gói DERIV vào (đọc SDRAM N_ITER lần)
    output wire        dr_req,
    input  wire        dr_valid,
    output wire        dr_ready,
    input  wire [63:0] dr_data,
    // tới ocsvm_core
    output wire        x_we,
    output wire [7:0]  x_addr,
    output wire [15:0] x_data,
    output reg         svm_start,
    // trạng thái
    output reg         busy,
    output reg         done,
    output reg  [1:0]  phase,                         // 0 rỗi · 1 DERIV · 2 HS_ITER (+FEAT) · 3 FEAT
    output wire [31:0] n_sat,
    output reg  [31:0] cycles                         // số chu kỳ từ start tới done (đo hiệu năng)
);
    wire d_done, d_busy, i_done, i_busy, f_done;
    wire        o_valid;
    wire signed [12:0] o_u, o_v;
    wire [6:0]  iter;
    reg         d_start, i_start;

    hs_deriv u_deriv (.clk(clk), .rst_n(rst_n), .start(d_start),
        .in_valid(pix_valid), .in_ready(pix_ready), .in_data(pix_data),
        .out_valid(dw_valid), .out_ready(dw_ready), .out_data(dw_data), .done(d_done), .busy(d_busy));

    hs_iter #(.NIT(NIT)) u_iter (.clk(clk), .rst_n(rst_n), .start(i_start),
        .in_valid(dr_valid), .in_ready(dr_ready), .in_data(dr_data),
        .out_valid(o_valid), .out_u(o_u), .out_v(o_v), .done(i_done), .busy(i_busy), .n_sat(n_sat), .iter(iter));

    hs_feat #(.LUT_FILE(LUT_FILE), .MEAN_FILE(MEAN_FILE), .INV_FILE(INV_FILE)) u_feat (
        .clk(clk), .rst_n(rst_n), .start(i_start), .in_valid(o_valid), .in_u(o_u), .in_v(o_v),
        .mag_valid(), .mag(), .res_valid(), .res_d(), .res_sum(), .res_mean(), .res_log(), .res_z(),
        .x_we(x_we), .x_addr(x_addr), .x_data(x_data), .done(f_done));

    assign dr_req = (phase == 2'd2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 2'd0; busy <= 1'b0; done <= 1'b0; svm_start <= 1'b0; d_start <= 1'b0; i_start <= 1'b0;
            cycles <= 0;
        end else begin
            done <= 1'b0; svm_start <= 1'b0; d_start <= 1'b0; i_start <= 1'b0;
            if (busy) cycles <= cycles + 1'b1;
            case (phase)
            2'd0: if (start) begin phase <= 2'd1; busy <= 1'b1; d_start <= 1'b1; cycles <= 0; end
            2'd1: if (d_done) begin phase <= 2'd2; i_start <= 1'b1; end
            2'd2: if (i_done) phase <= 2'd3;
            2'd3: if (f_done) begin phase <= 2'd0; busy <= 1'b0; done <= 1'b1; svm_start <= 1'b1; end
            endcase
        end
    end
endmodule
