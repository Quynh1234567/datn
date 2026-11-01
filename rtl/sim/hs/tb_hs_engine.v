// -----------------------------------------------------------------------------
// tb_hs_engine.v – kiểm toàn khối: img1/img2 → hs_engine → ocsvm_core → so z (feat_z.mem),
// f_q và cờ bất thường (expected.txt). SDRAM được thay bằng mô hình hành vi trên 3 luồng:
//   +RATE=2  : đọc gói DERIV tối đa 1 gói / 2 chu kỳ (như SDRAM 32 bit, gói 64 bit)
//   +STALL=% : chèn chờ ngẫu nhiên trên mọi luồng (kiểm logic valid/ready)
// In số chu kỳ từ start tới khi có 48 z → thời gian thật của khối ở 100 MHz.
// Biên dịch cần -I <thư mục hs_params.vh> và -I <thư mục ocsvm_params.vh> (rtl/build).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_hs_engine;
    parameter integer NIT = 64;
    parameter SV_FILE = "sv_lanes.mem", ALPHA_FILE = "alpha.mem", EXP_FILE = "exp_lut.mem";
    localparam integer NPIX = 19200;
    reg clk = 1'b0, rst_n = 1'b0, start = 1'b0;
    always #5 clk = ~clk;

    string cdir;                       // SystemVerilog string: chạy được cả Icarus (-g2012) và Vivado xsim
    integer stall, seed, rate, cyc, errs, nz, fd, r;
    reg [7:0]  img1 [0:NPIX-1];
    reg [7:0]  img2 [0:NPIX-1];
    reg [15:0] ez   [0:47];
    reg [63:0] dmem [0:NPIX-1];          // vùng DERIV của "SDRAM"
    reg signed [63:0] exp_f;
    integer exp_a;

    function go; input integer dummy; begin go = (($random(seed) & 127) >= (stall * 128) / 100); end endfunction

    // ---------------------------------------------------------------- DUT
    reg         pv, rv, dw_ready;
    integer     pp, wp, rp, rtot, ph;
    wire        pix_ready, dw_valid, dr_req, dr_ready, x_we, svm_start, busy, done;
    wire [63:0] dw_data;
    wire [7:0]  x_addr;
    wire [15:0] x_data;
    wire [1:0]  phase;
    wire [31:0] n_sat, cycles;
    hs_engine #(.NIT(NIT)) dut (.clk(clk), .rst_n(rst_n), .start(start),
        .pix_valid(pv), .pix_ready(pix_ready), .pix_data({img2[pp], img1[pp]}),
        .dw_valid(dw_valid), .dw_ready(dw_ready), .dw_data(dw_data),
        .dr_req(dr_req), .dr_valid(rv), .dr_ready(dr_ready), .dr_data(dmem[rp]),
        .x_we(x_we), .x_addr(x_addr), .x_data(x_data), .svm_start(svm_start),
        .busy(busy), .done(done), .phase(phase), .n_sat(n_sat), .cycles(cycles));

    wire svm_busy, svm_done, anomaly;
    wire signed [63:0] f_out;
    ocsvm_core #(.P(8), .SV_FILE(SV_FILE), .ALPHA_FILE(ALPHA_FILE), .LUT_FILE(EXP_FILE)) u_svm (
        .clk(clk), .rst_n(rst_n), .x_we(x_we), .x_addr(x_addr), .x_data(x_data),
        .start(svm_start), .busy(svm_busy), .done(svm_done), .f_out(f_out), .anomaly(anomaly));

    // ---------------------------------------------------------------- mô hình bộ nhớ
    always @(posedge clk) begin
        if (!rst_n) begin pp <= 0; pv <= 1'b0; wp <= 0; rp <= 0; rtot <= 0; rv <= 1'b0; dw_ready <= 1'b0; ph <= 0; end
        else begin
            // đọc ảnh
            if (pv && pix_ready) pp <= pp + 1;
            if (!pv || pix_ready) pv <= ((pv && pix_ready ? pp + 1 : pp) < NPIX) && go(0);
            // ghi DERIV
            if (dw_valid && dw_ready) begin dmem[wp] <= dw_data; wp <= wp + 1; end
            dw_ready <= go(0);
            // đọc DERIV N lần
            ph <= (ph + 1 >= rate) ? 0 : ph + 1;
            if (rv && dr_ready) begin rtot <= rtot + 1; rp <= (rp == NPIX - 1) ? 0 : rp + 1; end
            if (!rv || dr_ready)
                rv <= dr_req && ((rv && dr_ready ? rtot + 1 : rtot) < NPIX * NIT) && go(0) && (ph == 0);
        end
    end

    // ---------------------------------------------------------------- so z
    always @(posedge clk) if (rst_n && x_we) begin
        if (x_data !== ez[x_addr]) begin
            errs = errs + 1;
            if (errs <= 3) $display("FAIL tb_hs_engine %0s: z[%0d] RTL %0d mong doi %0d",
                                    cdir, x_addr, $signed(x_data), $signed(ez[x_addr]));
        end
        nz = nz + 1;
    end

    initial begin
        if (!$value$plusargs("CASE=%s", cdir)) begin $display("FAIL: thieu +CASE=<thu muc ca>"); $finish; end
        if (!$value$plusargs("STALL=%d", stall)) stall = 0;
        if (!$value$plusargs("SEED=%d", seed)) seed = 1;
        if (!$value$plusargs("RATE=%d", rate)) rate = 2;
        $readmemh($sformatf("%0s/img1.mem", cdir), img1);
        $readmemh($sformatf("%0s/img2.mem", cdir), img2);
        $readmemh($sformatf("%0s/feat_z.mem", cdir), ez);
        if ((^img1[0] === 1'bx) || (^img2[NPIX-1] === 1'bx) || (^ez[47] === 1'bx)) begin
            $display("FAIL tb_hs_engine %0s: khong doc duoc file .mem (sai duong dan +CASE?)", cdir); $finish;
        end
        fd = $fopen($sformatf("%0s/expected.txt", cdir), "r");
        exp_a = -1; exp_f = 0;
        if (fd != 0) begin r = $fscanf(fd, "f_q %d\nanomaly %d", exp_f, exp_a); $fclose(fd); end
        errs = 0; nz = 0; cyc = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk); start = 1'b1; @(posedge clk); start = 1'b0;
        while (!svm_done && cyc < 40_000_000) begin @(posedge clk); cyc = cyc + 1; end
        repeat (2) @(posedge clk);
        if (cyc >= 40_000_000) $display("FAIL tb_hs_engine %0s: het thoi gian (pha %0d)", cdir, phase);
        else if (nz != 48)      $display("FAIL tb_hs_engine %0s: %0d gia tri z (can 48)", cdir, nz);
        else if (errs != 0)     $display("FAIL tb_hs_engine %0s: %0d gia tri z sai", cdir, errs);
        else if (exp_a >= 0 && (f_out !== exp_f || anomaly !== exp_a[0]))
            $display("FAIL tb_hs_engine %0s: f_q RTL %0d co %0d, mong doi %0d co %0d", cdir, f_out, anomaly, exp_f, exp_a);
        else $display("PASS tb_hs_engine %0s: 48 z khop, f_q = %0d, bat thuong = %0d%0s | HS %0d chu ky = %0.2f ms o 100 MHz (RATE=%0d STALL=%0d%%), tong %0d chu ky, bao hoa %0d",
                      cdir, f_out, anomaly, exp_a >= 0 ? " (khop expected.txt)" : "", cycles, cycles / 100000.0,
                      rate, stall, cyc, n_sat);
        $finish;
    end
endmodule
