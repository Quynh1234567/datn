// -----------------------------------------------------------------------------
// tb_hs_iter.v – kiểm hs_iter: phát lại deriv.mem NIT lần → so (u', v') vòng cuối
//   NIT = 1  → so với uv_iter1.mem  (gỡ lỗi nhanh, ~20 k chu kỳ)
//   NIT = 64 → so với uv_final.mem  (đủ 64 vòng, ~1,3–2,5 triệu chu kỳ)
//   iverilog -P tb_hs_iter.NIT=1 …   ·   +CASE=<case_XX>  +STALL=<%>  +RATE=<chu kỳ / gói, 2 = như SDRAM>
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_hs_iter;
    parameter integer NIT = 1;
    localparam integer NPIX = 19200;
    reg clk = 1'b0, rst_n = 1'b0, start = 1'b0;
    always #5 clk = ~clk;

    string cdir;                       // SystemVerilog string: chạy được cả Icarus (-g2012) và Vivado xsim
    integer stall, seed, rate, cyc, errs, op, ph;
    reg [63:0] pk   [0:NPIX-1];
    reg [25:0] expd [0:NPIX-1];

    reg         iv;
    integer     ip, tot;               // tot = số gói đã gửi (0 … NPIX·NIT)
    wire        in_ready, out_valid, done, busy;
    wire signed [12:0] ou, ov;
    wire [31:0] n_sat;
    wire [6:0]  iter;
    hs_iter #(.NIT(NIT)) dut (.clk(clk), .rst_n(rst_n), .start(start),
        .in_valid(iv), .in_ready(in_ready), .in_data(pk[ip]),
        .out_valid(out_valid), .out_u(ou), .out_v(ov), .done(done), .busy(busy), .n_sat(n_sat), .iter(iter));

    function go; input integer dummy; begin go = (($random(seed) & 127) >= (stall * 128) / 100); end endfunction

    always @(posedge clk) begin
        if (!rst_n) begin ip <= 0; tot <= 0; iv <= 1'b0; ph <= 0; end
        else begin
            ph <= (ph + 1 >= rate) ? 0 : ph + 1;
            if (iv && in_ready) begin
                tot <= tot + 1;
                ip  <= (ip == NPIX - 1) ? 0 : ip + 1;
            end
            if (!iv || in_ready)
                iv <= ((iv && in_ready ? tot + 1 : tot) < NPIX * NIT) && go(0) && (ph == 0);
        end
    end

    always @(posedge clk) if (rst_n && out_valid) begin
        if (op < NPIX && {ov, ou} !== expd[op]) begin
            errs = errs + 1;
            if (errs == 1)
                $display("FAIL tb_hs_iter %0s: diem %0d (y=%0d, x=%0d) RTL u=%0d v=%0d  mong doi u=%0d v=%0d",
                         cdir, op, op / 160, op % 160, ou, ov,
                         $signed(expd[op][12:0]), $signed(expd[op][25:13]));
        end
        op = op + 1;
    end

    initial begin
        if (!$value$plusargs("CASE=%s", cdir)) begin $display("FAIL: thieu +CASE=<thu muc ca>"); $finish; end
        if (!$value$plusargs("STALL=%d", stall)) stall = 0;
        if (!$value$plusargs("SEED=%d", seed)) seed = 1;
        if (!$value$plusargs("RATE=%d", rate)) rate = 1;
        $readmemh($sformatf("%0s/deriv.mem", cdir), pk);
        if (NIT == 1) $readmemh($sformatf("%0s/uv_iter1.mem", cdir), expd);
        else          $readmemh($sformatf("%0s/uv_final.mem", cdir), expd);
        if ((^pk[0] === 1'bx) || (^pk[NPIX-1] === 1'bx) || (^expd[0] === 1'bx) || (^expd[NPIX-1] === 1'bx)) begin
            $display("FAIL tb_hs_iter %0s: khong doc duoc file .mem (sai duong dan +CASE?)", cdir); $finish;
        end
        errs = 0; op = 0; cyc = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk); start = 1'b1; @(posedge clk); start = 1'b0;
        while (!done && cyc < 20_000_000) begin @(posedge clk); cyc = cyc + 1; end
        repeat (2) @(posedge clk);
        if (cyc >= 20_000_000)  $display("FAIL tb_hs_iter %0s: het thoi gian (vong %0d)", cdir, iter);
        else if (op != NPIX)     $display("FAIL tb_hs_iter %0s: %0d diem ra (can %0d)", cdir, op, NPIX);
        else if (errs != 0)      $display("FAIL tb_hs_iter %0s: %0d diem sai", cdir, errs);
        else $display("PASS tb_hs_iter %0s: N=%0d vong, u v khop %0s, %0d chu ky (%0.2f chu ky/diem/vong, RATE=%0d STALL=%0d%%), bao hoa %0d",
                      cdir, NIT, NIT == 1 ? "uv_iter1.mem" : "uv_final.mem", cyc, 1.0 * cyc / NPIX / NIT, rate, stall, n_sat);
        $finish;
    end
endmodule
