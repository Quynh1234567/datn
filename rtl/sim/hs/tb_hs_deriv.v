// -----------------------------------------------------------------------------
// tb_hs_deriv.v – kiểm hs_deriv với một ca véc-tơ: img1/img2.mem → so từng gói với deriv.mem
//   +CASE=<thư mục case_XX>   +STALL=<0…90> (% chu kỳ chèn chờ ngẫu nhiên ở đầu vào và đầu ra)
// In "PASS …" hoặc "FAIL …" (gói sai đầu tiên: y, x, trường, giá trị RTL và mong đợi).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_hs_deriv;
    localparam integer NPIX = 19200;
    reg clk = 1'b0, rst_n = 1'b0, start = 1'b0;
    always #5 clk = ~clk;

    string cdir;                       // SystemVerilog string: chạy được cả Icarus (-g2012) và Vivado xsim
    integer stall, seed, cyc, errs;
    reg [7:0]  img1 [0:NPIX-1];
    reg [7:0]  img2 [0:NPIX-1];
    reg [63:0] expd [0:NPIX-1];

    // ---------------------------------------------------------------- DUT
    reg         iv;
    integer     ip, op;
    wire        in_ready, out_valid, done, busy;
    wire [63:0] out_data;
    reg         out_ready;
    hs_deriv dut (.clk(clk), .rst_n(rst_n), .start(start),
                  .in_valid(iv), .in_ready(in_ready), .in_data({img2[ip], img1[ip]}),
                  .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
                  .done(done), .busy(busy));

    function go; input integer dummy; begin go = (($random(seed) & 127) >= (stall * 128) / 100); end endfunction

    always @(posedge clk) begin
        if (!rst_n) begin ip <= 0; iv <= 1'b0; out_ready <= 1'b0; end
        else begin
            if (iv && in_ready) ip <= ip + 1;
            if (!iv || in_ready) iv <= ((iv && in_ready ? ip + 1 : ip) < NPIX) && go(0);
            out_ready <= go(0);
        end
    end

    // ---------------------------------------------------------------- so sánh
    task show_field;
        input [63:0] got, exp;
        begin
            if (got[10:0]  != exp[10:0])  $display("   Ix4: RTL %0d  mong doi %0d", $signed(got[10:0]),  $signed(exp[10:0]));
            if (got[21:11] != exp[21:11]) $display("   Iy4: RTL %0d  mong doi %0d", $signed(got[21:11]), $signed(exp[21:11]));
            if (got[32:22] != exp[32:22]) $display("   It4: RTL %0d  mong doi %0d", $signed(got[32:22]), $signed(exp[32:22]));
            if (got[47:33] != exp[47:33]) $display("   a  : RTL %0d  mong doi %0d", $signed(got[47:33]), $signed(exp[47:33]));
            if (got[62:48] != exp[62:48]) $display("   b  : RTL %0d  mong doi %0d", $signed(got[62:48]), $signed(exp[62:48]));
        end
    endtask

    always @(posedge clk) if (rst_n && out_valid && out_ready) begin
        if (op >= NPIX) begin
            $display("FAIL tb_hs_deriv: thua goi thu %0d", op); $finish;
        end else if (out_data !== expd[op]) begin
            errs = errs + 1;
            if (errs == 1) begin
                $display("FAIL tb_hs_deriv %0s: goi %0d (y=%0d, x=%0d) RTL %h mong doi %h",
                         cdir, op, op / 160, op % 160, out_data, expd[op]);
                show_field(out_data, expd[op]);
            end
        end
        op = op + 1;
    end

    initial begin
        if (!$value$plusargs("CASE=%s", cdir)) begin $display("FAIL: thieu +CASE=<thu muc ca>"); $finish; end
        if (!$value$plusargs("STALL=%d", stall)) stall = 0;
        if (!$value$plusargs("SEED=%d", seed)) seed = 1;
        $readmemh($sformatf("%0s/img1.mem", cdir), img1);
        $readmemh($sformatf("%0s/img2.mem", cdir), img2);
        $readmemh($sformatf("%0s/deriv.mem", cdir), expd);
        if ((^img1[0] === 1'bx) || (^img2[NPIX-1] === 1'bx) || (^expd[0] === 1'bx) || (^expd[NPIX-1] === 1'bx)) begin
            $display("FAIL tb_hs_deriv %0s: khong doc duoc file .mem (sai duong dan +CASE?)", cdir); $finish;
        end
        errs = 0; op = 0; cyc = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk); start = 1'b1; @(posedge clk); start = 1'b0;
        while (!done && cyc < 4_000_000) begin @(posedge clk); cyc = cyc + 1; end
        repeat (2) @(posedge clk);
        if (cyc >= 4_000_000)      $display("FAIL tb_hs_deriv %0s: het thoi gian (%0d goi)", cdir, op);
        else if (op != NPIX)        $display("FAIL tb_hs_deriv %0s: %0d goi (can %0d)", cdir, op, NPIX);
        else if (errs != 0)         $display("FAIL tb_hs_deriv %0s: %0d goi sai", cdir, errs);
        else $display("PASS tb_hs_deriv %0s: 19200 goi khop deriv.mem, %0d chu ky (STALL=%0d%%)", cdir, cyc, stall);
        $finish;
    end
endmodule
