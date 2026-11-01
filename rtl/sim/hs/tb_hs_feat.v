// -----------------------------------------------------------------------------
// tb_hs_feat.v – kiểm hs_feat: luồng uv_final.mem → so |v| (mag.mem), tổng ô (cell_sums.mem),
// trung bình (feat_mean.mem), log (feat_log.mem), z (feat_z.mem) và các lần ghi x_we vào OC-SVM.
//   +CASE=<case_XX>  +MEM=<thư mục mem/ chứa log_lut, scaler_*>  +STALL=<%>
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_hs_feat;
    localparam integer NPIX = 19200;
    reg clk = 1'b0, rst_n = 1'b0, start = 1'b0;
    always #5 clk = ~clk;

    string cdir;
    integer stall, seed, cyc, errs, ip, mp;
    reg [25:0] uv   [0:NPIX-1];
    reg [12:0] emag [0:NPIX-1];
    reg [23:0] esum [0:47];
    reg [15:0] emean[0:47], elog[0:47], ez[0:47];
    reg [15:0] xw   [0:47];
    integer    nxw;

    reg iv;
    wire mag_valid, res_valid, x_we, done;
    wire [12:0] mag;
    wire [5:0] res_d;
    wire signed [23:0] res_sum;
    wire signed [15:0] res_mean, res_log, res_z;
    wire [7:0]  x_addr;
    wire [15:0] x_data;

    hs_feat #(.LUT_FILE("log_lut.mem"), .MEAN_FILE("scaler_mean_q.mem"), .INV_FILE("scaler_inv_q.mem")) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .in_valid(iv),
        .in_u($signed(uv[ip][12:0])), .in_v($signed(uv[ip][25:13])),
        .mag_valid(mag_valid), .mag(mag), .res_valid(res_valid), .res_d(res_d), .res_sum(res_sum),
        .res_mean(res_mean), .res_log(res_log), .res_z(res_z),
        .x_we(x_we), .x_addr(x_addr), .x_data(x_data), .done(done));

    function go; input integer dummy; begin go = (($random(seed) & 127) >= (stall * 128) / 100); end endfunction
    reg running;
    always @(posedge clk) begin
        if (!rst_n || !running) begin ip <= 0; iv <= 1'b0; end
        else begin
            if (iv) ip <= ip + 1;                       // FEAT luôn nhận (không có áp lực ngược)
            iv <= ((iv ? ip + 1 : ip) < NPIX) && go(0);
        end
    end

    task fail1; input [8*64-1:0] what; input integer i; input integer got, exp;
        begin
            errs = errs + 1;
            if (errs <= 5) $display("FAIL tb_hs_feat %0s: %0s[%0d] RTL %0d mong doi %0d", cdir, what, i, got, exp);
        end
    endtask

    always @(posedge clk) if (rst_n) begin
        if (mag_valid) begin
            if (mag !== emag[mp]) fail1("mag", mp, mag, emag[mp]);
            mp = mp + 1;
        end
        if (res_valid) begin
            if (res_sum  !== $signed(esum[res_d]))  fail1("cell_sums", res_d, res_sum,  $signed(esum[res_d]));
            if (res_mean !== $signed(emean[res_d])) fail1("feat_mean", res_d, res_mean, $signed(emean[res_d]));
            if (res_log  !== $signed(elog[res_d]))  fail1("feat_log",  res_d, res_log,  $signed(elog[res_d]));
            if (res_z    !== $signed(ez[res_d]))    fail1("feat_z",    res_d, res_z,    $signed(ez[res_d]));
        end
        if (x_we) begin
            if (x_addr >= 48 || x_data !== ez[x_addr]) fail1("x_we", x_addr, $signed(x_data), $signed(ez[x_addr]));
            nxw = nxw + 1;
        end
    end

    initial begin
        if (!$value$plusargs("CASE=%s", cdir)) begin $display("FAIL: thieu +CASE=<thu muc ca>"); $finish; end
        if (!$value$plusargs("STALL=%d", stall)) stall = 0;
        if (!$value$plusargs("SEED=%d", seed)) seed = 1;
        $readmemh($sformatf("%0s/uv_final.mem", cdir), uv);
        $readmemh($sformatf("%0s/mag.mem", cdir), emag);
        $readmemh($sformatf("%0s/cell_sums.mem", cdir), esum);
        $readmemh($sformatf("%0s/feat_mean.mem", cdir), emean);
        $readmemh($sformatf("%0s/feat_log.mem", cdir), elog);
        $readmemh($sformatf("%0s/feat_z.mem", cdir), ez);
        if ((^uv[0] === 1'bx) || (^uv[NPIX-1] === 1'bx) || (^emag[NPIX-1] === 1'bx) || (^esum[47] === 1'bx) || (^emean[47] === 1'bx) || (^elog[47] === 1'bx) || (^ez[47] === 1'bx)) begin
            $display("FAIL tb_hs_feat %0s: khong doc duoc file .mem (sai duong dan +CASE?)", cdir); $finish;
        end
        errs = 0; mp = 0; nxw = 0; cyc = 0; running = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk); start = 1'b1; @(posedge clk); start = 1'b0; running = 1'b1;
        while (!done && cyc < 2_000_000) begin @(posedge clk); cyc = cyc + 1; end
        repeat (3) @(posedge clk);
        if (cyc >= 2_000_000)     $display("FAIL tb_hs_feat %0s: het thoi gian (%0d diem |v|)", cdir, mp);
        else if (mp != NPIX)       $display("FAIL tb_hs_feat %0s: %0d diem |v| (can %0d)", cdir, mp, NPIX);
        else if (nxw != 48)        $display("FAIL tb_hs_feat %0s: %0d lan ghi x_we (can 48)", cdir, nxw);
        else if (errs != 0)        $display("FAIL tb_hs_feat %0s: %0d loi", cdir, errs);
        else $display("PASS tb_hs_feat %0s: |v|, 48 tong o, mean, log, z va 48 lan ghi OC-SVM khop, %0d chu ky",
                      cdir, cyc);
        $finish;
    end
endmodule
