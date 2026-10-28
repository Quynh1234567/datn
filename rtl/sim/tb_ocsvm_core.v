// -----------------------------------------------------------------------------
// tb_ocsvm_core.v – kiểm chứng lõi OC-SVM khớp TỪNG BIT với Golden Model
//   Đọc test_input.mem (N_TEST × DIM), so f_out với test_expected_score.mem và cờ với
//   test_expected_label.mem; so giá trị kernel véc-tơ 0 không cần (xem tb_kernel nếu gỡ lỗi).
//   Chạy: xem run_sim.bat / Makefile (iverilog -g2012)
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_ocsvm_core;
`include "ocsvm_params.vh"
    parameter integer P = 8;

    reg clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;                              // 100 MHz

    reg         x_we = 1'b0, start = 1'b0;
    reg  [7:0]  x_addr = 0;
    reg  [15:0] x_data = 0;
    wire        busy, done, anomaly;
    wire signed [63:0] f_out;

    ocsvm_core #(.P(P)) dut (.clk(clk), .rst_n(rst_n), .x_we(x_we), .x_addr(x_addr), .x_data(x_data),
                             .start(start), .busy(busy), .done(done), .f_out(f_out), .anomaly(anomaly));

    reg [DATA_W-1:0] tin  [0:N_TEST*DIM-1];
    reg [ACC_W-1:0]  texp [0:N_TEST-1];
    reg              tlab [0:N_TEST-1];
    integer v, j, errs, lab_errs, cyc, cyc_first, t0;
    reg signed [ACC_W-1:0] got;

    initial begin
        $readmemh("test_input.mem", tin);
        $readmemh("test_expected_score.mem", texp);
        $readmemh("test_expected_label.mem", tlab);
        errs = 0; lab_errs = 0; cyc_first = -1;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        for (v = 0; v < N_TEST; v = v + 1) begin
            for (j = 0; j < DIM; j = j + 1) begin         // nạp véc-tơ x
                @(negedge clk);
                x_we = 1'b1; x_addr = j; x_data = tin[v*DIM+j];
            end
            @(negedge clk) x_we = 1'b0;
            @(negedge clk) start = 1'b1;
            @(negedge clk) start = 1'b0;
            t0 = $time;
            @(posedge done);
            cyc = ($time - t0) / 10 + 1;
            if (cyc_first < 0) cyc_first = cyc;
            got = f_out[ACC_W-1:0];
            if (got !== $signed(texp[v])) begin
                errs = errs + 1;
                if (errs <= 5) $display("SAI vector %0d: RTL f = %0d, Golden f = %0d", v, got, $signed(texp[v]));
            end
            if (anomaly !== tlab[v]) lab_errs = lab_errs + 1;
            @(negedge clk);
        end
        $display("--------------------------------------------------------------");
        $display("%0d vector | sai f_q: %0d | sai co bat thuong: %0d | do tre = %0d chu ky (P = %0d)",
                 N_TEST, errs, lab_errs, cyc_first, P);
        if (errs == 0 && lab_errs == 0) $display("PASS: RTL khop tung bit voi Golden Model");
        else                            $display("FAIL");
        $finish;
    end

    initial begin                                        // chống treo
        #(64'd2_000_000_000);
        $display("FAIL: qua thoi gian mo phong");
        $finish;
    end
endmodule
