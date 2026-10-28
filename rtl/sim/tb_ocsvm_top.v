// -----------------------------------------------------------------------------
// tb_ocsvm_top.v - kiểm thử toàn hệ thống qua UART (gửi gói như ESP32, nhận gói trả lời)
//   Dùng tần số/baud nhỏ để mô phỏng nhanh: CLK_HZ = 1 MHz, BAUD = 100 kbaud (10 chu kỳ/bit)
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_ocsvm_top;
`include "ocsvm_params.vh"
    localparam integer CLK_HZ = 1_000_000, BAUD = 100_000, BIT_NS = 1_000_000_000 / BAUD;
    parameter integer NVEC = 5;                         // số véc-tơ gửi qua UART

    reg clk = 1'b0, rst_n = 1'b0;
    always #500 clk = ~clk;                             // 1 MHz
    reg  rx = 1'b1;
    wire tx, led_b, led_a;
    ocsvm_top #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) dut (.clk(clk), .rst_n(rst_n), .uart_rx(rx), .uart_tx(tx),
                                                    .led_busy_n(led_b), .led_anomaly_n(led_a));

    reg [DATA_W-1:0] tin  [0:N_TEST*DIM-1];
    reg [ACC_W-1:0]  texp [0:N_TEST-1];
    reg              tlab [0:N_TEST-1];

    task send_byte(input [7:0] b);
        integer k;
        begin
            rx = 1'b0; #(BIT_NS);
            for (k = 0; k < 8; k = k + 1) begin rx = b[k]; #(BIT_NS); end
            rx = 1'b1; #(BIT_NS);
        end
    endtask

    task recv_byte(output [7:0] b);
        integer k;
        begin
            @(negedge tx);                              // bit start
            #(BIT_NS + BIT_NS / 2);
            for (k = 0; k < 8; k = k + 1) begin b[k] = tx; #(BIT_NS); end
        end
    endtask

    integer v, j, r, errs;                              // j: luồng gửi, r: luồng nhận (KHÔNG dùng chung)
    reg [7:0] chk, rb, resp [0:8], rchk;
    reg [15:0] w;
    reg [47:0] f;
    initial begin
        $readmemh("test_input.mem", tin);
        $readmemh("test_expected_score.mem", texp);
        $readmemh("test_expected_label.mem", tlab);
        errs = 0;
        #(10_000) rst_n = 1'b1;
        #(10_000);
        for (v = 0; v < NVEC; v = v + 1) begin
            fork
                begin                                   // phát gói yêu cầu
                    send_byte(8'hAA); send_byte(8'h55); chk = 8'h00;
                    for (j = 0; j < DIM; j = j + 1) begin
                        w = tin[v*DIM+j];
                        send_byte(w[7:0]); send_byte(w[15:8]); chk = chk ^ w[7:0] ^ w[15:8];
                    end
                    send_byte(chk);
                end
                begin                                   // nhận gói trả lời
                    for (r = 0; r < 9; r = r + 1) begin recv_byte(rb); resp[r] = rb; end
                end
            join
            f = {resp[6], resp[5], resp[4], resp[3], resp[2], resp[1]};
            rchk = resp[1] ^ resp[2] ^ resp[3] ^ resp[4] ^ resp[5] ^ resp[6] ^ resp[7];
            if (resp[0] !== 8'h5A || rchk !== resp[8] || f !== texp[v][47:0] || resp[7][0] !== tlab[v]) begin
                errs = errs + 1;
                $display("SAI vector %0d: sync %h, f %0d (mong doi %0d), co %0d, checksum %h/%h",
                         v, resp[0], $signed(f), $signed(texp[v]), resp[7], resp[8], rchk);
            end else $display("vector %0d: f_q = %0d, co = %0d  OK", v, $signed(f), resp[7][0]);
            #(BIT_NS * 20);
        end
        if (errs == 0) $display("PASS: giao tiep UART + loi OC-SVM dung tren %0d vector", NVEC);
        else           $display("FAIL: %0d vector sai", errs);
        $finish;
    end

    initial begin                                       // chống treo: mỗi véc-tơ ≈ 11 ms mô phỏng
        #(64'd1_000_000 * 20 * NVEC + 64'd1_000_000);
        $display("FAIL: qua thoi gian - FPGA khong tra loi (kiem tra goi gui / checksum)");
        $finish;
    end
endmodule