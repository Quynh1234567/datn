// -----------------------------------------------------------------------------
// tb_hs_board_test.v – mô phỏng đúng top nạp board (hs_board_test) và giải mã dòng UART đầu tiên.
// Dòng in ra phải TRÙNG dòng mà board gửi qua cổng COM, và trùng dòng mong đợi của
// scripts/25_hs_board_test.py (trừ trường C nếu không chạy mô phỏng).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_hs_board_test;
    reg clk = 1'b0;
    always #18.5 clk = ~clk;                         // 27 MHz
    wire tx, led0, led1;
    hs_board_test dut (.clk(clk), .uart_tx(tx), .led_busy_n(led0), .led_anomaly_n(led1));
    localparam real BIT_NS = 37.0 * 234;          // 234 chu kỳ / bit
    reg [7:0] b;
    integer i, n;
    reg [8*400-1:0] line;
    integer nl = 0;
    initial begin
        n = 0; line = 0;
        forever begin
            @(negedge tx);                            // bit start
            #(BIT_NS * 1.5);
            for (i = 0; i < 8; i = i + 1) begin b[i] = tx; #(BIT_NS); end
            if (b == 8'h0A) begin
                $display("UART: %0s", line);
                line = 0; nl = nl + 1;
                if (nl == 3) $finish;                      // HS, ZV, DB
            end else if (b != 8'h0D) line = {line[8*399-1:0], b};
            n = n + 1;
        end
    end
    initial begin #900_000_000; $display("FAIL: het thoi gian"); $finish; end
endmodule
