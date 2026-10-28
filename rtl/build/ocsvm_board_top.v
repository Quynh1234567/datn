// Sinh bởi scripts/15_export_rtl.py – top module cho Gowin EDA (P = 6, 27000000 Hz, 115200 baud)
module ocsvm_board_top (
    input  wire clk,
    input  wire rst_n,
    input  wire uart_rx,
    output wire uart_tx,
    output wire led_busy_n,
    output wire led_anomaly_n
);
    ocsvm_top #(
        .CLK_HZ(27000000), .BAUD(115200), .P(6),
        .SV_FILE("D:/New folder/ocsvm_anomaly_fpga/rtl/build/sv_lanes.mem"),
        .ALPHA_FILE("D:/New folder/ocsvm_anomaly_fpga/rtl/build/alpha.mem"),
        .LUT_FILE("D:/New folder/ocsvm_anomaly_fpga/rtl/build/exp_lut.mem")
    ) u_top (.clk(clk), .rst_n(rst_n), .uart_rx(uart_rx), .uart_tx(uart_tx),
             .led_busy_n(led_busy_n), .led_anomaly_n(led_anomaly_n));
endmodule
