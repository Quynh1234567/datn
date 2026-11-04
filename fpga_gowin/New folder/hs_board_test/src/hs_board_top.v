// hs_board_top.v – sinh bởi scripts/25_hs_board_test.py: top Gowin, đường dẫn .mem tuyệt đối
module hs_board_top (input wire clk, output wire uart_tx, output wire led_busy_n, output wire led_anomaly_n);
    hs_board_test #(
        .LUT_FILE("D:/datn/ocsvm_anomaly_fpga/fpga_gowin/hs_board_test/src/log_lut.mem"), .MEAN_FILE("D:/datn/ocsvm_anomaly_fpga/fpga_gowin/hs_board_test/src/scaler_mean_q.mem"),
        .INV_FILE("D:/datn/ocsvm_anomaly_fpga/fpga_gowin/hs_board_test/src/scaler_inv_q.mem"), .SV_FILE("D:/datn/ocsvm_anomaly_fpga/fpga_gowin/hs_board_test/src/sv_lanes.mem"),
        .ALPHA_FILE("D:/datn/ocsvm_anomaly_fpga/fpga_gowin/hs_board_test/src/alpha.mem"), .EXP_FILE("D:/datn/ocsvm_anomaly_fpga/fpga_gowin/hs_board_test/src/exp_lut.mem")
    ) u (.clk(clk), .uart_tx(uart_tx), .led_busy_n(led_busy_n), .led_anomaly_n(led_anomaly_n));
endmodule
