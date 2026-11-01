// gowin_rpll_stub.v – mô hình rPLL của Gowin CHỈ DÙNG MÔ PHỎNG: CLKOUT 54 MHz, CLKOUTP lệch 180°.
`timescale 1ns / 1ps
module rPLL #(
    parameter FCLKIN = "27", parameter DYN_IDIV_SEL = "false", parameter integer IDIV_SEL = 0,
    parameter DYN_FBDIV_SEL = "false", parameter integer FBDIV_SEL = 1, parameter DYN_ODIV_SEL = "false",
    parameter integer ODIV_SEL = 16, parameter PSDA_SEL = "1000", parameter DYN_DA_EN = "false",
    parameter DUTYDA_SEL = "1000", parameter CLKOUT_FT_DIR = 1'b1, parameter CLKOUTP_FT_DIR = 1'b1,
    parameter integer CLKOUT_DLY_STEP = 0, parameter integer CLKOUTP_DLY_STEP = 0, parameter CLKFB_SEL = "internal",
    parameter CLKOUT_BYPASS = "false", parameter CLKOUTP_BYPASS = "false", parameter CLKOUTD_BYPASS = "false",
    parameter integer DYN_SDIV_SEL = 2, parameter CLKOUTD_SRC = "CLKOUT", parameter CLKOUTD3_SRC = "CLKOUT",
    parameter DEVICE = "GW2AR-18C"
)(
    output reg CLKOUT = 1'b0, output wire LOCK, output wire CLKOUTP, output wire CLKOUTD, output wire CLKOUTD3,
    input RESET, input RESET_P, input CLKIN, input CLKFB,
    input [5:0] FBDSEL, input [5:0] IDSEL, input [5:0] ODSEL, input [3:0] PSDA, input [3:0] DUTYDA, input [3:0] FDLY
);
    localparam real HALF = 1000.0 / 54.0 / 2.0;     // ns
    always #(HALF) CLKOUT = ~CLKOUT;
    assign CLKOUTP = ~CLKOUT;
    assign CLKOUTD = CLKOUT;
    assign CLKOUTD3 = CLKOUT;
    reg lk = 1'b0;
    initial #500 lk = 1'b1;
    assign LOCK = lk;
endmodule
