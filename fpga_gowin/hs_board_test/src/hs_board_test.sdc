// 27 MHz (thạch anh). Muốn xem có đạt 100 MHz không: đổi period thành 10 rồi Run P&R, đọc Timing Report.
create_clock -name clk -period 37.037 [get_ports {clk}]
