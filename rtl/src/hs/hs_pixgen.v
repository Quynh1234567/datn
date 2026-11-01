// -----------------------------------------------------------------------------
// hs_pixgen.v – nguồn ảnh thử trên board (thay SDRAM/camera ở bước thử nghiệm):
//   A(x, y) = ((5x XOR 3y) + (x·y >> 3)) mod 256      i1 = A(x, y),  i2 = A(x + DX, y + DY)
// Phát luồng {i2, i1} theo hàng, 19 200 nhịp sau mỗi start; giống hệt hàm pattern() trong
// scripts/25_hs_board_test.py để PC tính được kết quả mong đợi.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps
module hs_pixgen #(
    parameter integer DX = 1,
    parameter integer DY = 0
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         valid,
    input  wire        ready,
    output wire [15:0] data
);
    reg [7:0] x;
    reg [6:0] y;
    function [7:0] pat; input [8:0] px; input [7:0] py;
        reg [15:0] m;
        begin
            m   = px * py;
            pat = ((px * 5) ^ (py * 3)) + (m >> 3);
        end
    endfunction
    assign data = {pat(x + DX, y + DY), pat(x, y)};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin valid <= 1'b0; x <= 0; y <= 0; end
        else if (start) begin valid <= 1'b1; x <= 0; y <= 0; end
        else if (valid && ready) begin
            if (x == 159) begin
                x <= 0;
                if (y == 119) begin y <= 0; valid <= 1'b0; end
                else y <= y + 1'b1;
            end else x <= x + 1'b1;
        end
    end
endmodule
