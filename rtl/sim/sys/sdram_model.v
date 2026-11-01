// sdram_model.v – mô hình hành vi SDR SDRAM 32 bit (2K hàng × 256 cột × 4 bank), CAS = 2, burst 1,
// đủ để chạy third_party/sdram.v trong mô phỏng. Dữ liệu đọc được lái từ sau cạnh thứ CL−1 tới hết
// cạnh thứ CL+1 (giống bus giữ giá trị trên chip thật). Chỉ dùng mô phỏng.
`timescale 1ns / 1ps
module sdram_model (
    input wire        clk, cke, cs_n, ras_n, cas_n, we_n,
    input wire [1:0]  ba,
    input wire [10:0] a,
    input wire [3:0]  dqm,
    inout wire [31:0] dq
);
    reg [31:0] mem [0:(1 << 21) - 1];
    reg [10:0] row [0:3];
    reg [31:0] dout;
    reg        doe = 1'b0;
    assign dq = doe ? dout : 32'bz;
    reg [20:0] rd_addr;
    reg [1:0]  rd_cnt = 0;
    wire [2:0] cmd = {ras_n, cas_n, we_n};
    integer i;
    always @(posedge clk) begin
        if (!cs_n) case (cmd)
            3'b011: row[ba] <= a;                                       // ACTIVE
            3'b101: begin rd_addr <= {ba, row[ba], a[7:0]}; rd_cnt <= 2'd1; end   // READ
            3'b100: begin                                               // WRITE
                for (i = 0; i < 4; i = i + 1)
                    if (!dqm[i]) mem[{ba, row[ba], a[7:0]}][8*i +: 8] <= dq[8*i +: 8];
            end
            default: ;
        endcase
        if (rd_cnt == 2'd1) begin rd_cnt <= 2'd2; #1 dout <= mem[rd_addr]; doe <= 1'b1; end
        else if (rd_cnt == 2'd2) rd_cnt <= 2'd3;
        else if (rd_cnt == 2'd3) begin rd_cnt <= 2'd0; #1 doe <= 1'b0; end
    end
endmodule
