// -----------------------------------------------------------------------------
// uart_rx.v – bộ thu UART 8N1, lấy mẫu giữa bit, đồng bộ 2 flip-flop chống mất ổn định
// -----------------------------------------------------------------------------
module uart_rx #(
    parameter integer CLKS_PER_BIT = 234          // = round(f_clk / baud), vd 27 MHz / 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,                         // chân RX (mức nghỉ = 1)
    output reg  [7:0] data,                       // byte nhận được
    output reg        valid                       // xung 1 chu kỳ khi có byte mới
);
    localparam integer CW = $clog2(CLKS_PER_BIT + 1);
    localparam [1:0] S_IDLE = 2'd0, S_START = 2'd1, S_DATA = 2'd2, S_STOP = 2'd3;

    reg rx_m, rx_s;                               // đồng bộ hai tầng
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin rx_m <= 1'b1; rx_s <= 1'b1; end
        else        begin rx_m <= rx;   rx_s <= rx_m; end

    reg [1:0]    state;
    reg [CW-1:0] cnt;
    reg [2:0]    bit_idx;
    reg [7:0]    shreg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; cnt <= 0; bit_idx <= 0; shreg <= 8'd0; data <= 8'd0; valid <= 1'b0;
        end else begin
            valid <= 1'b0;
            case (state)
                S_IDLE: if (!rx_s) begin state <= S_START; cnt <= 0; end
                S_START:                                  // tới giữa bit start
                    if (cnt == CLKS_PER_BIT / 2 - 1) begin
                        cnt <= 0;
                        if (!rx_s) begin state <= S_DATA; bit_idx <= 0; end
                        else       state <= S_IDLE;       // nhiễu → bỏ
                    end else cnt <= cnt + 1'b1;
                S_DATA:
                    if (cnt == CLKS_PER_BIT - 1) begin
                        cnt <= 0;
                        shreg <= {rx_s, shreg[7:1]};      // LSB trước
                        if (bit_idx == 3'd7) state <= S_STOP;
                        else bit_idx <= bit_idx + 1'b1;
                    end else cnt <= cnt + 1'b1;
                S_STOP:
                    if (cnt == CLKS_PER_BIT - 1) begin
                        cnt <= 0; state <= S_IDLE;
                        if (rx_s) begin data <= shreg; valid <= 1'b1; end   // bit stop hợp lệ
                    end else cnt <= cnt + 1'b1;
            endcase
        end
    end
endmodule
