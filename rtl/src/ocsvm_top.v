// -----------------------------------------------------------------------------
// ocsvm_top.v – bộ tăng tốc OC-SVM giao tiếp UART
//
//   Gói vào : 0xAA 0x55 | DIM × int16 little-endian | XOR các byte dữ liệu   (2·DIM + 3 byte)
//   Gói ra  : 0x5A | f_q 6 byte little-endian | cờ bất thường | XOR (f_q + cờ)  (9 byte)
//   FSM     : IDLE → RX_HDR → RX_DATA → CHECK → COMPUTE → SEND → IDLE
//             sai checksum / quá thời gian chờ byte → IDLE, không trả lời
// -----------------------------------------------------------------------------
module ocsvm_top #(
    parameter integer CLK_HZ     = 27_000_000,       // Tang Nano 9K/20K: thạch anh 27 MHz
    parameter integer BAUD       = 115_200,
    parameter integer P          = 8,
    parameter         SV_FILE    = "sv_lanes.mem",
    parameter         ALPHA_FILE = "alpha.mem",
    parameter         LUT_FILE   = "exp_lut.mem"
)(
    input  wire clk,
    input  wire rst_n,
    input  wire uart_rx,
    output wire uart_tx,
    output wire led_busy_n,                          // LED tích cực mức thấp (Tang Nano)
    output wire led_anomaly_n
);
`include "ocsvm_params.vh"

    localparam integer CLKS_PER_BIT = (CLK_HZ + BAUD / 2) / BAUD;
    localparam integer TIMEOUT      = CLKS_PER_BIT * 10 * 100;     // ≈ thời gian 100 byte
    localparam integer TW           = $clog2(TIMEOUT + 1);
    localparam integer NBYTES       = 2 * DIM;
    localparam integer ACC_BYTES    = 6;

    // ---------------------------------------------------------------- UART
    wire [7:0] rx_data;
    wire       rx_valid;
    reg  [7:0] tx_data;
    reg        tx_send;
    wire       tx_busy;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (.clk(clk), .rst_n(rst_n), .rx(uart_rx),
                                                 .data(rx_data), .valid(rx_valid));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (.clk(clk), .rst_n(rst_n), .data(tx_data),
                                                 .send(tx_send), .tx(uart_tx), .busy(tx_busy));

    // ---------------------------------------------------------------- lõi OC-SVM
    reg         x_we, core_start;
    reg  [7:0]  x_addr;
    reg  [15:0] x_data;
    wire        core_busy, core_done, core_anom;
    wire signed [63:0] core_f;
    ocsvm_core #(.P(P), .SV_FILE(SV_FILE), .ALPHA_FILE(ALPHA_FILE), .LUT_FILE(LUT_FILE)) u_core (
        .clk(clk), .rst_n(rst_n), .x_we(x_we), .x_addr(x_addr), .x_data(x_data),
        .start(core_start), .busy(core_busy), .done(core_done), .f_out(core_f), .anomaly(core_anom));

    // ---------------------------------------------------------------- FSM
    localparam [2:0] S_IDLE = 3'd0, S_HDR = 3'd1, S_DATA = 3'd2, S_CHECK = 3'd3,
                     S_COMP = 3'd4, S_SEND = 3'd5, S_WAIT = 3'd6;
    reg [2:0]    state;
    reg [7:0]    byte_cnt;
    reg [7:0]    lo_byte;
    reg [7:0]    xor_rx;
    reg [TW-1:0] tmo;
    reg [63:0]   f_lat;
    reg          flag_lat, last_flag;
    reg [3:0]    tx_idx;
    reg [7:0]    xor_tx;

    // byte thứ tx_idx của gói trả lời
    reg [7:0] resp_byte;
    always @* begin
        case (tx_idx)
            4'd0:    resp_byte = 8'h5A;
            4'd1:    resp_byte = f_lat[7:0];
            4'd2:    resp_byte = f_lat[15:8];
            4'd3:    resp_byte = f_lat[23:16];
            4'd4:    resp_byte = f_lat[31:24];
            4'd5:    resp_byte = f_lat[39:32];
            4'd6:    resp_byte = f_lat[47:40];
            4'd7:    resp_byte = {7'd0, flag_lat};
            default: resp_byte = xor_tx;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; byte_cnt <= 0; lo_byte <= 0; xor_rx <= 0; tmo <= 0;
            x_we <= 1'b0; x_addr <= 0; x_data <= 0; core_start <= 1'b0;
            f_lat <= 0; flag_lat <= 1'b0; last_flag <= 1'b0; tx_idx <= 0; xor_tx <= 0;
            tx_send <= 1'b0; tx_data <= 0;
        end else begin
            x_we <= 1'b0; core_start <= 1'b0; tx_send <= 1'b0;
            // quá thời gian chờ byte trong lúc đang nhận gói
            if ((state == S_HDR || state == S_DATA || state == S_CHECK) && !rx_valid) begin
                if (tmo == TIMEOUT) state <= S_IDLE;
                else tmo <= tmo + 1'b1;
            end else tmo <= 0;

            case (state)
                S_IDLE: if (rx_valid && rx_data == 8'hAA) state <= S_HDR;
                S_HDR:  if (rx_valid) begin
                            if (rx_data == 8'h55) begin state <= S_DATA; byte_cnt <= 0; xor_rx <= 0; end
                            else if (rx_data != 8'hAA) state <= S_IDLE;
                        end
                S_DATA: if (rx_valid) begin
                            xor_rx <= xor_rx ^ rx_data;
                            if (!byte_cnt[0]) lo_byte <= rx_data;
                            else begin
                                x_we   <= 1'b1;
                                x_addr <= byte_cnt[7:1];
                                x_data <= {rx_data, lo_byte};
                            end
                            if (byte_cnt == NBYTES - 1) state <= S_CHECK;
                            byte_cnt <= byte_cnt + 1'b1;
                        end
                S_CHECK: if (rx_valid) begin
                            if (rx_data == xor_rx) begin state <= S_COMP; core_start <= 1'b1; end
                            else state <= S_IDLE;
                        end
                S_COMP: if (core_done) begin
                            f_lat <= core_f; flag_lat <= core_anom; last_flag <= core_anom;
                            tx_idx <= 0; xor_tx <= 0; state <= S_SEND;
                        end
                S_SEND: if (!tx_busy && !tx_send) begin
                            tx_data <= resp_byte; tx_send <= 1'b1;
                            if (tx_idx >= 4'd1 && tx_idx <= 4'd7) xor_tx <= xor_tx ^ resp_byte;
                            state <= S_WAIT;
                        end
                S_WAIT: if (tx_busy) begin                  // bộ phát đã nhận byte
                            if (tx_idx == 4'd8) state <= S_IDLE;
                            else begin tx_idx <= tx_idx + 1'b1; state <= S_SEND; end
                        end
                default: state <= S_IDLE;
            endcase
        end
    end

    assign led_busy_n    = ~(state != S_IDLE);
    assign led_anomaly_n = ~last_flag;
endmodule
