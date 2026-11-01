// -----------------------------------------------------------------------------
// tb_hs_system.v – mô phỏng toàn hệ thống demo 1: SPI (như ESP32) → SDRAM (mô hình) → HS → OC-SVM → UART
//   +CASE=<case_XX>   gửi img1 (seq 1, RESET_PAIR) rồi img2 (seq 2) → so f_q, cờ, Σz với expected.txt,
//                     feat_z.mem; in các dòng UART (SD … / HS …) giống hệt board.
//   +SCK_NS=<nửa chu kỳ SCK ns> (mặc định 50 = 10 MHz)
// Biên dịch: rtl/src/hs/*.v rtl/src/sys/*.v rtl/src/sys/third_party/sdram.v rtl/src/ocsvm_core.v
//            rtl/src/uart_tx.v + gowin_rpll_stub.v sdram_model.v tb_hs_system.v, -I hs_params, -I rtl/build
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_hs_system;
    reg clk27 = 1'b0;
    always #18.518 clk27 = ~clk27;
    reg sck = 1'b0, mosi = 1'b0, cs_n = 1'b1;
    wire tx;
    wire [5:0] led;
    wire sd_clk, sd_cke, sd_cs, sd_cas, sd_ras, sd_we;
    wire [31:0] sd_dq;
    wire [10:0] sd_a;
    wire [1:0]  sd_ba;
    wire [3:0]  sd_dqm;
    hs_system_top dut (.clk(clk27), .spi_sck(sck), .spi_mosi(mosi), .spi_cs_n(cs_n), .uart_tx(tx), .led(led),
        .O_sdram_clk(sd_clk), .O_sdram_cke(sd_cke), .O_sdram_cs_n(sd_cs), .O_sdram_cas_n(sd_cas),
        .O_sdram_ras_n(sd_ras), .O_sdram_wen_n(sd_we), .IO_sdram_dq(sd_dq), .O_sdram_addr(sd_a),
        .O_sdram_ba(sd_ba), .O_sdram_dqm(sd_dqm));
    sdram_model u_mem (.clk(sd_clk), .cke(sd_cke), .cs_n(sd_cs), .ras_n(sd_ras), .cas_n(sd_cas), .we_n(sd_we),
                       .ba(sd_ba), .a(sd_a), .dqm(sd_dqm), .dq(sd_dq));

    string cdir;
    integer half, fd, r, i, k, nres, errs;
    reg [7:0]  img1 [0:19199];
    reg [7:0]  img2 [0:19199];
    reg [15:0] ez [0:47];
    reg signed [63:0] exp_f;
    integer exp_a;
    reg [15:0] exp_z;

    task spi_byte(input [7:0] b);
        integer j;
        begin
            for (j = 7; j >= 0; j = j - 1) begin
                mosi = b[j]; #(half); sck = 1'b1; #(half); sck = 1'b0;
            end
        end
    endtask
    task send_frame(input integer which, input [7:0] seq, input [7:0] flags);
        integer c, n, p;
        begin
            for (c = 0; c < 5; c = c + 1) begin
                n = (c < 4) ? 4096 : 19200 - 4 * 4096;
                cs_n = 1'b0; #(half);
                spi_byte(8'hA5); spi_byte(flags); spi_byte(c); spi_byte(seq);
                for (p = 0; p < n; p = p + 1) spi_byte(which == 1 ? img1[c * 4096 + p] : img2[c * 4096 + p]);
                #(half); cs_n = 1'b1; #2000;
            end
        end
    endtask

    // ---------------------------------------------------------------- giải mã UART
    localparam real BIT_NS = 468.0 * 1000.0 / 54.0;
    reg [8*100-1:0] line;
    reg [7:0] b;
    integer bi;
    initial begin
        line = 0;
        forever begin
            @(negedge tx);
            #(BIT_NS * 1.5);
            for (bi = 0; bi < 8; bi = bi + 1) begin b[bi] = tx; #(BIT_NS); end
            if (b == 8'h0A) begin $display("UART: %0s", line); line = 0; end
            else if (b != 8'h0D) line = {line[8*99-1:0], b};
        end
    end

    // ---------------------------------------------------------------- kiểm tra kết quả
    always @(posedge dut.sclk) if (dut.svm_done) begin
        nres = nres + 1;
        $display("KQ: seq=%0d f_q=%0d co=%0d Σz=%h | HS %0d chu ky = %0.2f ms @54 MHz, tu luc du anh %0d chu ky, drop %0d",
                 dut.e_seq, dut.f_out, dut.anomaly, dut.zsum, dut.cycles, dut.cycles / 54000.0, dut.lat, dut.n_drop);
        if (exp_a >= 0 && (dut.f_out !== exp_f || dut.anomaly !== exp_a[0])) errs = errs + 1;
        if (dut.zsum !== exp_z) errs = errs + 1;
    end

    initial begin
        if (!$value$plusargs("CASE=%s", cdir)) begin $display("FAIL: thieu +CASE"); $finish; end
        if (!$value$plusargs("SCK_NS=%d", half)) half = 50;
        $readmemh($sformatf("%0s/img1.mem", cdir), img1);
        $readmemh($sformatf("%0s/img2.mem", cdir), img2);
        $readmemh($sformatf("%0s/feat_z.mem", cdir), ez);
        if ((^img1[0] === 1'bx) || (^img2[19199] === 1'bx) || (^ez[47] === 1'bx)) begin
            $display("FAIL tb_hs_system: khong doc duoc file .mem"); $finish;
        end
        exp_z = 0; for (k = 0; k < 48; k = k + 1) exp_z = exp_z + ez[k];
        exp_a = -1; fd = $fopen($sformatf("%0s/expected.txt", cdir), "r");
        if (fd != 0) begin r = $fscanf(fd, "f_q %d\nanomaly %d", exp_f, exp_a); $fclose(fd); end
        nres = 0; errs = 0;
        wait (dut.tst == 2'd3);
        #20000;
        $display("SDRAM tu kiem tra: %0d tu sai / %0d", dut.sd_err, dut.sd_words);
        send_frame(1, 8'd1, 8'h01);
        #50000;
        send_frame(2, 8'd2, 8'h00);
        wait (nres == 1);
        #(BIT_NS * 10 * 90);                                   // chờ dòng UART
        if (dut.sd_err != 0) $display("FAIL tb_hs_system: SDRAM tu kiem tra sai %0d tu", dut.sd_err);
        else if (errs != 0)  $display("FAIL tb_hs_system %0s: ket qua khac Golden (mong doi f_q %0d co %0d Σz %h)", cdir, exp_f, exp_a, exp_z);
        else $display("PASS tb_hs_system %0s: SPI → SDRAM → HS → OC-SVM khop Golden, HS %0d chu ky = %0.2f ms o 54 MHz",
                      cdir, dut.q_cyc, dut.q_cyc / 54000.0);
        $finish;
    end
    initial begin #400_000_000; $display("FAIL tb_hs_system: het thoi gian"); $finish; end
endmodule
