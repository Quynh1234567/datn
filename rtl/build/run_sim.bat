@echo off
rem Mo phong RTL bang Icarus Verilog (https://bleyer.org/icarus/)
iverilog -g2012 -I . -P tb_ocsvm_core.P=6 -o core.vvp ../src/ocsvm_core.v ../sim/tb_ocsvm_core.v || exit /b 1
vvp core.vvp
iverilog -g2012 -I . -o top.vvp ../src/uart_rx.v ../src/uart_tx.v ../src/ocsvm_core.v ../src/ocsvm_top.v ../sim/tb_ocsvm_top.v || exit /b 1
vvp top.vvp
