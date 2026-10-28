@echo off
rem Mo phong bang Vivado xsim – mo 'Vivado 2026.1 Tcl Shell' hoac them Vivado\bin vao PATH
call xvlog -sv -i . ../src/ocsvm_core.v ../sim/tb_ocsvm_core.v || exit /b 1
call xelab -generic_top "P=6" -debug typical tb_ocsvm_core -s tb_core || exit /b 1
call xsim tb_core -R
call xvlog -sv -i . ../src/uart_rx.v ../src/uart_tx.v ../src/ocsvm_core.v ../src/ocsvm_top.v ../sim/tb_ocsvm_top.v || exit /b 1
call xelab -debug typical tb_ocsvm_top -s tb_top || exit /b 1
call xsim tb_top -R
