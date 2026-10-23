# Lối tắt (Linux/macOS/Git-Bash). Windows: chạy trực tiếp các lệnh python tương ứng.
PY ?= python
JOBS ?= 8

.PHONY: all inspect features train evaluate export verify cport test smoke clean
all:       ; $(PY) scripts/run_pipeline.py --jobs $(JOBS)
inspect:   ; $(PY) scripts/00_inspect_dataset.py
features:  ; $(PY) scripts/01_extract_features.py --jobs $(JOBS)
train:     ; $(PY) scripts/02_train.py
evaluate:  ; $(PY) scripts/03_evaluate.py
export:    ; $(PY) scripts/04_export_fpga.py
verify:    ; $(PY) scripts/06_verify_mem_files.py
cport:     ; $(PY) scripts/07_check_c_port.py
test:      ; $(PY) -m unittest discover tests -v
smoke:
	$(PY) scripts/make_synthetic_dataset.py --out data/synthetic
	$(PY) scripts/run_pipeline.py --config configs/synthetic.yaml
	$(PY) scripts/06_verify_mem_files.py --dir outputs_synthetic/fpga
clean:     ; rm -rf outputs outputs_synthetic
