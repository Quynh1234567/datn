# third_party/sdram.v

Bộ điều khiển SDRAM cho SDRAM 64 Mbit tích hợp trong GW2AR-18 (Tang Nano 20K), tác giả **nand2mario**,
lấy NGUYÊN VĂN từ <https://github.com/sipeed/TangNano-20K-example> (`nestang/src/sdram.v`),
giấy phép **GNU GPL v3** (xem `nestang/LICENSE` trong repo gốc). Đã chạy thật trên board trong dự án NESTang.

Đồ án chỉ dùng giao diện của nó (rd / wr / refresh / addr / din / dout32 / data_ready / busy) qua
`rtl/src/sys/sdram_arb.v`. Khi trích dẫn trong báo cáo: ghi rõ nguồn và giấy phép của file này.
