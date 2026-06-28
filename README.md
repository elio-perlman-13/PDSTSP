# PDSTSP
File chạy thuật toán chính: tabubu.cpp
File chạy thuật toán so sánh với Raj: others/tabubu_raj.cpp
File chạy thuật toán so sánh với Saleu & Dell Amico: others/tabubu_saleu.cpp
File chạy thuật toán với truck cho phép đi 1 trip: tabubu_truck1trip.cpp
File chạy thuật toán với thời gian vận tốc của truck thay đổi theo từng cạnh: tabubu_time.cpp

Bộ test gốc của bài toán: instance
Bộ test của Raj: instances/min-time
Bộ test của Saleu: saleu-2022

Plot lịch sử hàm mục tiêu qua các iteration của lời giải trong output_solution_best.txt: plot_iteration.py
Plot đường đi của các truck/drone trong lời giải: plot.py

Sinh velocity matrix (từ od_speed_kmh.csv và od_distance_m.csv): create_velocity_matrix.py
Sinh instance cho tabubu_time.cpp: create_hanoi_instance.py

## Cấu hình thực nghiệm tham số thuật toán

Trong quá trình thực nghiệm các tham số thuật toán, capacity của mỗi truck được cố định ở mức **1000 kg**. Các thông số của drone, bao gồm capacity, năng lượng, vận tốc bay, vận tốc cất cánh, vận tốc hạ cánh, độ cao bay và mô hình công suất, được sử dụng theo cấu hình trong file `drone_config.json`.
