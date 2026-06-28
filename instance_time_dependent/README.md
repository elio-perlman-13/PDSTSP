# Time-Dependent Instance Generation

Thư mục này chứa các instance time-dependent được sinh từ các file gốc trong `../instance`.

Với mỗi file gốc `X.txt`, bộ dữ liệu mới gồm ba file:

- `X.txt`: instance đã chỉnh `Demand` và `Lw`.
- `X.vmax_ij.txt`: vận tốc cơ sở theo cạnh của xe tải.
- `X.theta_ijl.txt`: hệ số vận tốc theo cạnh và khung giờ.

Mô hình vận tốc xe tải dùng trong `tabubu_TimeDependent.cpp` là:

```text
v_ijl = theta_ijl * vmax_ij
```

Trong đó:

- `i, j`: hai đỉnh của cạnh có hướng `i -> j`.
- `l`: chỉ số khung giờ, từ `0` đến `11`.
- `vmax_ij`: vận tốc cơ sở của cạnh `i -> j`.
- `theta_ijl`: hệ số điều chỉnh vận tốc của cạnh `i -> j` trong khung giờ `l`.

## Quy Luật Sinh `X.txt`

File instance gốc được giữ nguyên cấu trúc, nhưng chỉnh hai trường `Demand` và `Lw`.

### Demand

Với mỗi khách hàng:

```text
if Demand <= 2:
    giữ nguyên Demand gốc
else:
    Demand = Uniform(1.5, 2.0)
```

Tức là các demand lớn hơn `2` được đưa về khoảng drone-carryable `(1.5, 2.0)`.

### Lw

Với mỗi instance có `n` khách hàng, `Lw` được gán xấp xỉ đều cho ba mức:

```text
1800 giây  = 30 phút
2700 giây  = 45 phút
3600 giây  = 60 phút
```

Tỷ lệ là xấp xỉ:

```text
1/3 khách nhận 1800
1/3 khách nhận 2700
1/3 khách nhận 3600
```

Nếu `n` không chia hết cho `3`, số lượng mỗi nhóm chênh nhau tối đa `1` khách. Các giá trị được shuffle ngẫu nhiên trong từng instance.

## Quy Luật Sinh `vmax_ij`

Với mỗi cặp đỉnh khác nhau `i != j`, sinh một vận tốc cơ sở đối xứng:

```text
vmax_ij = vmax_ji
vmax_ij = Uniform(0.7, 0.95) * 15.6
```

Đơn vị là `m/s`.

Do đó:

```text
vmax_ij ∈ [10.92, 14.82] m/s
```

Format file `X.vmax_ij.txt`:

```text
i j vmax_ij
0 1 12.345678
1 0 12.345678
...
```

Không ghi dòng `i i`.

## Quy Luật Sinh `theta_ijl`

Thời gian được chia thành `12` khung, mỗi khung dài `1` giờ và quay vòng theo ngày:

```text
l = 0, 1, ..., 11
```

Vector hệ số cơ sở:

```text
base_L = [0.8, 0.5, 0.7, 0.8, 0.9, 0.7,
          0.7, 0.9, 0.9, 0.7, 0.6, 0.5]
```

Với mỗi cạnh `i -> j` và khung giờ `l`:

```text
theta_ijl = round(Uniform(0.9, 1.0) * base_L[l], 2)
```

Format file `X.theta_ijl.txt`:

```text
i j l theta_ijl
0 1 0 0.77
0 1 1 0.48
...
```

Không ghi dòng `i i`.

## Tính Tái Lập

Quá trình sinh dữ liệu dùng seed deterministic theo tên file:

```text
seed = 20260626 + sum((index + 1) * ASCII(character))
```

Vì vậy, cùng một file gốc sẽ luôn sinh ra cùng dữ liệu nếu dùng cùng script/quy luật.

## Cách Chạy Solver

Ví dụ với instance `6.5.1.txt`:

```bash
./tabubu_TimeDependent \
  instance_time_dependent/6.5.1.txt \
  --truck-vmax-file=instance_time_dependent/6.5.1.vmax_ij.txt \
  --truck-theta-file=instance_time_dependent/6.5.1.theta_ijl.txt
```

Solver sẽ tính vận tốc xe tải trên từng cạnh và từng khung giờ theo:

```text
v_ijl = theta_ijl * vmax_ij
```
