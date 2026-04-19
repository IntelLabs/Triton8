# Triton Kernel Optimization Results — Speedup & TFLOPS

Platform: Intel Arc Pro B50 (Battlemage G21), 128 XVEs

## Level 2 Kernels

| Kernel                                    | GFLOP |  Base us |   Opt us | Speedup | Base TFLOPS | Opt TFLOPS |
|-------------------------------------------|------:|---------:|---------:|--------:|------------:|-----------:|
| 14_Gemm_Divide_Sum_Scaling                |  34.4 |    621.2 |    106.1 |   5.85x |        55.3 |      323.9 |
| 35_Conv2d_Subtract_HardSwish_MaxPool_Mish | 309.2 | 37,592.2 | 15,044.1 |   2.50x |         8.2 |       20.6 |
| 39_Gemm_Scale_BatchNorm                   | 137.4 |  3,609.4 |  3,047.4 |   1.18x |        38.1 |       45.1 |
| 45_Gemm_Sigmoid_LogSumExp                 | 103.1 |  2,478.7 |  2,363.2 |   1.05x |        41.6 |       43.6 |
| 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid   |  17.9 | 18,729.5 |  8,602.0 |   2.18x |         1.0 |        2.1 |
| 55_Matmul_MaxPool_Sum_Scale               | 274.9 | 15,449.0 | 12,552.8 |   1.23x |        17.8 |       21.9 |
| 61_ConvTranspose3d_ReLU_GroupNorm         | 278.2 | 16,521.8 | 14,014.4 |   1.18x |        16.8 |       19.9 |
| 68_Matmul_Min_Subtract                    |  68.7 |  3,597.6 |  2,840.6 |   1.27x |        19.1 |       24.2 |
| 70_Gemm_Sigmoid_Scaling_ResidualAdd       | 137.4 |  3,303.4 |  2,945.7 |   1.12x |        41.6 |       46.7 |
| 81_Gemm_Swish_Divide_Clamp_Tanh_Clamp     |  34.4 |    931.1 |    610.5 |   1.53x |        36.9 |       56.3 |
| 84_Gemm_BatchNorm_Scaling_Softmax         | 137.4 |  3,412.5 |  2,936.7 |   1.16x |        40.3 |       46.8 |
| 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh    |  34.4 |  1,016.8 |    632.6 |   1.61x |        33.8 |       54.3 |
| 99_Matmul_GELU_Softmax                    |  34.4 |    680.7 |    623.1 |   1.09x |        50.5 |       55.1 |

## Flash Attention (Forward)

| Config                     |   GFLOP |     Base us |    Opt us | Speedup | Base TFLOPS | Opt TFLOPS |
|----------------------------|--------:|------------:|----------:|--------:|------------:|-----------:|
| S=2048 (B=1, A=32, D=128)  |    68.7 |     3,588.5 |   1,930.0 |   1.86x |        19.1 |       35.6 |
| S=2048, A=71 (B=1, D=64)   |    76.2 |     3,847.7 |   2,522.8 |   1.53x |        19.8 |       30.2 |
| S=4096 (B=1, A=32, D=128)  |   274.9 |    16,294.5 |   8,468.2 |   1.92x |        16.9 |       32.5 |
| S=8192 (B=1, A=32, D=128)  | 1,099.5 |    71,474.6 |  45,921.7 |   1.56x |        15.4 |       23.9 |
| S=16384 (B=1, A=40, D=128) | 5,497.6 | 1,298,123.0 | 177,391.3 |   7.32x |         4.2 |       31.0 |

## Notes

- TFLOPS = GFLOP / (runtime_us * 1e-6) / 1e3
- FLOP counts are dominated by matmul/convolution (2*M*N*K for GEMM, 2*B*Cout*Osize*Cin*Ksize for conv)
- Convolution FLOP counts (kernels 35, 48, 61) are estimates for the conv op only, excluding pooling/norm/activation
- Flash Attention FLOPs use the standard 4*B*A*S*S*D formula (2 matmuls: Q@K^T and P@V, each 2MNK)
- Baseline is PyTorch eager (or unoptimized Triton for flash attention)
