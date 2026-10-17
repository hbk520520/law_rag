import torch
import triton
import triton.language as tl
import os

# ==========================================
# 核心 Kernel
# ==========================================
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. 找座位（获取当前实例 ID）
    pid = tl.program_id(axis=0)

    # 2. 算范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3. 拉安全网（防止最后一块越界）
    mask = offsets < n_elements

    # [🌟 破坏测试点 1：解开下方注释，只有在 TRITON_INTERPRET=1 下才能看到完整输出]
    # print(f"方阵 {pid} 负责的卷子编号: {offsets}")

    # 4. 干活
    # [🌟 破坏测试点 2：你可以故意把 mask=mask 删掉，看看会发生什么]
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

# ==========================================
# CPU 端调度函数
# ==========================================
def run_add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()

    # 定义 BLOCK_SIZE
    BLOCK_SIZE = 1024

    # 计算需要几个方阵 (grid)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 发射！
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

# ==========================================
# 测试入口
# ==========================================
if __name__ == "__main__":
    # 故意选一个 2500 这种不是 1024 倍数的大小，必产生尾块（0~1023, 1024~2047, 2048~2500越界区）
    N = 2500
    torch.manual_seed(42)
    x = torch.rand(N, device='cuda')
    y = torch.rand(N, device='cuda')

    print(f"数据量 N={N}，开始执行 Triton Kernel...")
    out_triton = run_add(x, y)
    out_torch = x + y

    # 验证正确性
    if torch.allclose(out_triton, out_torch):
        print("✅ 恭喜！Triton 算出的结果和原生 PyTorch 一模一样！")
    else:
        print("❌ 结果不一致！")
