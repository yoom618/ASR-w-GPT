import os
import torch
# from mamba import GatedMamba
from mamba_ssm_custom import Mamba, GatedMamba

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'
torch.cuda.reset_peak_memory_stats(device=None)

# Create a Mamba object
batch_size = 7
seq_len = 10
input_dim = 32
output_dim = 64

# mamba = GatedMamba(
#         num_layers=1,
#         d_input=input_dim,
#         d_output=output_dim,
#         d_model=16,
#         d_state=16,
#         d_discr=None,
#         ker_size=4,
#         parallel=False,
#         residual=True,
# ).to("cuda")

for i in range(4,8):
    # i를 이진법으로 변환
    i = bin(i)[2:].zfill(3)
    print(*[bool(int(j)) for j in i])

    mamba = GatedMamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=128, # Model dimension d_model
        d_input=input_dim,
        d_output=output_dim,
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
        use_fast_path=False,  # Use fast path for block expansion
        timevariant_dt=bool(int(i[0])),
        timevariant_B=bool(int(i[1])),
        timevariant_C=bool(int(i[2])),
    ).to("cuda")

    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, input_dim).to("cuda")

    # Run the Mamba model
    # y, cache = mamba(x)
    y = mamba(x)

    # Print the output tensor
    print(y.shape)