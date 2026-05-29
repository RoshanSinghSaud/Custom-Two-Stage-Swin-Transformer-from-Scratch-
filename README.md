# Custom-Two-Stage-Swin-Transformer-from-Scratch-
A custom PyTorch reproduction of the Swin Transformer built from scratch. It features native implementations of local window partitioning, patch merging, relative position bias, and cyclic shifting with attention masking. The architecture is optimized into a clean two-stage hierarchical network evaluated on the MNIST image classification dataset.


🧠 Core Architectural Concepts:
The Swin Transformer's primary innovation is computing self-attention within localized, non-overlapping windows rather than globally across the entire image sequence. This keeps computational complexity linear to image size.

1. Window Partitioning (Local Attention):
 Instead of flattening a full feature map of size (B, H, W, C) into one massive sequence, the map is broken into isolated local grids of size win × win (e.g., 7 × 7).

The Mechanism: The tensor is reshaped and permuted from (B, H, W, C) into (B × N_windows, win, win, C).

Why it matters: Multi-Head Self-Attention (W-MSA) is computed only within these independent sub-batches. Pixels in Window A do not calculate attention scores with pixels in Window B, significantly reducing memory overhead.

2.  Window Shifting & Cyclic Masking (Cross-Window Context)
 Because standard window partitioning isolates pixels, features can never communicate across window boundaries. To fix this, every alternating block introduces a Shifted Window approach (SW-MSA).
The Shift: The entire feature map is physically rolled diagonally along its spatial axes using a displacement shift of -(win // 2) via torch.roll. New windows are then partitioned over this shifted image.

The Masking Challenge: Shifting causes pixels on the far edges of the image to cyclically wrap around to the opposite side. Because these wrapped edges are not actually adjacent in real space, computing attention across them would leak corrupt spatial context.

The Solution (Cyclic Mask): A custom attention mask is generated dynamically. It assigns a heavily negative bias parameter (-10000.0) to attention scores between non-contiguous wrapped segments. When passed through the Softmax layer, these invalid attention weights drop to 0, effectively isolating true structural context.




📊 Feature Map Tensor Flow
The pipeline processes input images through a hierarchical structure, altering shapes across the two main stages:

Input Image: (B, 1, 28, 28)
     │
[Patch Embedding] ──► Reduces spatial resolution by 2x via 2x2 convolutions    
     │
     ▼
STAGE 1: Feature Maps (B, 14, 14, 48)
     ├── Block 1: Regular Window Attention (W-MSA) ──► win=7, shift=0
     └── Block 2: Shifted Window Attention (SW-MSA) ──► win=7, shift=3
     │
     ▼
[Patch Merging] ──► Downsamples resolution 2x by concatenating 2x2 grid neighbors (B, 7, 7, 192)
     │              Linear projection layer squeezes channels back down to 96
     ▼
STAGE 2: Feature Maps (B, 7, 7, 96)
     ├── Block 3: Regular Window Attention (W-MSA) ──► win=3, shift=0 (Padded to 9x9)
     └── Block 4: Shifted Window Attention (SW-MSA) ──► win=3, shift=1 (Padded to 9x9)
     │
     ▼
[Global Average Pooling] ──► Linear Head Classifier ──► Output Logits (B, 10)

