# Persistent text weights and state allocation

Status: implementation in progress; live qualification pending. Full model
execution and all inference performance measurements remain pending.

The previously qualified GDN and full-attention trials reload subsets of weights
and maintain operation-local allocations. The next gate stores all 1,620 text
tensors (21,646,480,768 raw bytes) in one 256-byte-aligned device allocation. The
verified artifact reader streams and rehashes each object; the GPU trial reads
every tensor back in bounded chunks and compares its original artifact SHA256.
The separate 15 MTP tensors are deferred until base-model execution works.

The generic engine owns checked named allocation placement. The Qwen package
owns its 64-layer schedule: 48 GDN / 16 full-attention blocks, 56 NVFP4 / 8 FP8
MLPs. Each GDN block has three BF16 convolution-history rows and a FP32 recurrent
matrix. Full-attention blocks have separate BF16 K/V arrays. State storage is
153,944,064 + context_capacity * 65,536 bytes. At capacity 131,072 this is
8,743,878,656 bytes, initialized and fully read back as zero.

Admission checks aligned weights plus state bytes and a 1 GiB workspace reserve
against current free memory after module loading. The reserve is an admission
margin, not an allocated workspace or an observed inference peak. CUDA free
memory snapshots are allocation checkpoints, not a continuous peak measurement.

The first embedding/input-normalization operation uses pointers inside the full
resident weight arena, including vocabulary rows zero and 248,319. Its independent
CPU reference retains only those selected source rows while hashing the complete
embedding object. Neither CPU expected outputs nor device readbacks feed GPU
computation. Persistent state capacity is not tested usable inference context.

Reproduction (after building through the existing Just recipes):

```sh
target/release/xtask specialize qwen-residency-check \
  --artifact /data/ai/models/mesh-specialize/qwen3.8-27b-f0b7c9e7-raw-v1.mspec \
  --ptx target/specialize/probes.ptx --device 0 --output NEW_FILE
```

No device assembly changes are required. Exact revision, hardware, checks,
sanitizer results and raw evidence will be recorded after the bounded trial.
