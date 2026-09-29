# Native MTP Q8 operator qualification

The synthetic Q8 operator passed on Carrack GPU 0, an RTX 5090, at source
`48cbc42e74bdf99ffe3c901adf91fdc1c9f62bbd`. This does not qualify native MTP
model execution, real-weight arithmetic, proposal acceptance, or throughput.

Evidence: [native-q8-1](../evidence/reassess-20260928/native-q8-1/).

## Identity

- PTX SHA-256: `e482363228aa8eaedb146d12d5263edd4aabab6245e93d252dad0888f7eb80b9`.
- Executable SHA-256: `79dec8704eb7bf4451a603387163cee17ff766ab747b110580cef6876150686b`.
- GPU UUID: `GPU-80ded6bd-1a89-2628-3d94-902187dbab1d`.

## Results and limits

All six synthetic cases passed: K=128, 160, 256, 5120, 10240, and 17408.
Each projects two selected parent rows. The fixtures exercise a padded tail,
multiple 128-column splits, and reordered rows. Maximum absolute and scaled
errors against the independent reference were zero for these fixtures; the
declared scaled-error limit remains `0.0001`. Repeated raw outputs also matched.
The synthetic values use small BF16 activations and power-of-two scales, so zero
observed error is not a general exactness claim for arbitrary model weights.

Memcheck and synccheck reported zero errors. Racecheck reported zero hazards,
errors, or warnings. The kernel used 40 registers and no local or static shared
memory in this run. Its JIT log retains a `setmaxnreg` warning; no performance
claim follows from these operator checks.

Both initially active services were restored after qualification, and Ninfer
health returned HTTP 200. The Q4 proposal head, resident native MTP integration,
target verification and rollback, quality checks, and accepted-token timing
remain separate requirements.
