# Third-party notices

MLXL3 is an independent implementation that adapts algorithms and kernel
structures from the following projects.

## ExLlamaV3

Source: <https://github.com/turboderp-org/exllamav3>, pinned during development
at `ca5270c4b842876ddbe9a28594fbb6eac516cdf2`.

The EXL3 serialized format, trellis codec, procedural codebooks, permutation,
Hadamard reconstruction, and CUDA kernel algorithms are derived from
ExLlamaV3.

MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## PonyExl3

Source: <https://github.com/beamivalice/PonyExl3>, pinned during development at
`8e7fa6b1556f59fc669e25087903b279b9b0346f`.

Copyright 2026 Theinruj Toranavikrai

The Metal QMV/QMM work in `src/mlxl3/kernels/qmv.py` incorporates modified and
independently integrated kernel structures from PonyExl3. MLXL3 changes the
dispatch API, compile-time specialization, permutation embedding, split-K
policy, row crossover, output layout, and integration/tests. PonyExl3 is
licensed under the Apache License, Version 2.0; a copy is provided in
`LICENSES/Apache-2.0.txt`.
