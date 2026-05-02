| case | rank | total s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | CUDA peak | loaded weights | stage KV bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hf-text-generate | - | 8.76 | 0 B | 0 B | 0 B | 0.00 | 0 B | 8.28 GiB | - | - |
| hf-mm-generate | - | 10.92 | 0 B | 0 B | 0 B | 0.00 | 0 B | 8.55 GiB | - | - |
| pp-mm-generate | 0 | 30.25 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.39 GiB | 4.11 GiB | 44.09 MiB / 44.37 MiB |
| pp-mm-generate | 1 | 30.29 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.47 GiB | 4.11 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate | 0 | 52.95 | 11.51 MiB | 0 B | 0 B | 24.42 | 221.48 MiB | 6.53 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate | 1 | 52.99 | 11.51 MiB | 0 B | 0 B | 23.83 | 221.48 MiB | 6.52 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate-long | 0 | 62.03 | 11.51 MiB | 0 B | 0 B | 28.58 | 225.70 MiB | 6.53 GiB | 4.83 GiB | 44.09 MiB / 45.21 MiB |
| tp-mm-generate-long | 1 | 62.05 | 11.51 MiB | 0 B | 0 B | 27.89 | 225.70 MiB | 6.52 GiB | 4.83 GiB | 44.09 MiB / 45.21 MiB |
| tp-mm-generate-frame-regression | 0 | 52.96 | 11.51 MiB | 0 B | 0 B | 24.28 | 221.48 MiB | 6.53 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate-frame-regression | 1 | 52.97 | 11.51 MiB | 0 B | 0 B | 23.58 | 221.48 MiB | 6.52 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
