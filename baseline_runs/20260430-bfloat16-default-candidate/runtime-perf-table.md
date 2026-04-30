| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-text-generate-bfloat16 | 0 | 9.02 | 0.58 | 0.00 | 0.62 | 0.00 | 0 B | 0 B | 0 B | 2.21 | 6.68 MiB | 4.91 GiB | 4.92 GiB | 4.83 GiB |
| tp-text-generate-bfloat16 | 1 | 8.92 | 0.58 | 0.00 | 0.62 | 0.00 | 0 B | 0 B | 0 B | 2.03 | 6.68 MiB | 4.91 GiB | 4.92 GiB | 4.83 GiB |
| tp-mm-generate-bfloat16-wide | 0 | 53.19 | 19.21 | 1.16 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 24.37 | 221.48 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-bfloat16-wide | 1 | 53.12 | 0.37 | 21.11 | 0.09 | 0.00 | 11.53 MiB | 0 B | 0 B | 23.74 | 221.48 MiB | 6.52 GiB | 6.73 GiB | 4.83 GiB |
| hybrid-mm-generate-bfloat16-wide | 0 | 32.52 | 19.29 | 0.37 | 0.05 | 0.00 | 3.10 MiB | 11.53 MiB | 3.08 MiB | 1.95 | 113.82 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate-bfloat16-wide | 1 | 32.82 | 0.00 | 0.00 | 0.05 | 0.00 | 0 B | 11.53 MiB | 0 B | 1.48 | 113.82 MiB | 3.22 GiB | 3.34 GiB | 2.42 GiB |
| hybrid-mm-generate-bfloat16-wide | 2 | 32.64 | 0.13 | 23.71 | 0.04 | 0.33 | 3.10 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
| tp-mm-generate-long-default | 0 | 84.24 | 19.20 | 1.18 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 50.08 | 451.41 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-long-default | 1 | 84.15 | 0.12 | 21.21 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 49.53 | 451.41 MiB | 6.52 GiB | 6.73 GiB | 4.83 GiB |
| tp-mm-generate-long-bfloat16 | 0 | 63.00 | 19.26 | 1.18 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 29.05 | 225.70 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-long-bfloat16 | 1 | 62.88 | 0.37 | 21.29 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 28.29 | 225.70 MiB | 6.52 GiB | 6.73 GiB | 4.83 GiB |
