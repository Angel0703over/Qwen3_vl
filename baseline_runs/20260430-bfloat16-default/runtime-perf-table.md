| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-text-generate-default | 0 | 8.96 | 0.58 | 0.00 | 0.63 | 0.00 | 0 B | 0 B | 0 B | 2.16 | 6.68 MiB | 4.91 GiB | 4.92 GiB | 4.83 GiB |
| tp-text-generate-default | 1 | 8.89 | 0.57 | 0.00 | 0.62 | 0.00 | 0 B | 0 B | 0 B | 2.03 | 6.68 MiB | 4.91 GiB | 4.92 GiB | 4.83 GiB |
| tp-mm-generate-default | 0 | 53.44 | 19.30 | 1.21 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 24.51 | 221.48 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-default | 1 | 53.43 | 0.37 | 21.39 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 23.77 | 221.48 MiB | 6.52 GiB | 6.73 GiB | 4.83 GiB |
| hybrid-mm-generate-default | 0 | 32.58 | 19.34 | 0.38 | 0.06 | 0.00 | 3.10 MiB | 11.53 MiB | 3.08 MiB | 2.08 | 113.82 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate-default | 1 | 32.86 | 0.00 | 0.00 | 0.06 | 0.00 | 0 B | 11.53 MiB | 0 B | 1.63 | 113.82 MiB | 3.22 GiB | 3.34 GiB | 2.42 GiB |
| hybrid-mm-generate-default | 2 | 32.66 | 0.36 | 23.77 | 0.05 | 0.10 | 3.10 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
| tp-mm-generate-long-default-bfloat16 | 0 | 62.66 | 19.26 | 1.17 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 28.84 | 225.70 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-long-default-bfloat16 | 1 | 62.56 | 0.36 | 21.17 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 28.07 | 225.70 MiB | 6.52 GiB | 6.73 GiB | 4.83 GiB |
