| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-mm-generate | 0 | 74.78 | 19.24 | 1.16 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 45.65 | 449.12 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate | 1 | 74.72 | 0.36 | 21.17 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 44.91 | 449.12 MiB | 6.53 GiB | 6.73 GiB | 4.83 GiB |
| hybrid-mm-generate | 0 | 33.20 | 19.22 | 0.39 | 0.05 | 0.00 | 3.10 MiB | 11.53 MiB | 6.15 MiB | 2.23 | 227.64 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate | 1 | 33.45 | 0.00 | 0.00 | 0.06 | 0.00 | 0 B | 11.53 MiB | 0 B | 1.73 | 227.64 MiB | 3.22 GiB | 3.34 GiB | 2.42 GiB |
| hybrid-mm-generate | 2 | 33.31 | 0.36 | 23.57 | 0.04 | 0.11 | 3.10 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
