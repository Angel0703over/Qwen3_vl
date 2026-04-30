| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-text-generate | 0 | 9.78 | 0.59 | 0.00 | 0.62 | 0.00 | 0 B | 0 B | 0 B | 2.63 | 13.54 MiB | 4.91 GiB | 4.92 GiB | 4.83 GiB |
| tp-text-generate | 1 | 9.67 | 0.59 | 0.00 | 0.62 | 0.00 | 0 B | 0 B | 0 B | 2.56 | 13.54 MiB | 4.91 GiB | 4.92 GiB | 4.83 GiB |
| tp-mm-generate | 0 | 74.66 | 19.20 | 1.20 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 45.02 | 449.12 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate | 1 | 74.57 | 0.39 | 21.19 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 44.29 | 449.12 MiB | 6.53 GiB | 6.73 GiB | 4.83 GiB |
| hybrid-mm-generate | 0 | 35.59 | 19.28 | 0.37 | 0.05 | 0.00 | 3.10 MiB | 11.53 MiB | 6.15 MiB | 2.07 | 227.64 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate | 1 | 35.88 | 0.00 | 0.00 | 0.06 | 0.00 | 0 B | 11.53 MiB | 0 B | 1.60 | 227.64 MiB | 3.22 GiB | 3.34 GiB | 2.42 GiB |
| hybrid-mm-generate | 2 | 35.69 | 0.35 | 23.75 | 0.04 | 0.12 | 3.10 MiB | 0 B | 3.08 MiB | 1.53 | 648.46 MiB | 5.46 GiB | 5.59 GiB | 4.11 GiB |
