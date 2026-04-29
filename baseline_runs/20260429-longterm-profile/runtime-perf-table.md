| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-mm-generate | 0 | 74.64 | 19.22 | 1.19 | 0.09 | 0.00 | 11.53 MiB | 0 B | 0 B | 45.19 | 449.12 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate | 1 | 74.66 | 0.35 | 21.21 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 44.37 | 449.12 MiB | 6.53 GiB | 6.73 GiB | 4.83 GiB |
| hybrid-text-generate | 0 | 10.77 | 0.00 | 0.00 | 0.04 | 0.27 | 0 B | 377 B | 190.00 KiB | 0.72 | 6.87 MiB | 2.46 GiB | 2.47 GiB | 2.42 GiB |
| hybrid-text-generate | 1 | 7.43 | 0.00 | 0.00 | 0.04 | 0.27 | 0 B | 377 B | 0 B | 0.66 | 6.87 MiB | 2.46 GiB | 2.47 GiB | 2.42 GiB |
| hybrid-text-generate | 2 | 10.56 | 0.01 | 0.00 | 0.06 | 0.00 | 0 B | 0 B | 95.00 KiB | 0.47 | 19.56 MiB | 4.18 GiB | 4.19 GiB | 4.11 GiB |
| hybrid-mm-generate | 0 | 35.06 | 19.19 | 0.40 | 0.05 | 0.00 | 3.10 MiB | 11.53 MiB | 6.15 MiB | 1.97 | 227.64 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate | 1 | 35.36 | 0.00 | 0.00 | 0.06 | 0.00 | 0 B | 11.53 MiB | 0 B | 1.56 | 227.64 MiB | 3.22 GiB | 3.34 GiB | 2.42 GiB |
| hybrid-mm-generate | 2 | 35.14 | 0.36 | 23.55 | 0.04 | 0.11 | 3.10 MiB | 0 B | 3.08 MiB | 1.39 | 648.46 MiB | 5.46 GiB | 5.59 GiB | 4.11 GiB |
