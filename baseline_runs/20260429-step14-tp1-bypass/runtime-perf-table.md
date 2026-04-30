| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid-mm-generate | 0 | 33.18 | 19.32 | 0.41 | 0.05 | 0.00 | 3.10 MiB | 11.53 MiB | 6.15 MiB | 1.96 | 227.64 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate | 1 | 33.47 | 0.00 | 0.00 | 0.06 | 0.00 | 0 B | 11.53 MiB | 0 B | 1.53 | 227.64 MiB | 3.22 GiB | 3.34 GiB | 2.42 GiB |
| hybrid-mm-generate | 2 | 33.31 | 0.36 | 23.71 | 0.04 | 0.10 | 3.10 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
