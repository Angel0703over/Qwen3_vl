| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-mm-generate-step15 | 0 | 53.35 | 19.23 | 1.16 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 24.45 | 221.48 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-step15 | 1 | 53.29 | 0.35 | 21.22 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 23.83 | 221.48 MiB | 6.52 GiB | 6.73 GiB | 4.83 GiB |
| hybrid-mm-generate-step15 | 0 | 32.36 | 19.15 | 0.31 | 0.06 | 0.00 | 3.10 MiB | 11.53 MiB | 3.08 MiB | 2.13 | 113.82 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate-step15 | 1 | 32.70 | 0.00 | 0.00 | 0.05 | 0.00 | 0 B | 11.53 MiB | 0 B | 1.66 | 113.82 MiB | 3.22 GiB | 3.34 GiB | 2.42 GiB |
| hybrid-mm-generate-step15 | 2 | 32.46 | 0.36 | 23.52 | 0.04 | 0.05 | 3.10 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
