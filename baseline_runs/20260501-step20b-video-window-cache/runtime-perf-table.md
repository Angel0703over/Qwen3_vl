| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-mm-generate-step20b | 0 | 53.28 | 19.28 | 1.14 | 0.09 | 0.00 | 11.51 MiB | 0 B | 0 B | 24.51 | 221.48 MiB | 6.53 GiB | 6.60 GiB | 4.83 GiB |
| tp-mm-generate-step20b | 1 | 53.22 | 0.38 | 21.13 | 0.08 | 0.00 | 11.51 MiB | 0 B | 0 B | 23.80 | 221.48 MiB | 6.52 GiB | 6.62 GiB | 4.83 GiB |
| hybrid-mm-generate-step20b | 0 | 33.00 | 19.34 | 0.32 | 0.08 | 0.00 | 3.07 MiB | 11.51 MiB | 3.08 MiB | 2.34 | 113.82 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate-step20b | 1 | 33.15 | 0.00 | 0.00 | 0.08 | 0.00 | 0 B | 11.51 MiB | 0 B | 1.64 | 113.82 MiB | 3.23 GiB | 3.28 GiB | 2.42 GiB |
| hybrid-mm-generate-step20b | 2 | 33.13 | 0.41 | 23.63 | 0.04 | 0.02 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.47 GiB | 5.59 GiB | 4.11 GiB |
