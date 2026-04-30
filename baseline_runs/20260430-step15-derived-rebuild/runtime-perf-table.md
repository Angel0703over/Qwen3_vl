| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-mm-generate-derived-rebuild | 0 | 53.33 | 19.31 | 1.11 | 0.13 | 0.00 | 11.51 MiB | 0 B | 0 B | 24.37 | 221.48 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-derived-rebuild | 1 | 53.28 | 0.39 | 21.17 | 0.10 | 0.00 | 11.51 MiB | 0 B | 0 B | 23.68 | 221.48 MiB | 6.52 GiB | 6.73 GiB | 4.83 GiB |
| hybrid-mm-generate-derived-rebuild | 0 | 32.65 | 19.30 | 0.29 | 0.07 | 0.00 | 3.07 MiB | 11.51 MiB | 3.08 MiB | 2.36 | 113.82 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate-derived-rebuild | 1 | 32.76 | 0.00 | 0.00 | 0.09 | 0.00 | 0 B | 11.51 MiB | 0 B | 1.58 | 113.82 MiB | 3.22 GiB | 3.34 GiB | 2.42 GiB |
| hybrid-mm-generate-derived-rebuild | 2 | 32.84 | 0.41 | 23.64 | 0.04 | 0.04 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
