| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid-mm-generate-derived-opt | 0 | 35.21 | 19.18 | 0.42 | 0.05 | 0.00 | 3.10 MiB | 0 B | 6.15 MiB | 1.48 | 648.46 MiB | 5.39 GiB | 5.44 GiB | 4.11 GiB |
| hybrid-mm-generate-derived-opt | 1 | 35.16 | 0.39 | 23.53 | 0.04 | 0.31 | 3.10 MiB | 0 B | 3.08 MiB | 1.47 | 648.46 MiB | 5.46 GiB | 5.59 GiB | 4.11 GiB |
| pp-mm-generate-derived-opt | 0 | 30.73 | 19.23 | 0.38 | 0.04 | 0.00 | 3.10 MiB | 0 B | 6.15 MiB | 0.00 | 0 B | 5.37 GiB | 5.42 GiB | 4.11 GiB |
| pp-mm-generate-derived-opt | 1 | 30.79 | 0.36 | 23.60 | 0.06 | 0.32 | 3.10 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
