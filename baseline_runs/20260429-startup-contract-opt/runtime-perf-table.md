| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid-mm-generate-startup-opt | 0 | 35.10 | 19.22 | 0.55 | 0.03 | 0.00 | 4.15 MiB | 0 B | 6.15 MiB | 1.39 | 648.46 MiB | 5.39 GiB | 5.44 GiB | 4.11 GiB |
| hybrid-mm-generate-startup-opt | 1 | 35.08 | 0.37 | 23.76 | 0.03 | 0.32 | 4.15 MiB | 0 B | 3.08 MiB | 1.44 | 648.46 MiB | 5.46 GiB | 5.59 GiB | 4.11 GiB |
| pp-mm-generate-startup-opt | 0 | 31.13 | 19.20 | 0.52 | 0.03 | 0.00 | 4.15 MiB | 0 B | 6.15 MiB | 0.00 | 0 B | 5.37 GiB | 5.42 GiB | 4.11 GiB |
| pp-mm-generate-startup-opt | 1 | 31.15 | 0.35 | 23.89 | 0.03 | 0.34 | 4.15 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
