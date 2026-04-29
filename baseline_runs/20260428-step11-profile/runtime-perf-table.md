| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pp-text-generate | 0 | 5.49 | 0.00 | 0.00 | 0.03 | 0.10 | 0 B | 0 B | 190.00 KiB | 0.00 | 0 B | 4.17 GiB | 4.19 GiB | 4.11 GiB |
| pp-text-generate | 1 | 5.44 | 0.01 | 0.00 | 0.06 | 0.00 | 0 B | 0 B | 95.00 KiB | 0.00 | 0 B | 4.18 GiB | 4.19 GiB | 4.11 GiB |
| pp-mm-generate | 0 | 31.27 | 19.16 | 0.84 | 0.03 | 0.00 | 7.21 MiB | 0 B | 6.15 MiB | 0.00 | 0 B | 5.37 GiB | 5.42 GiB | 4.11 GiB |
| pp-mm-generate | 1 | 31.26 | 0.36 | 24.12 | 0.03 | 0.29 | 7.21 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
| tp-text-generate | 0 | 10.42 | 0.58 | 0.00 | 0.63 | 0.00 | 0 B | 0 B | 0 B | 3.10 | 13.54 MiB | 4.91 GiB | 4.92 GiB | 4.83 GiB |
| tp-text-generate | 1 | 10.27 | 0.58 | 0.00 | 0.62 | 0.00 | 0 B | 0 B | 0 B | 3.02 | 13.54 MiB | 4.91 GiB | 4.92 GiB | 4.83 GiB |
| tp-mm-generate | 0 | 72.31 | 19.14 | 0.00 | 0.07 | 0.00 | 0 B | 0 B | 0 B | 44.40 | 449.12 MiB | 6.53 GiB | 6.75 GiB | 4.83 GiB |
| tp-mm-generate | 1 | 72.22 | 19.11 | 0.00 | 0.07 | 0.00 | 0 B | 0 B | 0 B | 44.27 | 449.12 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
