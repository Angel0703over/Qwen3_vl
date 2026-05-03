| case | rank | total s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | CUDA peak | loaded weights | stage KV bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pp-mm-generate | 0 | 30.41 | 3.06 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 5.37 GiB | 4.11 GiB | 43.88 MiB / 44.16 MiB |
| pp-mm-generate | 1 | 30.51 | 3.06 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 5.46 GiB | 4.11 GiB | 43.88 MiB / 44.16 MiB |
| tp-mm-generate | 0 | 53.06 | 11.50 MiB | 0 B | 0 B | 24.36 | 220.43 MiB | 6.52 GiB | 4.83 GiB | 43.88 MiB / 44.16 MiB |
| tp-mm-generate | 1 | 53.18 | 11.50 MiB | 0 B | 0 B | 23.77 | 220.43 MiB | 6.52 GiB | 4.83 GiB | 43.88 MiB / 44.16 MiB |
| hybrid-mm-generate | 0 | 44.73 | 3.06 MiB | 11.50 MiB | 3.06 MiB | 13.23 | 113.28 MiB | 3.73 GiB | 2.42 GiB | 21.94 MiB / 22.08 MiB |
| hybrid-mm-generate | 1 | 44.90 | 0 B | 11.50 MiB | 0 B | 12.93 | 113.28 MiB | 3.23 GiB | 2.42 GiB | 21.94 MiB / 22.08 MiB |
| hybrid-mm-generate | 2 | 44.81 | 3.06 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 5.46 GiB | 4.11 GiB | 43.88 MiB / 44.16 MiB |
