| case | rank | total s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | CUDA peak | loaded weights | stage KV bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hf-mm-generate | - | 11.03 | 0 B | 0 B | 0 B | 0.00 | 0 B | 8.55 GiB | - | - |
| pp-mm-generate | 0 | 30.26 | 3.06 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 5.37 GiB | 4.11 GiB | 43.88 MiB / 44.16 MiB |
| pp-mm-generate | 1 | 30.45 | 3.06 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 5.46 GiB | 4.11 GiB | 43.88 MiB / 44.16 MiB |
| pp3-mm-generate | 0 | 31.09 | 6.11 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 3.86 GiB | 2.98 GiB | 29.25 MiB / 29.44 MiB |
| pp3-mm-generate | 1 | 31.23 | 3.06 MiB | 0 B | 6.12 MiB | 0.00 | 0 B | 3.12 GiB | 2.26 GiB | 29.25 MiB / 29.44 MiB |
| pp3-mm-generate | 2 | 31.23 | 3.06 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 3.95 GiB | 2.98 GiB | 29.25 MiB / 29.44 MiB |
| tp-mm-generate | 0 | 53.03 | 11.50 MiB | 0 B | 0 B | 24.35 | 220.43 MiB | 6.52 GiB | 4.83 GiB | 43.88 MiB / 44.16 MiB |
| tp-mm-generate | 1 | 53.15 | 11.50 MiB | 0 B | 0 B | 23.81 | 220.43 MiB | 6.52 GiB | 4.83 GiB | 43.88 MiB / 44.16 MiB |
| hybrid-mm-generate | 0 | 44.39 | 3.06 MiB | 11.50 MiB | 3.06 MiB | 13.18 | 113.28 MiB | 3.73 GiB | 2.42 GiB | 21.94 MiB / 22.08 MiB |
| hybrid-mm-generate | 1 | 44.59 | 0 B | 11.50 MiB | 0 B | 12.85 | 113.28 MiB | 3.23 GiB | 2.42 GiB | 21.94 MiB / 22.08 MiB |
| hybrid-mm-generate | 2 | 44.61 | 3.06 MiB | 0 B | 3.06 MiB | 0.00 | 0 B | 5.46 GiB | 4.11 GiB | 43.88 MiB / 44.16 MiB |
| tp-mm-generate-long | 0 | 61.70 | 11.50 MiB | 0 B | 0 B | 28.36 | 224.65 MiB | 6.53 GiB | 4.83 GiB | 43.88 MiB / 45.00 MiB |
| tp-mm-generate-long | 1 | 61.78 | 11.50 MiB | 0 B | 0 B | 27.65 | 224.65 MiB | 6.52 GiB | 4.83 GiB | 43.88 MiB / 45.00 MiB |
