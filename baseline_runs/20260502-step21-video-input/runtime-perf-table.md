| case | rank | total s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | CUDA peak | loaded weights | stage KV bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hf-mm-generate-video-builder-prompt | - | 37.12 | 0 B | 0 B | 0 B | 0.00 | 0 B | 9.76 GiB | - | - |
| pp-mm-generate-video | 0 | 74.54 | 7.42 MiB | 0 B | 7.41 MiB | 0.00 | 0 B | 7.48 GiB | 4.11 GiB | 106.52 MiB / 106.80 MiB |
| pp-mm-generate-video | 1 | 76.42 | 7.42 MiB | 0 B | 7.41 MiB | 0.00 | 0 B | 7.44 GiB | 4.11 GiB | 106.52 MiB / 106.80 MiB |
| tp-mm-generate-video | 0 | 106.15 | 29.10 MiB | 0 B | 0 B | 55.69 | 533.67 MiB | 8.86 GiB | 4.83 GiB | 106.52 MiB / 106.80 MiB |
| tp-mm-generate-video | 1 | 106.20 | 29.10 MiB | 0 B | 0 B | 55.12 | 533.67 MiB | 8.85 GiB | 4.83 GiB | 106.52 MiB / 106.80 MiB |
| hybrid-mm-generate-video-pp2tp1 | 0 | 58.76 | 7.42 MiB | 0 B | 7.41 MiB | 0.00 | 0 B | 7.50 GiB | 4.11 GiB | 106.52 MiB / 106.80 MiB |
| hybrid-mm-generate-video-pp2tp1 | 1 | 58.79 | 7.42 MiB | 0 B | 7.41 MiB | 0.00 | 0 B | 7.45 GiB | 4.11 GiB | 106.52 MiB / 106.80 MiB |
| tp-mm-generate-frame-regression | 0 | 53.06 | 11.51 MiB | 0 B | 0 B | 24.63 | 221.48 MiB | 6.53 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
| tp-mm-generate-frame-regression | 1 | 53.08 | 11.51 MiB | 0 B | 0 B | 23.84 | 221.48 MiB | 6.52 GiB | 4.83 GiB | 44.09 MiB / 44.37 MiB |
