| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-mm-generate-step16-default-j23 | 0 | 53.47 | 19.28 | 1.12 | 0.09 | 0.00 | 11.51 MiB | 0 B | 0 B | 24.34 | 221.48 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-step16-default-j23 | 1 | 53.21 | 0.38 | 21.16 | 0.09 | 0.00 | 11.51 MiB | 0 B | 0 B | 23.76 | 221.48 MiB | 6.52 GiB | 6.73 GiB | 4.83 GiB |
| tp-mm-generate-step16-pinned-j23 | 0 | 53.01 | 19.27 | 1.41 | 0.08 | 0.00 | 11.51 MiB | 0 B | 0 B | 23.91 | 221.48 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-step16-pinned-j23 | 1 | 52.97 | 0.15 | 21.41 | 0.08 | 0.00 | 11.51 MiB | 0 B | 0 B | 23.51 | 221.48 MiB | 6.52 GiB | 6.73 GiB | 4.83 GiB |
| hybrid-mm-generate-step16-default-j23shared | 0 | 32.69 | 19.23 | 0.35 | 0.08 | 0.00 | 3.07 MiB | 11.51 MiB | 3.08 MiB | 2.25 | 113.82 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate-step16-default-j23shared | 1 | 32.74 | 0.00 | 0.00 | 0.08 | 0.00 | 0 B | 11.51 MiB | 0 B | 1.58 | 113.82 MiB | 3.22 GiB | 3.34 GiB | 2.42 GiB |
| hybrid-mm-generate-step16-default-j23shared | 2 | 32.85 | 0.39 | 23.43 | 0.04 | 0.03 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
| hybrid-mm-generate-step16-pinned-j23shared | 0 | 32.80 | 19.22 | 0.57 | 0.08 | 0.00 | 3.07 MiB | 11.51 MiB | 3.08 MiB | 2.27 | 113.82 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate-step16-pinned-j23shared | 1 | 32.85 | 0.00 | 0.00 | 0.08 | 0.00 | 0 B | 11.51 MiB | 0 B | 1.49 | 113.82 MiB | 3.22 GiB | 3.34 GiB | 2.42 GiB |
| hybrid-mm-generate-step16-pinned-j23shared | 2 | 32.96 | 0.15 | 23.82 | 0.04 | 0.30 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.46 GiB | 5.59 GiB | 4.11 GiB |
