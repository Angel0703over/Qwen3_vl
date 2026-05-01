| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-text-generate-step20a | 0 | 9.07 | 0.58 | 0.00 | 0.62 | 0.00 | 0 B | 0 B | 0 B | 2.14 | 6.68 MiB | 4.91 GiB | 4.92 GiB | 4.83 GiB |
| tp-text-generate-step20a | 1 | 8.96 | 0.58 | 0.00 | 0.62 | 0.00 | 0 B | 0 B | 0 B | 2.08 | 6.68 MiB | 4.91 GiB | 4.92 GiB | 4.83 GiB |
| tp-mm-generate-step20a | 0 | 53.44 | 19.39 | 1.16 | 0.08 | 0.00 | 11.51 MiB | 0 B | 0 B | 24.42 | 221.48 MiB | 6.53 GiB | 6.60 GiB | 4.83 GiB |
| tp-mm-generate-step20a | 1 | 53.36 | 0.39 | 21.37 | 0.09 | 0.00 | 11.51 MiB | 0 B | 0 B | 23.77 | 221.48 MiB | 6.52 GiB | 6.62 GiB | 4.83 GiB |
| hybrid-mm-generate-step20a | 0 | 32.51 | 19.26 | 0.31 | 0.07 | 0.00 | 3.07 MiB | 11.51 MiB | 3.08 MiB | 2.30 | 113.82 MiB | 3.73 GiB | 3.75 GiB | 2.42 GiB |
| hybrid-mm-generate-step20a | 1 | 32.55 | 0.00 | 0.00 | 0.08 | 0.00 | 0 B | 11.51 MiB | 0 B | 1.60 | 113.82 MiB | 3.23 GiB | 3.28 GiB | 2.42 GiB |
| hybrid-mm-generate-step20a | 2 | 32.62 | 0.38 | 23.51 | 0.05 | 0.05 | 3.07 MiB | 0 B | 3.08 MiB | 0.00 | 0 B | 5.47 GiB | 5.59 GiB | 4.11 GiB |
