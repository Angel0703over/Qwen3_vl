| case | rank | total s | prepare s | contract s | materialize s | barrier s | startup bytes | scaffold bytes | handoff bytes | TP coll s | TP coll bytes | cuda peak alloc | cuda peak reserved | loaded weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp-mm-generate-comm-bfloat16 | 0 | 53.68 | 19.24 | 1.16 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 24.77 | 224.56 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-comm-bfloat16 | 1 | 53.62 | 0.39 | 21.14 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 24.12 | 224.56 MiB | 6.53 GiB | 6.73 GiB | 4.83 GiB |
| tp-mm-generate-comm-float16 | 0 | 54.03 | 19.30 | 1.17 | 0.08 | 0.00 | 11.53 MiB | 0 B | 0 B | 24.96 | 224.56 MiB | 6.53 GiB | 6.74 GiB | 4.83 GiB |
| tp-mm-generate-comm-float16 | 1 | 53.99 | 0.36 | 21.32 | 0.09 | 0.00 | 11.53 MiB | 0 B | 0 B | 24.33 | 224.56 MiB | 6.53 GiB | 6.73 GiB | 4.83 GiB |
