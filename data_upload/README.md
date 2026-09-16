# Pasta de transferência do dataset

Depois de executar `git pull` no servidor, envie o arquivo abaixo para esta pasta:

```text
rtdetr_botos/data_upload/BOTOS_RTDETR_E2C_TEST_20260915_r1.zip
```

O ZIP não deve ser enviado pelo Git. O script `scripts/run_remote.sh` verifica e
extrai o pacote para `rtdetr_botos/data/` antes do treinamento.

