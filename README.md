# Pipeline RT-DETR para BOTOS

Pipeline reproduzível para treinar `RT-DETR-L` no dataset humano revisado
`BOTOS_PPGZOO_v3`, selecionar o checkpoint pela validação de desenvolvimento e
avaliá-lo uma única vez no teste temporal fechado.

## Protocolo adotado

- **Treino/validação:** split `v3_E2c_balanced_cap75_gap10`, com 637 imagens de
  treino, 124 de validação, redução temporal de 10 quadros e limite de 75 por
  vídeo PPGZOO (o legado é preservado).
- **Teste fechado:** 52 imagens do intervalo temporal de teste do DJI_0203.
- **Classe:** `boto-cinza` (`class_id=0`).
- **Métrica principal:** AP COCO `AP50–95`; também são salvos AP50, AP75,
  AP/AR por porte, precisão, recall, curvas e matriz de confusão.
- **Regra:** o teste não participa de treino, early stopping ou escolha de
  hiperparâmetros. Validação não é apresentada como teste independente.

O E2c foi escolhido por reduzir redundância temporal e por ser o split usado no
melhor resultado de validação já observado no projeto (E8). Isso não prova que o
RT-DETR será superior: esta execução é o teste controlado dessa hipótese.

## 1. Envio pelo Git e transferência do ZIP

Suba esta pasta de código pelo Git. O ZIP e os resultados estão ignorados pelo
`.gitignore`.

No servidor:

```bash
git pull
cd rtdetr_botos
```

Transfira o arquivo para exatamente:

```text
rtdetr_botos/data_upload/BOTOS_RTDETR_E2C_TEST_20260915_r1.zip
```

Exemplo a partir da máquina local (ajuste usuário e host):

```bash
rsync -avhP --append-verify \
  rtdetr_botos/data_upload/BOTOS_RTDETR_E2C_TEST_20260915_r1.zip \
  USUARIO@SERVIDOR:/CAMINHO/DO/REPOSITORIO/rtdetr_botos/data_upload/
```

Compare o SHA-256 local e remoto antes de extrair. O valor esperado também está
versionado em `checksums/`:

```bash
sha256sum -c checksums/BOTOS_RTDETR_E2C_TEST_20260915_r1.zip.sha256
```

## 2. Configuração do ambiente remoto

Requer Python 3.10+ e uma GPU NVIDIA. O servidor Curupira usa driver 550 com
CUDA 12.4; por isso o instalador fixa `torch==2.6.0`, `torchvision==0.21.0` e o
índice oficial `cu124`, combinação que também suporta Python 3.13.

```bash
bash scripts/setup_remote.sh
```

O script agora encerra com erro se o wheel não for CUDA 12.4 ou se
`torch.cuda.is_available()` não retornar `True`.

Se o ambiente anterior instalou `torch 2.14.0+cu130`, recrie-o antes de executar
o instalador atualizado:

```bash
deactivate 2>/dev/null || true
mv .venv .venv-cu130-backup
bash scripts/setup_remote.sh
```

Depois de confirmar que CUDA e o treino funcionam, o ambiente antigo pode ser
removido. Ele não contém dataset, código ou resultados.

## 3. Treino + validação + teste final

Execute dentro de `tmux` ou do gerenciador de jobs do servidor:

```bash
bash scripts/run_remote.sh
```

O fluxo realiza: integridade do ZIP → extração segura → auditoria do dataset →
preflight CUDA → treino → avaliação em `val` → avaliação em `test`.

Para nomear a execução:

```bash
bash scripts/run_remote.sh data_upload/BOTOS_RTDETR_E2C_TEST_20260915_r1.zip rtdetr_l_e2c_seed42_run01
```

Os artefatos ficam em `runs/<run_id>/`, incluindo checkpoints, `results.csv`,
curvas, matriz de confusão, predições JSON e `evaluation/*/EVALUATION.json`.
O arquivo de avaliação inclui `AP_small` COCO (com a convenção oficial de até
100 detecções), hashes dos pesos/dataset, versões do ambiente e tempo de
inferência. A avaliação Ultralytics mantém `max_det=300` separadamente.

## Ajustes de hardware

O padrão para a Quadro RTX 5000 de 16 GB é `imgsz=1280`, `batch=1`, GPU `0`.
Não use `batch=2` antes de verificar a memória livre e executar um smoke test.

```bash
.venv/bin/python scripts/train.py \
  --data data/BOTOS_RTDETR_E2C_TEST_20260915_r1/data.yaml \
  --run-id rtdetr_l_e2c_seed42_run01
```

Não reduza `imgsz` sem registrar um novo experimento: os alvos são pequenos e a
resolução é parte do protocolo. A documentação atual do Ultralytics informa que
o `grid_sample` usado pelo RT-DETR em CUDA não oferece backward determinístico;
por isso `deterministic: false`, mantendo `seed: 42` para controlar as demais
fontes de aleatoriedade. Se aparecerem NaNs durante a associação bipartida,
altere `amp: false` no YAML e reinicie com um novo `run_id`.

### Falta de VRAM em GPU compartilhada

Para `imgsz=1280`, o preflight exige por padrão pelo menos 14 GiB livres na GPU.
O RT-DETR-L observado usou cerca de 11,7 GiB de forma sustentada, mas os picos
podem ser maiores. O script encerra antes do treino se outros processos deixarem
menos memória que o limite.

Consulte os donos dos processos antes de interromper qualquer PID:

```bash
ps -o user,pid,etime,cmd -p PID1,PID2
nvidia-smi
```

Se houver uma GPU dedicada com memória suficiente, selecione-a, por exemplo:

```bash
RTDETR_DEVICE=1 bash scripts/run_remote.sh \
  data_upload/BOTOS_RTDETR_E2C_TEST_20260915_r1.zip \
  rtdetr_l_e2c_i1280_b1_gpu1_seed42_run01
```

Se nenhuma GPU puder ser liberada, use o fallback controlado em 1024 px. Isso é
um experimento diferente e não deve ser comparado diretamente com resultados em
1280 px sem declarar a mudança:

```bash
RTDETR_IMGSZ=1024 RTDETR_MIN_FREE_GIB=10 bash scripts/run_remote.sh \
  data_upload/BOTOS_RTDETR_E2C_TEST_20260915_r1.zip \
  rtdetr_l_e2c_i1024_b1_seed42_run01
```

As variáveis `RTDETR_DEVICE`, `RTDETR_BATCH`, `RTDETR_IMGSZ` e
`RTDETR_MIN_FREE_GIB` são registradas indiretamente nos argumentos e no snapshot
da execução. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` é aplicado para
reduzir falhas por fragmentação, mas não substitui VRAM física livre.

## Execuções separadas

```bash
.venv/bin/python scripts/unpack_dataset.py data_upload/BOTOS_RTDETR_E2C_TEST_20260915_r1.zip --destination data
.venv/bin/python scripts/verify_dataset.py --data data/BOTOS_RTDETR_E2C_TEST_20260915_r1/data.yaml
.venv/bin/python scripts/train.py --data data/BOTOS_RTDETR_E2C_TEST_20260915_r1/data.yaml --run-id meu_run
.venv/bin/python scripts/evaluate.py --weights runs/meu_run/weights/best.pt --data data/BOTOS_RTDETR_E2C_TEST_20260915_r1/data.yaml --split test --output runs/meu_run/evaluation/test
```
