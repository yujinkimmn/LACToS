# Wafer diffusion 학습부터 LACToS inference까지

정상 wafer 이미지로 diffusion model을 새로 학습하고, 검사 이미지마다 여러 reconstruction을 생성한 뒤 **공식 LACToS intensity-only 점수**를 계산한다. 데이터·checkpoint·결과는 사용자가 지정한 사내 실행 환경에만 저장한다.

이 저장소의 `wafer_diffusion/`는 새로 구현한 학습·sampling 코드다. 저자 모델과 같은 architecture나 논문 성능을 재현했다고 주장하지 않는다. 공식 저장소의 학습 코드는 toy 전용이므로, 별도의 epsilon-predicting U-Net/DDPM 학습과 full-chain DPS를 구현했다. 마지막 superpixel/typicality 계산만 검토한 공식 파일 두 개를 직접 호출한다. Feature/ResNet/autoencoder는 사용하지 않는다.

## 먼저 준비할 것

| 준비물 | 설명 |
|---|---|
| 정상 이미지 | 검출하려는 결함이 없는 이미지. 기존 classifier의 `background`가 항상 정상이라는 가정은 하지 않는다. |
| 그룹 ID | lot+wafer 또는 lot 전체. 동일 그룹은 train/val/test를 넘지 않는다. |
| 불량 이미지 | 선택 사항. 제공하면 test에서 image-level recall, false alarm, AUROC/AP를 계산한다. |
| 실행 환경 | Python 3.12, GPU용 PyTorch는 해당 서버의 driver/CUDA와 호환되는 승인된 wheel 사용. CPU 기능 검증은 완료, CUDA 실행은 미검증. |
| 공식 source 두 파일 | 아래 고정 commit의 `ood/scoring.py`, `ood/superpixels.py`. 승인된 경로로 사내에 반입한다. |

지원 입력은 8-bit PNG/JPEG/BMP/TIFF이다. RGB로 변환하고, 선택한 border crop 후 정사각형으로 resize한다. DAT, 16-bit 센서 데이터, multi-frame TIFF는 조용히 변환하지 않고 중단한다. 이 경우 실제 파일 규격과 intensity 변환 규칙에 맞춘 사전 변환이 필요하다. 이미지 원본의 물리 좌표로 heatmap을 사용하려면 crop/resize 역변환도 별도로 적용해야 한다.

## 설치와 코드 준비

학습 중 다운로드는 없다. 아래 설치/반입 단계는 **데이터를 열기 전에**, 회사의 허용된 설치 절차로 수행한다. 기존 논문 검증 환경(torch 2.0.1)과 분리한다.

```bash
python3.12 -m venv .venv-wafer
source .venv-wafer/bin/activate
python -m pip install --only-binary=:all: 'pip>=26.2' 'setuptools>=84.1'
```

GPU 서버에서는 승인된 PyTorch 2.15 계열 GPU wheel을 먼저 설치한다. CPU 검증 환경은 다음 명령을 사용한다.

```bash
python -m pip install --only-binary=:all: torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=:all: -r requirements-wafer.lock
```

`requirements-wafer.lock`은 검증한 Linux/Python 3.12/CPU 환경의 직접·간접 의존성을 고정한다. GPU wheel에 추가되는 CUDA 라이브러리까지 검증했다는 의미는 아니다. 사내 wheelhouse가 있다면 `python -m pip install --no-index --find-links /approved/wheels -r requirements-wafer.lock`을 사용한다. 패키지 변조 검증은 사내 저장소의 hash/signature 절차를 따른다. 이 lock 파일 자체에는 wheel hash가 없다.

공식 source는 기존 논문 asset 7.5 GB나 pretrained 모델을 받을 필요가 없다. 인터넷을 사용할 수 있는 승인된 준비 환경에서는 아래처럼 source만 가져올 수 있다. Windows도 줄바꿈 변경을 끄고 checkout해야 SHA-256이 일치한다.

```bash
git -c core.autocrlf=false clone --no-checkout https://github.com/voilalab/LACToS.git /approved/LACToS-source
git -C /approved/LACToS-source -c core.autocrlf=false checkout --detach d1ebf01dd268629bed18f3e3e9ae9c22c666f117
python -m wafer_diffusion doctor --upstream /approved/LACToS-source
```

학습·inference 명령은 source를 자동으로 받지 않는다. 두 공식 파일의 정확한 바이트를 SHA-256으로 검증하고, 저장소의 다른 Python 파일은 import하지 않는다. 파일이 바뀌면 실행을 거부한다. 라이선스 확인 사항은 [보안 검토](SECURITY_REVIEW.md)에 별도로 기록했다.

## 데이터 폴더와 split

가장 쉬운 구성은 정상 이미지 아래에 **전역적으로 유일한 lot+wafer 폴더**를 두는 것이다.

| 이미지 경로 예시 | `--group-parent`에서의 그룹 |
|---|---|
| `/data/normal/LOT_A_W01/a_preproc.png` | `LOT_A_W01` |
| `/data/normal/LOT_A_W01/b_preproc.png` | `LOT_A_W01` |
| `/data/normal/LOT_A_W02/c_preproc.png` | `LOT_A_W02` |
| `/data/normal/LOT_B_W01/d_preproc.png` | `LOT_B_W01` |

`W01`처럼 여러 lot에 반복되는 이름만으로 그룹을 정하지 않는다. 폴더 구조가 다르면 `--group-regex '(?P<group>LOT_[A-Z]+_W\d+)'`처럼 root-relative 경로에서 그룹을 추출한다. lot holdout을 원하면 그룹을 lot ID만으로 정한다. 불량 클래스별 여러 폴더를 `--anomaly-root`에 전달할 때도 동일한 lot/wafer를 같은 그룹으로 추출해야 한다.

```bash
python -m wafer_diffusion prepare --normal-root /data/normal --anomaly-root /data/defects --group-parent --include '*preproc*' --work-dir /work/wafer_exp01
```

- `--normal-root`, `--anomaly-root`는 공백으로 여러 경로를 받는다. 불량 데이터가 없으면 `--anomaly-root`를 생략한다.
- `--include`는 실제 파일명에 맞춘다. 예를 들어 `prepoc`라면 `'*prepoc*'`를 사용한다. 생략하면 지원되는 이미지 전체를 선택한다.
- 기본 split은 normal 그룹 기준 train 약 60%, val 약 20%, test 약 20%이다. 최소 3개 정상 그룹이 필요하다.
- 불량 이미지는 이미 정해진 그룹 split을 따른다. train 그룹에 속한 불량은 manifest에 남지만 **학습·평가 모두 사용하지 않는다**. 정상 데이터에 없는 불량 그룹은 test로 배정된다. 이 구성과 그 그룹의 분포를 확인한 후 성능을 해석한다.
- 같은 decoded image가 서로 다른 그룹/label에 존재하면 중단한다. 같은 그룹·label의 동일 이미지는 하나만 남긴다.
- `manifest.csv`, `manifest.summary.json`에 split과 사용하지 않는 train 불량 수를 기록한다. 학습 전에 이 결과를 확인한다. 파일 내용이 바뀌면 재검사에서 중단한다.
- Source dataset은 복사하지 않는다. Manifest는 원본 경로만 참조한다. Symlink 폴더는 순회하지 않고 중단하므로 실제 경로를 root로 지정한다.

## 단계별 실행

먼저 합성 데이터로 설치를 확인할 수 있다. 실제 wafer 데이터 경로를 받지 않는 명령이다.

```bash
python -m wafer_diffusion smoke --upstream /approved/LACToS-source --output /work/wafer_smoke
```

정상 데이터로 새 모델을 학습한다.

```bash
python -m wafer_diffusion train --manifest /work/wafer_exp01/manifest.csv --train-config configs/wafer_train.json --device cuda --output /work/wafer_exp01/model
```

`last.npz`는 최근 모델, `best.npz`는 normal validation denoising MSE가 가장 낮았던 EMA 모델을 포함한다. EMA는 학습 중 가중치를 이동평균한 복원용 모델이다. 이 MSE가 낮다고 결함 검출 성능이 높은 것은 아니다. `loss.jsonl`에서 학습 상태를 확인한다. 기본값인 50,000 steps는 초기 실험 설정이며 필요한 학습량을 보장하지 않는다. 먼저 설정 사본의 `max_steps`를 줄여 속도와 loss를 확인할 수 있다.

중단 후 재개하려면 같은 manifest와 모델 설정으로 `--resume /work/wafer_exp01/model/last.npz`를 추가한다. 목표 `max_steps`는 저장된 step보다 커야 한다. AdamW 상태, EMA, minibatch/noise RNG를 복원한다. 장치나 PyTorch가 바뀌었을 때 bit 단위 재현은 보장하지 않는다.

**검사 이미지 한 장**을 넣으면 reconstruction 생성과 scoring이 함께 실행된다.

```bash
python -m wafer_diffusion infer --checkpoint /work/wafer_exp01/model/last.npz --upstream /approved/LACToS-source --input /data/heldout/example_preproc.png --inference-config configs/wafer_inference.json --device cuda --output /work/wafer_exp01/preview --save-recon
```

각 이미지 폴더에는 원본을 전처리한 `input.png`, `recon_mean.png`, `score_map.npy`, `heatmap.png`, `overlay.png`, `result.json`이 생성된다. `--save-recon`을 켜면 복원 후보 각각도 저장한다. 기본값은 후보를 메모리에서 scoring하고 개별 PNG는 남기지 않아 용량을 줄인다. `scores.csv`에서 원본 경로와 image score를 연결할 수 있다.

Heatmap 색상은 보기 좋게 이미지별로 정규화한다. **판정에는 정규화 이전의 signed score를 사용한다.** 기본 image score는 픽셀 점수 상위 1%의 평균이다. 이는 이 파이프라인의 image-level 집계 규칙이며, 논문 Table 1의 pixel metric과는 다르다.

정상 validation 이미지로 threshold를 구하고, 학습·calibration에 사용하지 않은 test 그룹에서 평가한다.

```bash
python -m wafer_diffusion calibrate --checkpoint /work/wafer_exp01/model/last.npz --upstream /approved/LACToS-source --manifest /work/wafer_exp01/manifest.csv --inference-config configs/wafer_inference.json --device cuda --output /work/wafer_exp01/calibration --target-fpr 0.01
python -m wafer_diffusion evaluate --checkpoint /work/wafer_exp01/model/last.npz --upstream /approved/LACToS-source --manifest /work/wafer_exp01/manifest.csv --calibration /work/wafer_exp01/calibration/calibration.json --inference-config configs/wafer_inference.json --device cuda --output /work/wafer_exp01/test
```

`target-fpr=0.01`은 정상 validation score의 경험적 99% quantile을 사용한다. 미래 데이터에서 과검출 1%를 보장하지 않는다. 적은 validation 데이터로 작은 과검출률을 정밀하게 추정할 수 없다. 모델·draws·noise·scoring 설정이 바뀌면 이전 threshold 사용을 거부한다. GT mask가 없으므로 pixel AUROC/AP/AUPRO를 가짜 mask로 만들지 않으며, 불량 이미지가 없으면 image AUROC/AP와 미검출률도 `null`로 기록한다.

## 한 번에 실행

폴더 구조와 설정을 확인한 후 새 work directory에서 실행한다.

```bash
python -m wafer_diffusion run --normal-root /data/normal --anomaly-root /data/defects --group-parent --include '*preproc*' --work-dir /work/wafer_exp01 --upstream /approved/LACToS-source --train-config configs/wafer_train.json --inference-config configs/wafer_inference.json --device cuda --save-recon
```

Windows PowerShell에서도 같은 Python 명령을 한 줄로 사용하고 경로만 바꾼다. 예를 들어 `--normal-root 'D:\wafer\normal' --work-dir 'D:\wafer_runs\exp01' --upstream 'D:\approved\LACToS-source'`처럼 지정한다. Windows 자체 동작은 이번 검증에서 실행하지 않았다.

## 먼저 확인할 실험 결과

1. Train loss와 정상 validation loss가 내려가는지 본다.
2. 학습하지 않은 정상 wafer의 색·패턴이 reconstruction에서 유지되는지 본다.
3. 결함이 reconstruction에 그대로 남는지, 정상 부분이 크게 변하는지 본다.
4. Validation에서 `sigma`, `dps_scale`, `draws`, superpixel 크기를 조정한다. 최종 test 결과를 보고 조정하지 않는다.
5. 새 lot/wafer별 false alarm과 miss 사례를 직접 살핀다. 복원 품질이 나쁘면 threshold만 바꾸어 해결하려 하지 않는다.

Diffusion의 매 reverse step에 모델 계산과 입력 gradient 계산이 들어간다. 기본 1,000 steps × 5 draws이며 후보를 순차 생성해 메모리를 제한한다. DINOv2+MLP처럼 한 번의 forward로 끝나는 방식이 아니다. `result.json`의 실제 `seconds`를 측정한 뒤 검사 규모를 늘린다. 이번 검증은 CPU 기능 확인이고 실제 wafer 성능·GPU 처리량은 아직 측정하지 않았다.

## 기존 논문 재현과의 관계

기존 `scripts/`와 `paper-validation.yml`은 공개 benchmark 검증용이며 wafer 실행에서 호출하지 않는다. 해당 workflow는 공개 asset을 내려받고 로그를 GitHub에 남기므로 사내 데이터에 사용하지 않는다. 새 `wafer-ci.yml`은 합성 이미지 테스트만 하며 dataset·checkpoint·이미지 artifact를 업로드하지 않는다.

공식 source와 새 코드는 구분된다. 새 U-Net의 `.npz`를 공식 `sample_fast.py`에 바로 넣을 수 없고, 이 파이프라인의 `infer`로 사용해야 한다. 새 모델이 생성한 reconstruction은 공식 scoring에 직접 전달한다. Feature 기반 구성이 필요하면 이후 별도의 feature extractor/autoencoder 학습과 검토를 추가해야 한다.
