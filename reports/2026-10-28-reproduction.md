# LACToS 재현 결과 — 2026-10-28

**GitHub Actions에서 실제 실행한 세 단계가 모두 통과했습니다.** 아래 수치는 실행 로그의 결과이며, 전체 정밀도와 환경 정보는 [verification.json](2026-10-28/verification.json)에 보존했습니다.

- [성공한 실행 #2](https://github.com/yujinkimmn/LACToS/actions/runs/40622745437) · [전체 작업 로그](https://github.com/yujinkimmn/LACToS/actions/runs/40622745437/job/122600406445)
- 검증한 코드: [`7f88c317e75ab7350d636a445eab3107c1888fc2`](https://github.com/yujinkimmn/LACToS/commit/7f88c317e75ab7350d636a445eab3107c1888fc2)
- 공식 구현: [voilalab/LACToS @ d1ebf01](https://github.com/voilalab/LACToS/tree/d1ebf01dd268629bed18f3e3e9ae9c22c666f117). 실행 시 고정 커밋을 가져왔으며 공식 소스는 수정하지 않았습니다.
- 실행: 2026-10-28 02:44:57–02:58:32 UTC (한국 시간 11:44:57–11:58:32), 약 13분 35초.
- [원본 텍스트 결과 artifact](https://github.com/yujinkimmn/LACToS/actions/runs/40622745437/artifacts/12788845848): 15,508 bytes, 7일 보관. 이 보고서와 JSON은 저장소에 남습니다.

## 검증한 범위

| 단계 | 실제 수행 | 관측 결과 |
|---|---|---|
| reference | 저자 제공 CHAOS·FFHQ·DVXray score map에서 7개 방법의 Table 1 지표 재계산 | 63/63개 셀 소수점 4자리 일치 |
| xray | 논문 DVXray의 정상 72장과 scissors 72장 전부를 공식 LACToS 코드로 새로 scoring | 144장 처리, 실패 0장; 3개 지표 소수점 4자리 일치 |
| ffhq | 논문 FFHQ의 2장에 대해 특징 추출·autoencoder·intensity 결합 경로 실행 | 두 경로 모두 2장 처리, 실패 0장; 참조 결과와 차이 ≤ 0.005 |

reference는 저장된 점수 맵에서 지표만 재계산한 것입니다. xray와 ffhq는 저자가 제공한 diffusion reconstruction으로 **점수 맵을 새로 생성**했습니다. Diffusion sampling과 모델 학습은 다시 실행하지 않았습니다.

## DVXray 전체 144장

이미지당 reconstruction 20개(총 2,880개)를 사용했습니다. 논문 설정인 intensity-only `lambda=0`, histogram bins 16, histogram smoothing 0, superpixel target 10, spatial sigma 5를 유지했습니다.

아래는 pooled pixel-level 지표이며, AUPRO는 공식 코드의 FPR ≤ 0.3 설정입니다. 논문 값은 고정된 공식 소스의 [expected/table1.json](https://github.com/voilalab/LACToS/blob/d1ebf01dd268629bed18f3e3e9ae9c22c666f117/expected/table1.json)과 비교했습니다.

| 지표 | 논문 | 이번 실행 | 절대 차이 |
|---|---:|---:|---:|
| AUROC | 0.9607 | 0.9606820481 | 0.0000179519 |
| AP | 0.0568 | 0.0567774365 | 0.0000225635 |
| AUPRO | 0.8179 | 0.8178839383 | 0.0000160617 |

세 지표 모두 논문의 표시 정밀도인 소수점 4자리까지 같습니다. 새 scoring과 평가에 약 280.65초가 걸렸습니다. 공식 코드가 개별 오류를 처리하고 exit 0으로 끝낼 수 있어, 성공한 종료 코드 외에도 실제 처리 개수와 실패 목록을 검사했습니다.

## FFHQ 특징 경로 검증 — 2장

`faces`의 정렬상 첫 두 샘플인 `samples_1`, `samples_10`을 사용했습니다. ResNet-101과 저자 제공 PixelAutoEncoder(1792→3)를 사용하여 full 및 intensity-only 점수를 각각 새로 계산하고, `rgb + 0.25 × (full − rgb)`로 결합했습니다. Spatial sigma 5와 direct AE encoding은 유지했습니다.

CPU 메모리를 고려해 embedding batch를 논문의 4에서 1로 변경했습니다. 비교 대상은 **같은 두 이미지의 저자 참조 결과**입니다.

| 지표 | 동일 2장 참조 | 이번 실행 | 절대 차이 |
|---|---:|---:|---:|
| AUROC | 0.9808262394 | 0.9808792527 | 0.0000530133 |
| AP | 0.2377311869 | 0.2389516240 | 0.0012204371 |
| AUPRO | 0.9187269429 | 0.9199063009 | 0.0011793581 |

모든 차이가 허용 오차 0.005 이내였고 약 181.38초가 걸렸습니다. 이 수치는 **FFHQ 전체 100장의 논문 벤치마크 성능이 아닙니다.**

## 실행 환경과 데이터

- GitHub-hosted Ubuntu 22.04, CPU, 2 threads.
- Python 3.10.22; torch 2.0.1+cpu; torchvision 0.15.2+cpu; numpy 1.24.3; scipy 1.10.1; scikit-image 0.19.2; scikit-learn 1.2.2.
- [저자 공식 asset bundle 안내](https://github.com/voilalab/LACToS/blob/d1ebf01dd268629bed18f3e3e9ae9c22c666f117/docs/BUNDLE.md)의 데이터와 reconstruction을 사용했습니다.
- 총 7,481,497,600 bytes를 스트리밍했고 tar 파일은 저장하지 않았습니다. 필요한 11,377개 파일, 1,487,041,208 bytes만 임시 runner에 추출했습니다.
- 다운로드 SHA-256: `d46969f2ee9c89bcbf7dd286b9dfd930e07586a5c704ec9defb1cbd3c7d78e21`. 이번 다운로드에서 계산한 기록용 값이며, 저자 공인 checksum과 대조한 값은 아닙니다.
- 데이터·reconstruction·가중치·raw score map은 Git이나 artifact에 포함하지 않았습니다. 임시 데이터 삭제 단계도 성공했습니다.
- 데이터 다운로드와 계산은 GitHub Actions에서 수행했습니다. 사용자의 PC 디스크에 데이터셋을 저장하지 않았습니다.

## 해석과 재실행

이번 실행은 논문 데이터에서 LACToS scoring 및 평가 코드가 동작하고, DVXray 전체 평가 수치가 재현됨을 확인합니다. CHAOS·FFHQ 전체 데이터에 대한 새 scoring, diffusion 생성, 학습, wafer 성능까지 검증한 결과로 해석하지 않습니다.

재실행하려면 [Actions](https://github.com/yujinkimmn/LACToS/actions/workflows/paper-validation.yml)에서 **Run workflow → all**을 선택합니다. 각 실행의 실제 로그와 Summary를 확인할 수 있습니다. 데이터 다운로드 스크립트는 GitHub Actions 밖에서는 실행을 중단합니다.
