# LACToS 논문 실행 검증

사용자 검증 저장소: `yujinkimmn/LACToS`. 논문 저자의 공식 저장소는 아래의 `voilalab/LACToS`입니다.

[공식 LACToS](https://github.com/voilalab/LACToS)의 코드를 고정된 커밋으로 가져와 **GitHub Actions에서만** 검증합니다. 사용자의 PC에는 데이터셋을 다운로드하지 않습니다.

## 실행 범위

| 단계 | 데이터 / 실행 | 통과 조건 |
|---|---|---|
| `reference` | CHAOS, FFHQ, DVXray의 저자 제공 score map으로 Table 1 재계산 | 63개 셀 모두 소수점 4자리 일치 |
| `xray` | 논문의 DVXray 정상 72장 + scissors 72장, 이미지당 저자 제공 reconstruction 20개를 공식 코드로 새로 scoring | 144장 누락·실패 없음, AUROC/AP/AUPRO의 논문 대비 차이 ≤ 0.005 |
| `ffhq` | 논문 FFHQ `faces`의 첫 2장, ResNet-101 + 제공 autoencoder, intensity/feature 결합 | 두 이미지의 참조 결과 대비 AUROC/AP/AUPRO 차이 ≤ 0.005 |

`reference`는 저자가 계산한 맵의 **지표 재계산**입니다. `xray`와 `ffhq`는 제공 reconstruction으로 **점수 맵을 새로 계산**합니다. Diffusion sampling, prior 학습, wafer 데이터 성능은 검증 범위에 포함되지 않습니다. 본 실험의 diffusion/autoencoder 학습 코드는 공식 저장소에 포함되어 있지 않습니다.

DVXray는 논문 설정 자체가 intensity-only (`lambda=0`)입니다. FFHQ는 논문의 `lambda=0.25`, spatial sigma 5, direct AE encoding을 유지하되 CPU 메모리 제한으로 embedding batch만 4→1로 줄입니다. 2장 결과를 FFHQ 전체 100장 성능으로 해석하지 않습니다. 공식 코드의 signed `NLL − entropy` 점수를 그대로 사용합니다.

## GitHub에서 실행

1. 이 저장소의 **Actions → LACToS paper validation → Run workflow**에서 `all`을 선택합니다.
2. 실행 페이지의 Summary와 각 단계 로그를 확인합니다.
3. `lactos-paper-report-<run_id>` artifact에는 요약·지표·환경·로그만 들어갑니다. 웹에서 Summary를 확인하면 사용자 PC에 다운로드할 필요가 없습니다.

첫 `main` 업로드 시에도 자동 실행됩니다. 데이터 관련 스크립트는 GitHub Actions 밖에서 실행하면 다운로드 전에 중단합니다. 검증 결과가 없는 단계는 PASS로 표시하지 않습니다. 공식 `evaluate.py`가 예외를 잡고 exit 0으로 끝나는 경우도 있어, 래퍼가 실패 목록과 실제 처리 개수를 별도로 검사합니다.

## 저장 공간

- 데이터는 GitHub Actions의 임시 runner에만 둡니다.
- 약 7.5GB의 저자 공개 tar를 스트리밍하고, 필요한 부분만 추출합니다. tar 자체는 저장하지 않습니다.
- baseline/diffusion 학습 가중치는 건너뜁니다.
- 데이터, 가중치, reconstruction, raw score map은 Git 커밋과 Actions artifact에 포함하지 않습니다.
- 실행 후 임시 자료를 삭제합니다. 텍스트 결과 artifact 보관 기간은 7일입니다.
- 데이터 원본을 GitHub 저장소나 Git LFS에 업로드하지 않습니다.

## 출처와 재현성

- 공식 구현: `voilalab/LACToS`, commit `d1ebf01dd268629bed18f3e3e9ae9c22c666f117` (`upstream.json`).
- 논문: [LACToS: Local Anomaly Detection via Conditional Typicality of Superpixels](https://arxiv.org/abs/2610.31538).
- 저자 데이터·체크포인트: [공식 asset 안내](https://github.com/voilalab/LACToS/blob/d1ebf01dd268629bed18f3e3e9ae9c22c666f117/docs/BUNDLE.md).
- 데이터의 구성, 합성 이상, 선택 방법: [PROVENANCE.md](https://github.com/voilalab/LACToS/blob/d1ebf01dd268629bed18f3e3e9ae9c22c666f117/docs/PROVENANCE.md).
- Python 3.10, torch 2.0.1 / torchvision 0.15.2 CPU와 공식 requirements를 사용합니다. 환경 freeze, 고정 commit, 다운로드 SHA-256, 실행 결과를 기록합니다.

공식 코드는 `vendor/`에 실행 시 가져오며 수정하거나 이 저장소의 라이선스로 재배포하지 않습니다. 데이터와 가중치의 이용 조건은 각각의 원 출처를 따릅니다. 논문에서 선택한 하이퍼파라미터를 고정해서 재현하며 추가 test-set 튜닝을 하지 않습니다.
