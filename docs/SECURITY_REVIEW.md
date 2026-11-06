# Wafer 파이프라인 보안 검토 — 2026-11-06

## 확인한 범위

- 공식 source commit: `voilalab/LACToS@d1ebf01dd268629bed18f3e3e9ae9c22c666f117`.
- Python/shell/requirements/config 등 93개 텍스트 파일의 정적 패턴 스캔. 학습, sampler, checkpoint loader, embedding, scorer와 entry point의 관련 경로를 직접 읽었다.
- 새 `wafer_diffusion/`의 학습·DPS·입출력·공식 코드 연결 경로, synthetic 실행, 네트워크/파일 변조/역직렬화 거부 테스트.
- Linux/Python 3.12 CPU runtime 의존성 19개를 `pip-audit`로 검사. PyTorch의 `+cpu` suffix를 base version으로 정규화해 audit에서 누락되지 않게 했다.
- 모든 종속 라이브러리의 소스 전체, CUDA native 코드, 회사 서버·OS·보안 정책, 실제 wafer 데이터는 검토 범위가 아니다. 자동 검사 통과가 악성 코드 부재나 보안 승인을 증명하지 않는다.

## 원본에서 발견한 동작과 적용한 처리

| 항목 | 확인한 위치 | 새 wafer 경로의 처리 |
|---|---|---|
| Pretrained ResNet 자동 다운로드 | `ood/embeddings.py:68`; `evaluate.py:311–317`에서 `use_pca` 확인 전에 생성 | `evaluate.py` 및 embeddings를 import하지 않는다. Feature 없는 공식 score만 직접 호출한다. |
| Pickle 기반 checkpoint 로딩 | `evaluate.py:55`, `ood/samplers/io.py:72,122`, `sampling/dps/guided_diffusion/unet.py:88` | 원본 pretrained checkpoint를 사용하지 않는다. 새 모델은 숫자 tensor 전용 NPZ, `allow_pickle=False`, archive 크기·header·shape·dtype 검증으로 로드한다. |
| Baseline의 명시적 unsafe 역직렬화 | `baselines/infer_feature.py:50,105,122,138,140`의 pickle 및 `weights_only=False` | Baseline 코드는 실행하거나 import하지 않는다. |
| 추가 repository·asset 다운로드 | `scripts/fetch_third_party.sh`, README의 asset 다운로드 명령, 기존 paper workflow | Wafer runtime에서는 호출하지 않는다. 승인된 별도 준비 단계에서 source/의존성만 반입한다. |
| 넓은 dependency 및 오래된 재현 환경 | 원본 requirements의 torch 2.0.1 등 | 검증한 별도 환경과 lock 사용. 이번 초기 설치에서 검출된 Pillow/설치 도구 취약점을 수정한 후 다시 검사했다. |
| Histogram int16 overflow 및 메모리 fallback | `ood/scoring.py` | 실제 superpixel 픽셀 수 × draw 수가 32767을 넘으면 중단. Dense histogram 한도를 넘겨 다른 계산 경로로 조용히 바뀌는 것도 차단한다. |
| Source 변경 후 실행 | 런타임 upstream 파일 | 두 파일을 SHA-256 allowlist로 검증하고 검증한 바이트만 실행한다. 다른 파일이나 package initializer는 실행하지 않는다. |

스캔하고 읽은 경로에서는 wafer를 외부로 전송하는 명시적인 코드나 telemetry를 발견하지 않았다. 다만 원본 환경 전체의 안전성을 보증하는 결론은 아니다. 자세한 파일 목록과 hash는 `reports/wafer/upstream_scan.json`에 기록한다.

## 새 실행 경로의 데이터 보호

- 학습은 외부 checkpoint 없이 처음부터 진행한다. API key, 로그인, W&B, MLflow, Hugging Face client가 필요하지 않다.
- `python -m wafer_diffusion`은 다른 모듈을 import하기 전에 Python audit hook으로 DNS/socket 연결과 외부 프로세스 실행을 거부한다. 설치 명령은 별도 준비 단계다.
- 이 hook은 Python 수준의 방어이며 OS network sandbox는 아니다. Native extension의 임의 system call까지 차단한다고 주장하지 않는다. 사내 실행에서는 기존 서버 egress 차단·권한 정책을 유지해야 한다.
- 파일 확장자와 Pillow decoder allowlist, file size/pixel 수를 확인한다. EPS를 PNG로 위장해 Ghostscript를 실행하는 경로를 허용하지 않는다. 16-bit/DAT를 잘못 해석해 결과를 왜곡하는 변환도 거부한다.
- 사용자가 명시한 image roots만 읽는다. Symlink tree 순회, URL, UNC 입력은 거부한다. Mounted filesystem의 원격 여부까지 판별하는 기능은 아니다.
- 출력은 Git repository 밖의 work directory에만 허용한다. 추가 `.gitignore`는 실수 방지용이며 Git 강제 추가나 다른 도구의 업로드까지 막는 접근 제어는 아니다.
- Hosted GitHub Actions에서 wafer 데이터 명령을 거부한다. CI는 코드로 만든 합성 데이터만 사용하고 이미지/가중치 artifact 업로드 단계가 없다.
- Checkpoint 자체와 reconstruction, manifest, CSV에도 데이터 특성이 포함될 수 있으므로 사내 자료와 같은 저장·접근 정책을 적용한다.

## 실행하는 공식 파일의 SHA-256

| 파일 | SHA-256 |
|---|---|
| `ood/scoring.py` | `c962c7078091eb569a912bcdf4f1e6648823a674c92a9af24c8ef241c4f85f34` |
| `ood/superpixels.py` | `bd48e73d2d5e053f7768041f08617654725f4b09ab565696f559dcf5ec1b1adf` |

`scoring.py` 원본은 마지막 newline이 없으므로 편집기가 newline을 추가해도 검증에 실패한다. 파일을 수정해 hash 검사만 통과시키는 방식 대신, 변경 내용을 재검토하고 테스트한 다음 allowlist를 갱신해야 한다.

## 의존성 검사 결과

초기 검사에서 Pillow 12.3.0과 pip/setuptools 구버전의 advisory가 검출됐다. Pillow 13.0.0, pip 26.3, setuptools 84.2.1로 수정했다. 최종 `requirements-wafer.lock`의 19개 package에 대해 알려진 취약점 **0건**, 누락된 package **0개**를 확인했다. 원시 결과는 `reports/wafer/dependency_audit.json`에 있다. 결과는 2026-11-06 조회 기준이며 향후 advisory가 추가될 수 있다.

이 검사는 package/version에 대한 공개 advisory 조회다. Wheel의 무결성·배포망 침해·학습 데이터 poisoning·모델 품질을 검증하지 않는다. 특히 GPU wheel과 추가 CUDA dependency는 실제 사내 환경에서 다시 확인해야 한다.

## 라이선스는 별도 확인 사항

검토한 공식 commit에는 repository 최상위 LICENSE가 없고 GitHub metadata의 license도 지정되지 않았다. 논문의 공개 라이선스와 코드·third-party dependency의 이용 조건을 동일하게 가정하지 않는다. 공식 코드 두 파일을 이 저장소에 복사하거나 재라이선스하지 않았으며, 승인된 로컬 source 경로를 연결한다. 회사에서 사용하기 전 해당 코드의 이용 권한을 확인해야 한다. 보안 검사 통과가 라이선스 허용을 뜻하지 않는다.

## 검증의 한계

합성 이미지로 실제 gradient update, optimizer/EMA 재개, 독립적인 DPS reconstruction, 공식 signed intensity scoring, calibration, test 평가를 확인했다. 해시 변조·pickle/object array·과도한 tensor header·외부 연결·외부 프로세스·데이터 중복을 거부하는 테스트도 실행했다. 기본 256×256 architecture의 forward/backward를 별도로 확인했다.

실제 wafer 데이터 학습, defect 검출 성능, GPU memory/latency, Windows 실행은 아직 검증하지 않았다. 현재 구성은 intensity-only이고 ResNet feature/autoencoder 경로는 포함하지 않는다.

참고: [공식 LACToS](https://github.com/voilalab/LACToS), [PyTorch security advisories](https://github.com/pytorch/pytorch/security/advisories), [Pillow 13.0.0 release notes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html).
