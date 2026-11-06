# Wafer 파이프라인 검증 — 2026-11-06

**실제 wafer 데이터는 사용하지 않았다.** 합성 이미지로 코드의 학습·sampling·scoring 기능과 안전장치가 동작하는지 검증했다. 이 결과를 defect 검출 성능이나 논문 재현 수치로 해석하지 않는다.

| 검사 | 관찰한 결과 |
|---|---|
| Unit/integration tests | 24 passed, 0 failed, 0 skipped |
| 실제 gradient update와 checkpoint resume | 성공. 3 steps 학습 후 4번째 step으로 재개해 가중치 변경 확인 |
| Resume 재현성 | 별도 테스트에서 CPU 연속 학습과 재개 학습의 model/EMA/optimizer 배열이 정확히 일치 |
| DPS reconstruction | 이미지당 2개 후보 생성. 동일 seed 반복 결과 일치, 서로 다른 draw 결과 구분 |
| 공식 LACToS scoring | 해시 검증한 원본 파일로 실행, finite map 확인. 작은 histogram에서 계산한 NLL−entropy 해석값과 일치 |
| Calibration와 test 평가 | 정상 val로 threshold 계산 후 분리된 test 6장 처리. 수치는 기능 확인용이므로 성능 결과로 제시하지 않음 |
| 기본 architecture | 256×256 forward/backward와 finite gradient 확인 |
| 보안 거부 경로 | source 변조, object/pickle array, 압축 archive, 과도한 tensor shape, network/process 호출, 위장 EPS, 중복/변경 이미지 거부 |
| Dependency audit | runtime package 19개, 알려진 취약점 0건, audit skip 0개 |

최소 smoke model은 16×16, base channels 8, attention off, 8-step cosine diffusion을 사용했다. 실제 설정의 256×256 / 1,000-step 모델이 충분히 학습됐다는 검증이 아니다. CPU PyTorch 2.15.0, Python 3.12, Pillow 13.0.0 환경이며 `requirements-wafer.lock`에 나머지 버전을 기록했다.

원시 증거:

- [테스트 이름과 상태](tests.json)
- [합성 end-to-end 실행](synthetic_smoke.json)
- [의존성 audit](dependency_audit.json)
- [upstream 파일 스캔 범위](upstream_scan.json)

CUDA 학습, GPU latency/memory, Windows, 실제 wafer generalization은 미검증이다. GitHub workflow는 같은 테스트를 공개 합성 데이터로 반복하도록 구성했다. 원본 코드의 feature/autoencoder 경로와 pretrained checkpoint는 사용하지 않는다.
