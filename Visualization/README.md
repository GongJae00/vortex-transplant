# Visualization

데이터가 어떻게 생겼는지 눈으로 확인할 수 있는 도구들.

## 빠른 시작

```bash
python Visualization/demo.py
```

`Visualization/output/`에 PNG 이미지 + ASCII 텍스트 파일 생성.

## 생성되는 파일

| 파일 | 내용 |
|------|------|
| `01_synthetic_vortex_decomposition.png` | 깨끗한 vortex pair의 charge/phase/decomposition |
| `01_synthetic_vortex.txt` | ASCII charge map + phase arrows |
| `02_untrained_8channels.png` | 실제 untrained U1ConvRNN 8채널 charge |
| `02_untrained_model.txt` | ASCII charge map |
| `03_u1_vs_plain.png` | U1 vs Plain charge 비교 |
| `03_u1_vs_plain.txt` | Defect count 비교 |
| `04_branch_margins.png` | Branch margin (clean vs random field) |

## 모듈 사용법

```python
from Visualization.visualize import (
    plot_charge_map,       # (H,W) charge → image
    plot_phase_map,        # (H,W) phase → image  
    plot_magnitude_map,    # (H,W) magnitude → image
    plot_decomposition,    # full decomposition panel
    plot_multichannel_charge,  # multi-channel grid
    charge_to_ascii,       # charge → text
    phase_to_arrows,       # phase → Unicode arrows
)
```
