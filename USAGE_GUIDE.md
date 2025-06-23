# Cosine Annealing with Warmup 사용 가이드

이 문서는 `cosine_annealing_warmup` 스케줄러를 효과적으로 사용하는 방법을 단계별로 안내합니다. 다양한 예시와 함께 실전 적용법을 설명합니다.

---

## 1. 프로젝트 클론 및 설치

먼저, GitHub에서 프로젝트를 클론하세요.

```bash
git clone https://github.com/your-username/pytorch-cosine-annealing-with-warmup.git
cd pytorch-cosine-annealing-with-warmup
```

의존성 설치:

```bash
pip install -r requirements.txt
```

또는 패키지로 설치:

```bash
pip install .
```

---

## 2. 기본 사용법

### 2.1 PyTorch에서 스케줄러 사용하기

```python
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

model = ...  # 모델 정의
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=100,
    cycle_mult=2,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=10,
    gamma=0.5
)

for epoch in range(epochs):
    for batch in dataloader:
        # ... 학습 코드 ...
        optimizer.step()
        scheduler.step()
```

---

## 3. 주요 파라미터 설명

- `first_cycle_steps`: 첫 번째 사이클의 step 수
- `cycle_mult`: 다음 사이클의 step 수를 곱하는 계수
- `max_lr`: 사이클 내 최대 learning rate
- `min_lr`: 사이클 내 최소 learning rate
- `warmup_steps`: 워밍업에 사용할 step 수
- `gamma`: 각 사이클마다 learning rate를 곱하는 계수

---

## 4. 다양한 예시

### 4.1 워밍업 없이 사용하기

```python
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=50,
    cycle_mult=1,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=0,
    gamma=1.0
)
```

### 4.2 여러 번의 사이클로 반복

```python
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=30,
    cycle_mult=2,
    max_lr=0.005,
    min_lr=0.0005,
    warmup_steps=5,
    gamma=0.8
)
```

### 4.3 Step마다 learning rate 확인하기

```python
lrs = []
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
```

---

## 5. 시각화 예시

학습률 변화를 시각화하려면:

```python
import matplotlib.pyplot as plt
plt.plot(lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing with Warmup')
plt.show()
```

---

## 6. 자주 묻는 질문(FAQ)

- **Q: 워밍업이란?**
  - A: 학습 초기에 learning rate를 점진적으로 증가시켜 안정적인 학습을 돕는 기법입니다.
- **Q: 여러 optimizer에 적용할 수 있나요?**
  - A: 네, PyTorch의 모든 optimizer에 적용 가능합니다.

---

## 7. 참고 자료

- [공식 PyTorch 문서](https://pytorch.org/docs/stable/optim.html)
- [Cosine Annealing 논문](https://arxiv.org/abs/1608.03983)

---

문의사항은 이슈로 남겨주세요.
