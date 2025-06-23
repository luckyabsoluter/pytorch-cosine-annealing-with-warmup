# Step-by-Step Tutorial: Cosine Annealing with Warmup Scheduler

Welcome! This hands-on tutorial will guide you through using the `cosine_annealing_warmup` scheduler in PyTorch, from setup to advanced usage. Each step includes code, explanations, and practical exercises.

---

## 1. Prerequisites

- Basic Python and PyTorch knowledge
- Python 3.7+
- Git

---

## 2. Environment Setup

**a. Clone the repository:**
```bash
git clone https://github.com/your-username/pytorch-cosine-annealing-with-warmup.git
cd pytorch-cosine-annealing-with-warmup
```

**b. (Optional) Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**c. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 3. Your First Cosine Annealing Scheduler

Let's use the scheduler in a minimal PyTorch training loop.

```python
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=20,
    cycle_mult=1,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=5,
    gamma=1.0
)

for epoch in range(30):
    optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch+1}, LR: {optimizer.param_groups[0]['lr']:.6f}")
```

**Try running this code and observe the learning rate changes!**

---

## 4. Experiment: Change the Parameters

- Change `warmup_steps` to 0 and see the effect.
- Set `cycle_mult=2` and run for 60 epochs. What happens to the cycle length?
- Try different `gamma` values (e.g., 0.5) and observe the learning rate decay.

**Exercise:**
- Plot the learning rate schedule using matplotlib.

```python
import matplotlib.pyplot as plt
lrs = []
for epoch in range(60):
    optimizer.step()
    scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])
plt.plot(lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing with Warmup')
plt.show()
```

---

## 5. Integrate with Your Own Training Loop

Replace the optimizer and scheduler in your own PyTorch project with the above code. Make sure to call `scheduler.step()` after each optimizer step.

**Tip:**
- You can use any PyTorch optimizer (SGD, Adam, etc).
- The scheduler is compatible with most training loops.

---

## 6. Common Pitfalls & FAQ

- **Q: Why is my learning rate not changing?**
  - Make sure you call `scheduler.step()` after every `optimizer.step()`.
- **Q: Can I use this with multiple parameter groups?**
  - Yes, all groups will be updated.
- **Q: How do I resume training?**
  - Save and load the scheduler state dict with `scheduler.state_dict()` and `scheduler.load_state_dict()`.

---

## 7. Further Reading

- [PyTorch Optimizer Docs](https://pytorch.org/docs/stable/optim.html)
- [Cosine Annealing Paper](https://arxiv.org/abs/1608.03983)

---

**Now try modifying the code, experiment, and apply the scheduler to your real projects!**
