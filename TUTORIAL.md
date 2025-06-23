# Step-by-Step Tutorial: Cosine Annealing with Warmup Scheduler

Welcome! This hands-on tutorial will guide you through using the `cosine_annealing_warmup` scheduler in PyTorch, from setup to advanced usage. Each step is small and focused, with code, explanations, and expected output.

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

## 3. Minimal Steps: Learn by Small Units

### Step 1: Create the Scheduler Object

Let's just create the scheduler. No training, no optimizer step yet.

```python
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=5,
    cycle_mult=1,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=2,
    gamma=1.0
)
print('Scheduler created!')
```

**Expected output:**
```
Scheduler created!
```

---

### Step 2: Step the Scheduler Once

Let's see what happens to the learning rate after one step.

```python
print('Before step:', optimizer.param_groups[0]['lr'])
optimizer.step()
scheduler.step()
print('After step:', optimizer.param_groups[0]['lr'])
```

**Expected output:**
```
Before step: 0.01
After step: 0.0045  # (value will depend on warmup/cosine)
```

---

### Step 3: Try Without Warmup

Set `warmup_steps=0` and see the difference.

```python
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=5,
    cycle_mult=1,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=0,
    gamma=1.0
)
optimizer.step()
scheduler.step()
print('LR without warmup:', optimizer.param_groups[0]['lr'])
```

---

### Step 4: Observe a Full Cycle

Let's print the learning rate for each step in a cycle.

```python
lrs = []
for i in range(5):
    optimizer.step()
    scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])
print(lrs)
```

**Expected output:**
```
[0.01, 0.0081, 0.0045, 0.0019, 0.001]
```

---

### Step 5: Change cycle_mult and gamma

Try changing `cycle_mult` and `gamma` and see the effect.

```python
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=3,
    cycle_mult=2,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=1,
    gamma=0.5
)
lrs = []
for i in range(10):
    optimizer.step()
    scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])
print(lrs)
```

---

### Step 6: Integrate into a Simple Training Loop

Now, let's use the scheduler in a minimal training loop.

```python
for epoch in range(3):
    for step in range(5):
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch}, Step {step}, LR: {optimizer.param_groups[0]['lr']:.6f}")
```

---

## 4. More Practical Examples

### Example 1: No Warmup, Different Cycle Multipliers

```python
# No warmup, cycle_mult=1
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=20,
    cycle_mult=1,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=0,
    gamma=1.0
)

# No warmup, cycle_mult=2
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=10,
    cycle_mult=2,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=0,
    gamma=1.0
)
```

### Example 2: Using Different Optimizers

```python
# Using SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=20,
    cycle_mult=1,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=5,
    gamma=0.9
)

# Using AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=20,
    cycle_mult=1,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=5,
    gamma=0.9
)
```

### Example 3: Multiple Parameter Groups

```python
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 0.01},
    {'params': model.layer2.parameters(), 'lr': 0.001}
])
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=20,
    cycle_mult=1,
    max_lr=0.01,
    min_lr=0.0001,
    warmup_steps=5,
    gamma=0.8
)
```

### Example 4: Real Dataset (MNIST)

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

transform = transforms.ToTensor()
dataset = datasets.MNIST('.', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = torch.nn.Linear(28*28, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=100,
    cycle_mult=1,
    max_lr=0.01,
    min_lr=0.001,
    warmup_steps=10,
    gamma=0.9
)

for epoch in range(5):
    for images, labels in dataloader:
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        output = model(images)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

### Example 5: Saving and Loading Scheduler State

```python
# Save
torch.save(scheduler.state_dict(), 'scheduler.pt')
# Load
scheduler.load_state_dict(torch.load('scheduler.pt'))
```

### Example 6: Error Handling

```python
try:
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=0,  # Invalid, must be > 0
        cycle_mult=1,
        max_lr=0.01,
        min_lr=0.001,
        warmup_steps=0,
        gamma=1.0
    )
except Exception as e:
    print(f"Error: {e}")
```

---

**Now try modifying the code, experiment, and apply the scheduler to your real projects!**
