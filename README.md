# Homework

## Contents

- `hw1/` (40 pts): implement and benchmark operations with different arithmetic intensity.
- `hw2/` (60 pts): profile and optimize an autoregressive generation loop.
- `hw3/` *(optional, ungraded)*: build the memory and scheduling core of a mini LLM inference engine.

HW1 and HW2 together add up to **100 points**. See each subfolder's `README.md` for the per-part point breakdown and the expected submission format.

## Setup

```bash
sudo apt-get install -y python3-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

Use a fresh virtualenv for this repo. Reusing an older environment with extra packages can create version conflicts with the pinned dependencies.

See the `README.md` inside each subfolder for task details, requirements, and expected outputs.
