# pop
Search &amp; optimalization classes @ WUT

## How to run the program?

### 1. Training model

```bash
python3 src/main.py train --episodes <num_of_episodes>
```

### 2. Test the model

```bash
python3 src/main.py eval
```

### 3. Test logic without AI model

```bash
python3 src/main.py eval --autopilot
```

### 4. Visualisation

We've created a --render flag which enables display of the actual actions and state of The Santa during evaluation.

```bash
python3 src/main.py eval --render
```

### 5. Genetic model

You can also train a genetic model using

```bash
python3 src/train_genetic.py
```