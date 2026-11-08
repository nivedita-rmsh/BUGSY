# BUGSY — Autonomous Navigation & Exploration Module

> **Navigation & Exploration Component (Kezia)**
> 
> This module implements the autonomous navigation and exploration policy for the BUGSY project using **Deep Reinforcement Learning (PPO)** and **count-based spatial novelty rewards**.
> 
> While this agent explores the 3D parkour environment, the companion **Anomaly Detection Module** (Nivedita, documented in [`evaluation/`](../evaluation/)) independently monitors physical state invariants (velocity, position continuity, wall collisions) to detect game physics bugs.

---

## 1. Overview & System Role

Automated game testing requires an agent capable of exploring a large, complex, obstacle-dense environment autonomously without relying on handcrafted scripted paths:

- **Navigation Module (Kezia)**: Trains a PPO agent driven by intrinsic exploration rewards to maximize spatial cell discovery across the 3D maze.
- **Detection Module (Nivedita)**: Continuously evaluates whether the agent's observed physics (velocity, trajectory, collision bounds) remain valid, logging alerts and evaluating fault injection scenarios.
- **Shared Unity Core (`Assets/`)**: The unified 3D parkour world with dynamic hazards (lasers, boulders, pendulums, spinners) where both modules operate seamlessly together.

---

## 2. Directory Structure

```text
navigation/
├── baseline-results/               # 100-episode CSV evaluation traces
│   ├── baseline_random.csv         # Uniform random baseline
│   ├── baseline_wallfollower.csv   # Scripted right-hand rule baseline
│   ├── baseline_v8.csv             # PPO agent (training maze, seed 21)
│   ├── baseline_v8_unseen.csv      # PPO agent generalization (unseen maze, seed 7)
│   ├── baseline_v9.csv             # Extended training checkpoint
│   ├── baseline_nomap.csv          # Ablation: novelty reward without visited map
│   └── baseline_nomap300.csv       # Ablation run (300 episodes)
├── figures/                        # Generated evaluation charts
│   ├── fig1_cumulative_unique.png  # Cumulative cell discovery over 100 episodes
│   ├── fig2_coverage_improved.png  # Mean cells/episode & total coverage %
│   ├── fig3_success_rate.png       # Success rates (≥49 and ≥25 cells)
│   ├── fig4_generalization_gap.png # Generalization to unseen maze layout
│   ├── fig5_coverage_distribution.png # Distribution quartiles & outliers
│   └── fig11_cumulative_regret_difference.png # Regret reduction vs random
├── models/                         # Exported ONNX neural network models
│   ├── BugsyAgent_v8.onnx          # Primary evaluation model (assigned in Unity)
│   └── BugsyAgent_v9.onnx          # Extended iteration checkpoint
├── BUGSY_Parkour_RL_Explanation.md # Full line-by-line code explanation & RL mathematical theory
├── bugsy_ppo.yaml                  # ML-Agents PPO training configuration
├── make_figures.py                 # Statistical analysis and figure generation script
└── README.md                       # This documentation
```

---

## 3. RL Architecture & Implementation

### 3.1 Algorithm: Proximal Policy Optimization (PPO)

The agent policy is trained using Unity ML-Agents PPO with a 2-layer MLP (128 hidden units per layer).

Configuration (`bugsy_ppo.yaml`):

| Hyperparameter | Value | Description |
|---|---|---|
| `trainer_type` | `ppo` | On-policy actor-critic algorithm |
| `batch_size` | 256 | Minibatch size for policy updates |
| `buffer_size` | 4096 | Rollout buffer capacity (48 gradient updates/buffer) |
| `learning_rate` | 0.0003 | Linear decay schedule |
| `epsilon` | 0.2 | PPO surrogate clipping threshold |
| `lambd` | 0.95 | Generalized Advantage Estimation (GAE) parameter |
| `beta` | 0.005 | Entropy regularization coefficient |
| `gamma` | 0.995 | Long-horizon discount factor |
| `normalize` | `true` | Running-mean/variance observation normalization |
| `max_steps` | 300,000 | Total training budget |

### 3.2 Intrinsic Exploration Reward

Rather than depending on a single sparse terminal goal, BUGSY utilizes **count-based spatial novelty**:

$$r_t = \beta \cdot \mathbf{1}[g(s_{t+1}) \notin V_t]$$

- The maze floor is partitioned into discrete $3 \times 3\text{ m}$ spatial cells ($g(s)$).
- $V_t$ denotes the set of cells already visited during episode $t$.
- When the player enters a previously unvisited cell, an intrinsic reward of $\beta = 0.1$ is awarded.
- Implemented in [`Assets/Scripts/CoverageTracker.cs`](../Assets/Scripts/CoverageTracker.cs).

### 3.3 Observation Space (43 Dimensions)

| Component | Dimensions | Representation |
|---|:---:|---|
| Normalized Position | 2 | $(x / 66,\ z / 66)$ mapped to $[0, 1]$ |
| Local Linear Velocity | 3 | Rigidbody velocity in agent local frame / 10 |
| Heading Direction | 2 | Continuous $(\sin\theta,\ \cos\theta)$ to prevent angle discontinuity |
| Physical State Flags | 2 | Binary grounded & sliding booleans |
| Previous Action Feedback | 9 | One-hot encodings: move (2), jump (2), slide (2), turn (3) |
| Local Visited-Cell Grid | 25 | $5 \times 5$ local spatial window around agent ($0 = \text{unvisited}, 1 = \text{visited/wall}$) |

An additional 11-ray `RayPerceptionSensor3D` detects walls, obstacles, and dynamic platforms.

### 3.4 Action Space (Discrete Multi-Branch)

The agent selects actions across 4 independent discrete branches:

1. **Movement**: Stand still (0), Move forward (1)
2. **Jump**: Do not jump (0), Vertical jump (1)
3. **Slide**: Stand/run (0), Crouch-slide under obstacles (1)
4. **Turn**: Maintain heading (0), Turn left (1), Turn right (2)

Total action combinations per decision step: $2 \times 2 \times 2 \times 3 = 24$.

---

## 4. Experimental Evaluation & Baseline Comparisons

Performance was evaluated over 100 independent episodes per configuration across the 492 reachable cells of the maze (calculated via BFS flood fill in `BugsyReachableCells.cs`):

| Agent Configuration | Mean Cells / Episode | 95% Bootstrap CI | Cumulative Maze Discovered | Success Rate (≥25 cells) |
|---|:---:|:---:|:---:|:---:|
| **Random Exploration** | 4.6 | [3.9, 5.3] | 63.6% (313 cells) | 0.0% |
| **Wall Follower (Heuristic)** | 15.0 | [13.2, 16.9] | 85.6% (421 cells) | 17.0% |
| **v8 PPO (Novelty + Map)** | **35.6** | **[30.9, 40.6]** | **97.0% (477 cells)** | **63.0%** |
| **No-Map Ablation** | 40.6 | [35.8, 45.6] | 98.2% (483 cells) | 70.0% |
| **v8 Unseen Maze (Seed 7)** | 27.7 | [23.9, 31.5] | 98.0% (482 cells) | 47.0% |

### Key Findings

1. **Superior Autonomous Coverage**: The PPO exploration agent discovers **35.6 cells/episode** on average, outperforming the classical wall-follower heuristic (15.0 cells) by **+137%** and random exploration (4.6 cells) by **+673%**.
2. **High Cumulative Exploration**: Across 100 evaluation episodes, the v8 agent uncovered **97.0% (477 / 492)** of all reachable maze cells.
3. **Generalization to Unseen Layouts**: Tested on an unseen maze generated with a different random seed (`seed = 7`), the agent maintained **27.7 cells/episode**, confirming robust obstacle avoidance and navigation rather than path memorization.

---

## 5. Running and Reproducing

### 5.1 Evaluating the Trained Agent in Unity

1. Open the project in **Unity 6**.
2. Open the scene `Assets/Scenes/MazeBugged.unity` (or `MazeBugFree.unity`).
3. Select the `Player` GameObject in the Hierarchy.
4. In `Behavior Parameters`:
   - Set **Model** to `Assets/Models/BugsyAgent_v8.onnx` (or `navigation/models/BugsyAgent_v8.onnx`).
   - Set **Behavior Type** to `Inference Only`.
5. In `CoverageTracker`:
   - Set `Csv Label` to e.g. `v8_eval`.
   - Set `Target Episodes` to `100` and `Speed Up` to `10`.
6. Press **Play**. Unity will execute 100 evaluation episodes at $10\times$ speed and output the CSV to the project root.

### 5.2 Generating Figures and Statistical Tables

```bash
cd navigation
pip install pandas matplotlib numpy
python make_figures.py
```

This generates:
- `summary_table.csv`: Complete metrics (mean, median, standard deviation, bootstrap CIs, regret, success rates).
- `figures/fig1_cumulative_unique.png`: Cumulative unique cells discovered over 100 episodes.
- `figures/fig2_coverage_improved.png`: Mean coverage comparison with 95% bootstrap error bars.
- `figures/fig3_success_rate.png`: Success rate meeting coverage thresholds with Wilson score intervals.
- `figures/fig4_generalization_gap.png`: Performance comparison between training and unseen maze layouts.
- `figures/fig5_coverage_distribution.png`: Box plots showing distribution and quartiles.
- `figures/fig11_cumulative_regret_difference.png`: Cumulative regret avoided compared to random exploration.

### 5.3 Training from Scratch

```bash
# Ensure ml-agents Python package is installed
pip install mlagents

# Start training run
mlagents-learn navigation/bugsy_ppo.yaml --run-id=bugsy_training --force
```

Press **Play** in Unity when prompted. Monitor training via TensorBoard:

```bash
tensorboard --logdir results/
```

---

## 6. Shared Unity Script Inventory

The navigation and exploration logic interfaces with the shared Unity scripts located in [`Assets/Scripts/`](../Assets/Scripts/):

| Script | Purpose |
|---|---|
| [`BugsyAgent.cs`](../Assets/Scripts/BugsyAgent.cs) | Core agent implementation: observation vector, discrete action processing, stuck detection, reset |
| [`CoverageTracker.cs`](../Assets/Scripts/CoverageTracker.cs) | Spatial grid division, visited-cell state, intrinsic reward computation, CSV logging |
| [`PlayerController.cs`](../Assets/Scripts/PlayerController.cs) | Character physics controller (velocity translation, grounded raycast, jump/slide actuation) |
| [`WallFollowerBrain.cs`](../Assets/Scripts/WallFollowerBrain.cs) | Scripted right-hand-rule heuristic baseline with raycast obstacle detection and heading snapping |
| [`GameManager.cs`](../Assets/Scripts/GameManager.cs) | Life system, game states (Playing, GameOver, Won), and in-place training reset |
| [`BugsyMazeBuilder.cs`](../Assets/Editor/BugsyMazeBuilder.cs) | Procedural maze generator using recursive backtracker DFS + BFS shortest-path calculation |
| [`BugsyReachableCells.cs`](../Assets/Editor/BugsyReachableCells.cs) | Editor flood-fill tool that computes the 492 reachable ground-truth cell count |
| [`BugsyTrainingMode.cs`](../Assets/Editor/BugsyTrainingMode.cs) | One-click toggle between training (10 Hz decisions, headless-friendly) and inference mode |

---

*For full line-by-line code breakdowns and mathematical derivations, refer to [`BUGSY_Parkour_RL_Explanation.md`](BUGSY_Parkour_RL_Explanation.md).*
