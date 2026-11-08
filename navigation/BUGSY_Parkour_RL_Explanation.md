# BUGSY Parkour — Reinforcement Learning Project: Full Code Explanation

> [!NOTE]
> This document covers every **RL-important** C# script in the project. Unity boilerplate files (`ReadmeEditor.cs`, `Readme.cs`, `MiniMap.cs`) are included briefly since they are not central to RL. All file links are clickable.

---

## Table of Contents
1. [Project Architecture Overview](#1-project-architecture-overview)
2. [BugsyAgent.cs — The RL Agent](#2-bugsyagentcs--the-rl-agent)
3. [CoverageTracker.cs — Intrinsic Reward & Exploration](#3-coveragetrackercs--intrinsic-reward--exploration)
4. [WallFollowerBrain.cs — Scripted Baseline](#4-wallfollowerbraincs--scripted-baseline)
5. [PlayerController.cs — Physics & Action Execution](#5-playercontrollercs--physics--action-execution)
6. [GameManager.cs — Episode Lifecycle](#6-gamemanagercs--episode-lifecycle)
7. [FinishZone.cs — Terminal Win Condition](#7-finishzonecs--terminal-win-condition)
8. [BoulderTrap.cs — Hazard / Penalty Source](#8-bouldertrapcss--hazard--penalty-source)
9. [LaserGrid.cs — Timed Hazard](#9-lasergridcs--timed-hazard)
10. [MovingPlatform.cs — Dynamic Obstacle](#10-movingplatformcs--dynamic-obstacle)
11. [Pendulum.cs — Dynamic Obstacle](#11-pendulumcs--dynamic-obstacle)
12. [Rotator.cs — Dynamic Obstacle](#12-rotatorcs--dynamic-obstacle)
13. [CameraFollow.cs — Viewer Utility](#13-camerafollowcs--viewer-utility)
14. [bugsy\_ppo.yaml — PPO Hyperparameters](#14-bugsy_ppoyaml--ppo-hyperparameters)
15. [make\_figures.py — Evaluation & Analysis](#15-make_figurespy--evaluation--analysis)
16. [BugsyMazeBuilder.cs — Procedural Maze Generation](#16-bugsymazebuildercss--procedural-maze-generation)
17. [BugsyAgentSetup.cs — Editor Automation](#17-bugsyagentsetupcss--editor-automation)
18. [BugsyTrainingMode.cs — Training Mode Switcher](#18-bugsytrainingmodecs--training-mode-switcher)
19. [BugsyReachableCells.cs — Maze Analysis Tool](#19-bugsyreachablecellscss--maze-analysis-tool)
20. [Key RL Concepts & Equations](#20-key-rl-concepts--equations)

---

## 1. Project Architecture Overview

```
BUGSY Parkour
│
├── BugsyAgent          ← inherits Unity ML-Agents Agent; drives the whole RL loop
│   ├── CollectObservations()  → what the agent SEES  (43 numbers)
│   ├── OnActionReceived()     → what the agent DOES  (4 discrete axes)
│   └── FixedUpdate()          → episode termination logic
│
├── CoverageTracker     ← intrinsic novelty reward (count-based exploration)
│
├── WallFollowerBrain   ← hand-coded heuristic baseline (right-hand-rule maze solver)
│
├── PlayerController    ← physics character controller (reads agent OR keyboard)
│
├── GameManager         ← lives, win/lose state, restart, GUI
│
└── Environment objects
    ├── FinishZone      ← triggers Win()
    ├── BoulderTrap     ← proximity-drop hazard → LoseLife()
    ├── LaserGrid       ← timed on/off hazard   → LoseLife()
    ├── MovingPlatform  ← kinematic obstacle
    ├── Pendulum        ← swinging obstacle
    └── Rotator         ← spinning obstacle
```

The agent uses **Proximal Policy Optimisation (PPO)** via ML-Agents and is rewarded exclusively with **count-based intrinsic motivation** (a bonus for each new grid cell visited), not for task completion. This makes it an **exploration-first** design.

---

## 2. [`BugsyAgent.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/BugsyAgent.cs)

### Concept: ML-Agents Agent Class
`BugsyAgent` inherits from `Agent` (Unity ML-Agents SDK). The SDK calls four lifecycle methods automatically:
`Initialize` → `OnEpisodeBegin` → (`CollectObservations` + `OnActionReceived`) × N steps → `EndEpisode`.

---

### Line-by-line

```csharp
// Line 1-5: Imports
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
```
Pulls in the core Unity engine types and the three ML-Agents namespaces needed: core Agent, action buffers, and sensor interfaces.

```csharp
// Line 6
public class BugsyAgent : Agent
```
Inherits from `Agent` — this is what ML-Agents looks for to drive training.

```csharp
// Line 8
const float MazeSize = 66f;
```
The maze is 11 cells × 6 m each = 66 m per side. Used to **normalise position** into [0, 1] before feeding it into the neural network. Normalisation helps gradient-based optimisers converge faster.

```csharp
// Line 9
const int StuckLimitSteps = 800;   // 8 s of game time
```
`Time.fixedDeltaTime = 0.01 s`, so 800 physics steps = 8 s. If the agent explores no new cell in 8 seconds, the episode ends. This prevents indefinite episodes that contribute no useful gradient signal.

```csharp
// Lines 11-21: Field declarations
PlayerController pc;   // character physics
Rigidbody rb;          // Unity physics body
CameraFollow camFollow;
CoverageTracker coverage;
int prevMove, prevJump, prevSlide, prevTurn;  // last-step actions
int stuckSteps;   // counter for stuck detection
int lastCells;    // last snapshot of cells visited
int spawnCount, spawnFallbacks;  // diagnostic counters
```

---

### `Initialize()` — Lines 23-30
```csharp
public override void Initialize()
{
    pc = GetComponent<PlayerController>();
    rb = GetComponent<Rigidbody>();
    coverage = GetComponent<CoverageTracker>();
    camFollow = FindFirstObjectByType<CameraFollow>();
    pc.agentControlled = true;   // tells PlayerController to ignore keyboard
}
```
Called once when the Agent GameObject wakes up. Caches components and switches the player into agent-controlled mode. **RL relevance:** separating "agent mode" from "human mode" lets the same physics controller serve both training and manual play.

---

### `OnEpisodeBegin()` — Lines 32-51
```csharp
public override void OnEpisodeBegin()
{
    if (coverage != null) coverage.NewEpisode();  // clear the visited grid
    prevMove = prevJump = prevSlide = prevTurn = 0;
    stuckSteps = 0; lastCells = 0;
    if (GameManager.Instance != null) GameManager.Instance.Restart();
    RandomSpawn();
}
```
Called at the start of every episode. Resets the world and **randomises the spawn position**. Random starts are a curriculum-learning technique that:
- Increases state-space coverage during training
- Prevents the agent memorising a single path

---

### `RandomSpawn()` — Lines 53-112
```csharp
float x = Random.Range(0, 11) * 6f + Random.Range(-1.5f, 1.5f);
float z = Random.Range(0, 11) * 6f + Random.Range(-1.5f, 1.5f);
Vector3 cand = new Vector3(x, 1.2f, z);
// Physics check: reject position if any collider (not self) overlaps
Collider[] hits = Physics.OverlapCapsule(cand + Vector3.down*0.3f, cand + Vector3.up*0.6f, 0.5f, ...);
```
- Samples a maze-grid-aligned candidate `(x, z)` with small jitter (±1.5 m inside the 6 m cell).
- Uses `Physics.OverlapCapsule` to reject positions that intersect geometry.
- Up to **60 tries** before falling back to a bare grid-centre (fallback counted for diagnostics).
- Sets a **random yaw** (0–360°) so the agent starts facing a random direction — prevents direction bias.

---

### `CollectObservations()` — Lines 114-148

**Concept: Observation Space**
The agent's neural network input is a **43-dimensional vector**:

| # | Observation | Range | Code |
|---|---|---|---|
| 2 | Normalised X, Z position | [0, 1] | `pos.x / MazeSize` |
| 3 | Local-space velocity (x, y, z) | ≈[−1, 1] | `InverseTransformDirection(rb.velocity) / 10` |
| 2 | Heading as (sin θ, cos θ) | [−1, 1] | `Sin(yaw)`, `Cos(yaw)` |
| 1 | Is grounded flag | {0, 1} | `pc.IsGroundedNow()` |
| 1 | Is sliding flag | {0, 1} | `pc.IsSlidingNow()` |
| 2 | One-hot: previous move (2 classes) | — | `AddOneHotObservation(prevMove, 2)` |
| 2 | One-hot: previous jump (2 classes) | — | — |
| 2 | One-hot: previous slide (2 classes) | — | — |
| 3 | One-hot: previous turn (3 classes) | — | — |
| 25 | Visited-cell 5×5 local window | {0, 1} | `coverage.AddVisitedObservations` |

**Why sin/cos for heading?** A raw angle has a discontinuity at 0°/360°. Encoding as $(\sin\theta, \cos\theta)$ gives a continuous 2D representation that the network can differentiate smoothly.

$$\text{heading\_obs} = (\sin\theta,\ \cos\theta), \quad \theta = \text{yaw in radians}$$

**Why one-hot for previous actions?** Discrete action values (0, 1, 2…) imply false ordinal relationships. One-hot encoding treats each action as independent, which is correct for categorical choices.

**Why the 5×5 visited window?** The agent needs to know which nearby cells it *hasn't* visited yet to bias exploration toward them. This is a form of **partial observability map** embedded in the observation vector.

---

### `OnActionReceived()` — Lines 150-169
```csharp
int move  = actions.DiscreteActions[0];  // 0 = stand, 1 = forward
int jump  = actions.DiscreteActions[1];  // 0 = no,    1 = jump
int slide = actions.DiscreteActions[2];  // 0 = no,    1 = slide
int turn  = actions.DiscreteActions[3];  // 0 = none,  1 = left, 2 = right

pc.agentMoveF = move == 1 ? 1f : 0f;
pc.agentTurn  = turn == 1 ? -1f : (turn == 2 ? 1f : 0f);
// store for next observation
prevMove = move; prevJump = jump; ...
// No reward here — all reward is from CoverageTracker
```
The action space is **4 discrete axes** (total combinations: 2×2×2×3 = 24). The agent never strafes (agentMoveS is always 0); it turns and walks forward.

> [!IMPORTANT]
> The comment says "No task reward on purpose." This is a deliberate **intrinsic motivation only** design: the policy gradient is driven entirely by exploration bonuses, not by reaching the goal.

---

### `Heuristic()` — Lines 170-200
```csharp
// Random baseline
if (coverage.randomActions) {
    d[0] = Random.Range(0, 2); ...  return;
}
// Wall-follower baseline
WallFollowerBrain wf = GetComponent<WallFollowerBrain>();
if (wf != null && wf.enabled) { wf.Decide(d); return; }
// Human keyboard fallback
d[0] = Input.GetKey(KeyCode.W) ? 1 : 0; ...
```
`Heuristic()` is called instead of the neural network when **Behavior Type = Heuristic Only**. Three modes: random, scripted wall-follower, or human keyboard. This allows comparing RL against non-RL baselines without changing the rest of the code.

---

### `FixedUpdate()` — Lines 211-251
```csharp
// Win → optional +1 reward, then end episode
if (gm.state == GameManager.State.Won) {
    if (coverage.taskRewardMode) AddReward(1f);
    EndEpisode(); return;
}
// Stuck detection: no new cell in 800 steps → end episode
if (coverage.EpisodeCells != lastCells) { lastCells = ...; stuckSteps = 0; }
else { stuckSteps++; if (stuckSteps >= StuckLimitSteps) EndEpisode(); }
```
Episode termination criteria:
1. **Win** — agent reached `FinishZone`
2. **Game Over** — agent lost all lives (handled by `GameManager`)
3. **Stuck** — no new cell for 8 s of simulated time

---

## 3. [`CoverageTracker.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/CoverageTracker.cs)

### Concept: Count-Based Intrinsic Motivation

Classic RL needs an extrinsic reward signal. In **sparse-reward environments** (like a maze where the only goal is reaching a distant exit), the agent rarely stumbles onto the reward by accident, so learning stalls. **Intrinsic motivation** adds a reward for exploring novel states, encouraging systematic coverage.

**Count-based bonus** (the simplest form):

$$r_{\text{int}}(s) = \begin{cases} +r_{\text{new}} & \text{if cell } (g_x, g_z) \text{ not previously visited} \\ 0 & \text{otherwise} \end{cases}$$

Here $r_{\text{new}} = 0.1$ (configurable via `newCellReward`).

---

### Line-by-line

```csharp
// Lines 8-22: Fields
public float gridSize = 3f;    // 3 m per grid cell
public int gridCount = 30;     // 30×30 = 900 cells max
public float origin = 12f;     // world-space offset so coords stay non-negative
public float newCellReward = 0.1f;

public bool taskRewardMode = false;  // if true: no novelty, only +1 at finish (standard PPO)
public bool hideVisitedMap = false;  // ablation: novelty reward but observation zeroed
public bool randomActions = false;   // random agent baseline
```

```csharp
// Line 36-46: Awake – grid allocation
visited = new bool[gridCount, gridCount];    // this-episode cells
everVisited = new bool[gridCount, gridCount]; // lifetime unique cells
// CSV logging setup (optional, used for offline baseline measurement)
csvPath = Path.Combine(root, "baseline_" + csvLabel + ".csv");
```

```csharp
// Lines 59-67: Cell coordinate helpers
int CellX() { return Mathf.FloorToInt((transform.position.x + origin) / gridSize); }
int CellZ() { return Mathf.FloorToInt((transform.position.z + origin) / gridSize); }
```

The cell index formula:

$$g_x = \left\lfloor \frac{x + \text{origin}}{\text{gridSize}} \right\rfloor$$

`origin = 12` ensures that even agents with small negative x/z coordinates map to valid (≥ 0) indices.

```csharp
// Lines 69-91: FixedUpdate – reward computation
void FixedUpdate()
{
    int gx = CellX(); int gz = CellZ();
    if (gx < 0 || ... || gx >= gridCount || ...) return;  // bounds check
    if (!visited[gx, gz])
    {
        visited[gx, gz] = true;
        visitedCount++;
        if (!everVisited[gx, gz]) { everVisited[gx, gz] = true; everCount++; }
        if (agent != null && !taskRewardMode)
            agent.AddReward(newCellReward);   // ← THE REWARD
    }
}
```

Key design decisions:
- `visited` resets every episode; `everVisited` accumulates across all episodes (lifetime novelty, only logged to CSV, not fed back as reward).
- Reward is only added **once per cell per episode**, preventing the agent from pacing back and forth for free reward.

```csharp
// Lines 94-117: AddVisitedObservations – the 5×5 window
for (int dz = 2; dz >= -2; dz--)
    for (int dx = -2; dx <= 2; dx++)
    {
        int x = cx + dx;  int z = cz + dz;
        bool inside = x >= 0 && z >= 0 && x < gridCount && z < gridCount;
        sensor.AddObservation((inside && !visited[x, z]) ? 0f : 1f);
    }
```
Adds 25 observations (5×5 grid centred on agent). Value 0 = unvisited cell in-bounds (a target), value 1 = visited or outside (no reward available). This lets the network develop a *local novelty map* that biases it toward unvisited corridors.

```csharp
// Lines 119-146: NewEpisode – logging & reset
Academy.Instance.StatsRecorder.Add("Bugsy/CellsVisited", visitedCount);
Academy.Instance.StatsRecorder.Add("Bugsy/Finished", wonThisEpisode ? 1f : 0f);
// write CSV row
visited = new bool[gridCount, gridCount];
visitedCount = 0;
wonThisEpisode = false;
```
`StatsRecorder.Add` sends metrics to TensorBoard. The CSV is used for offline baseline analysis (comparing PPO vs random vs wall-follower). `visitedCount` resets; `everCount` does not.

---

## 4. [`WallFollowerBrain.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/WallFollowerBrain.cs)

### Concept: Right-Hand Rule (Scripted Baseline)

The **right-hand rule** is a classic maze-solving heuristic: always keep your right hand touching the wall. It guarantees solving any simply-connected maze. This provides a deterministic upper-bound baseline for comparison with the RL agent's coverage.

---

### Line-by-line

```csharp
// Lines 7-10: Tunable parameters
public float openDistance = 7f;       // right ray > this → opening detected
public float frontStopDistance = 3f;  // front ray < this → wall ahead, turn left
public int advanceDecisions = 3;      // extra steps forward before turning right
public int cooldownDecisions = 10;    // lock-out after a turn to prevent oscillation
public bool hop = true;               // always jump (clear low hurdles)
```

```csharp
// Lines 13-16: State machine variables
int mode;          // 0 = drive straight, 1 = advance to opening centre, 2 = rotate to target yaw
int advanceLeft;   // countdown for mode 1
int cooldown;      // countdown after a turn
float targetYaw;   // desired heading after a right turn
```

```csharp
// Lines 20-33: Dist() helper
float Dist(Vector3 dir)
{
    RaycastHit[] hits = Physics.RaycastAll(transform.position, dir, 20f, ~0,
        QueryTriggerInteraction.Ignore);
    float best = 20f;
    foreach (var h in hits)
        if (h.collider.CompareTag("Wall") && h.distance < best)
            best = h.distance;
    return best;
}
```
Fires a ray of up to 20 m in a given direction and returns the distance to the nearest **wall-tagged** collider. Trigger colliders are ignored to avoid counting sensors and zones as walls.

```csharp
// Lines 36-43: HeadingLock()
int HeadingLock(float yaw)
{
    float snapped = Mathf.Round(yaw / 90f) * 90f;  // nearest cardinal direction
    float delta   = Mathf.DeltaAngle(yaw, snapped);
    if (delta > 6.5f)  return 2;   // need to turn right to snap
    if (delta < -6.5f) return 1;   // need to turn left
    return 0;
}
```
Keeps the agent heading snapped to cardinal directions (N/E/S/W). The ±6.5° dead-zone prevents micro-corrections. This is necessary because the physics controller can drift slightly during forward movement.

```csharp
// Lines 45-118: Decide() — the main state machine
public void Decide(ActionSegment<int> d)
{
    d[0] = 0; d[1] = hop ? 1 : 0; d[2] = 0; d[3] = 0;
    float yaw = transform.eulerAngles.y;
    Vector3 f = transform.forward; f.y = 0; f.Normalize();
    Vector3 r = new Vector3(f.z, 0f, -f.x);   // 90° right perpendicular

    if (cooldown > 0) cooldown--;

    // MODE 2: rotating in-place to targetYaw
    if (mode == 2) {
        float delta = Mathf.DeltaAngle(yaw, targetYaw);
        if (Mathf.Abs(delta) > 6.5f) { d[3] = delta > 0 ? 2 : 1; return; }
        mode = 0; cooldown = cooldownDecisions;
    }

    // MODE 1: advance a few steps before turning
    if (mode == 1) {
        if (advanceLeft > 0) { advanceLeft--; d[0]=1; d[3]=HeadingLock(yaw); return; }
        targetYaw = Mathf.Round(yaw/90f)*90f + 90f;  // snap right
        mode = 2; d[3] = 2; return;
    }

    // MODE 0: drive forward
    float front = Dist(f);  float right = Dist(r);
    if (cooldown == 0 && right > openDistance) { mode=1; advanceLeft=advanceDecisions; d[0]=1; return; }
    if (front < frontStopDistance)             { targetYaw=...-90f; mode=2; d[3]=1; return; }
    d[0] = 1; d[3] = HeadingLock(yaw);
}
```

**Right perpendicular formula:**
If the forward vector is $(f_x, 0, f_z)$, then the right vector is $(f_z, 0, -f_x)$.
This is a 90° clockwise rotation in the XZ plane:

$$\vec{r} = \begin{pmatrix} f_z \\ 0 \\ -f_x \end{pmatrix}$$

**State machine logic:**
```
mode 0 (drive):
  right open?  → mode 1 (advance before turning)
  wall ahead?  → mode 2 (turn left)
  else         → move forward + heading correction

mode 1 (advance):
  countdown > 0 → keep moving
  done          → set targetYaw = current + 90°, mode 2

mode 2 (rotate):
  |Δyaw| > 6.5° → keep rotating
  aligned       → mode 0, set cooldown
```

---

## 5. [`PlayerController.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/PlayerController.cs)

### Concept: Physics-Based Character Controller

This script acts as the **actuator layer** — it translates the agent's discrete action choices into real physics forces. Keeping physics separate from the agent class is good software design: the same controller works for both human play and RL training.

---

### Line-by-line

```csharp
// Lines 5-8: Speed constants
public float moveSpeed  = 8f;   // m/s during normal movement
public float slideSpeed = 5f;   // m/s while crouching/sliding
public float jumpSpeed  = 7f;   // m/s vertical impulse
public float turnSpeed  = 120f; // degrees/second
```

```csharp
// Lines 11-16: Agent command inputs (written by BugsyAgent)
public float agentMoveF;   // forward axis: [-1, 0, 1]
public float agentMoveS;   // strafe axis (always 0 in current agent design)
public float agentTurn;    // turn rate: [-1, 0, 1]
public bool  agentJump;
public bool  agentSlide;
```

```csharp
// Lines 24-33: Start()
rb = GetComponent<Rigidbody>();
rb.maxDepenetrationVelocity = 25f;   // fast collision recovery
rb.solverIterations = 12;            // more accurate constraint solving
rb.solverVelocityIterations = 4;
```
Higher solver iterations prevent the agent from "sinking" into moving platforms or boulders.

```csharp
// Lines 35-69: Update() — input reading + slide/jump
if (agentControlled) {
    inputF = agentMoveF; ...
} else {
    inputF = Input.GetAxisRaw("Vertical"); ...  // WASD for humans
}
// Slide toggling
if (wantSlide && isGrounded && !isSliding) SetSlide(true);
else if (!wantSlide && isSliding && HasHeadroom()) SetSlide(false);
// Jump: immediate velocity override
if (wantJump && isGrounded && !isSliding)
    rb.linearVelocity = new Vector3(rb.linearVelocity.x, jumpSpeed, rb.linearVelocity.z);
```
Jump is implemented as a direct Y-velocity set, not an impulse force. This gives predictable, deterministic jump height regardless of frame-rate.

**Jump height** can be derived from kinematics:

$$h_{\max} = \frac{v_0^2}{2g} = \frac{7^2}{2 \times 9.81} \approx 2.5 \text{ m}$$

```csharp
// Lines 71-110: FixedUpdate() — movement
float half = isSliding ? 0.5f : 1f;
isGrounded = Physics.Raycast(transform.position, Vector3.down, half + 0.1f);
```
Grounded check: a short downward ray (`0.5 + 0.1` when sliding, `1.0 + 0.1` standing). Simple and deterministic — avoids false positives from coyote-time or spherecasts.

```csharp
// When agent-controlled: turn, then compute facing-relative movement
transform.Rotate(0f, agentTurn * turnSpeed * Time.fixedDeltaTime, 0f);
Vector3 f = transform.forward; f.y = 0; f.Normalize();
Vector3 right = new Vector3(fwd.z, 0f, -fwd.x);
// Movement velocity (preserves Y for gravity)
Vector3 move = (right * inputS + fwd * inputF).normalized * speed;
rb.linearVelocity = new Vector3(move.x, rb.linearVelocity.y, move.z);
```

**Movement equation:**

$$\vec{v}_{xz} = \hat{n}(\vec{f} \cdot I_F + \vec{r} \cdot I_S) \times s_{\text{speed}}$$

where $\vec{f}$ = forward unit vector, $\vec{r}$ = right unit vector, $I_F, I_S$ = forward/strafe inputs ∈ {−1, 0, 1}, and $s$ = speed. The Y component is preserved from the Rigidbody (gravity, jump momentum).

```csharp
// Lines 112-124: SetSlide / HasHeadroom
void SetSlide(bool slide) {
    isSliding = slide;
    s.y = slide ? 0.5f : 1f;        // squash the scale
    transform.position += Vector3.up * (slide ? -0.5f : 0.5f);  // re-centre
}
bool HasHeadroom() { return !Physics.Raycast(transform.position, Vector3.up, 1.6f); }
```
Sliding shrinks the character to half height. `HasHeadroom` prevents un-sliding into a ceiling (would clip geometry).

---

## 6. [`GameManager.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/GameManager.cs)

### Concept: Episode Lifecycle & Game State Machine

`GameManager` is a **Singleton** (accessed via `GameManager.Instance`) that coordinates:
- Life management
- State transitions (Playing → Won / GameOver)
- World reset on restart

---

### Line-by-line (key parts)

```csharp
// Line 5: Singleton pattern
public static GameManager Instance;
```
Static reference allows any script to call `GameManager.Instance.LoseLife(...)` without needing a direct reference.

```csharp
// Lines 7-13: Config
public int maxLives = 3;
public float invulnTime = 1.5f;  // seconds of invulnerability after being hit
public bool trainingMode = false; // disables freeze + confetti during RL training
```

`trainingMode = true` is **critical during training**: without it, `EndGame` would call `Time.timeScale = 0`, freezing the simulation and blocking ML-Agents' episode loop.

```csharp
// Lines 31-46: Awake
Instance = this;
if (!trainingMode) Time.timeScale = 1f;
Time.fixedDeltaTime = 0.01f;   // 100 Hz physics; matches StuckLimitSteps arithmetic
lives = maxLives;
```

```csharp
// Lines 79-101: LoseLife(reason)
if (Time.time < invulnUntil) return;   // grace period check
lives--;
invulnUntil = Time.time + invulnTime;
if (lives <= 0) EndGame(State.GameOver);
```
The invulnerability window prevents one hit from chain-triggering multiple `LoseLife` calls in the same physics frame.

```csharp
// Lines 117-147: EndGame(s)
state = s;
if (trainingMode) return;   // in training: agent's FixedUpdate calls EndEpisode instead
// otherwise: disable all MonoBehaviours, freeze rigidbody, Time.timeScale = 0
```

```csharp
// Lines 150-210: Restart()
state = State.Playing;
lives = maxLives;
// Reset rigidbody: zero velocity, position to spawnPos
rb.linearVelocity = rb.angularVelocity = Vector3.zero;
rb.position = spawnPos; rb.rotation = spawnRot;
// Reset all BoulderTraps
BoulderTrap[] traps = FindObjectsByType<BoulderTrap>(...);
for each trap: trap.ResetNow();
```
`Restart()` is called at the start of every training episode (via `OnEpisodeBegin → GameManager.Instance.Restart()`). It is a **full in-place reset** — the scene is not reloaded, which is much faster.

---

## 7. [`FinishZone.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/FinishZone.cs)

### Concept: Terminal State (Goal)

```csharp
void OnTriggerEnter(Collider other)
{
    if (!other.CompareTag("Player")) return;
    Debug.Log("FINISH reached!");
    if (GameManager.Instance != null)
        GameManager.Instance.Win();
}
```
A Unity **trigger collider** that fires `Win()` when the player enters. This transitions the game state to `State.Won`, which `BugsyAgent.FixedUpdate` detects and calls `EndEpisode()`.

In `taskRewardMode`, this is also where `AddReward(1f)` is issued — the only extrinsic reward signal in that ablation.

---

## 8. [`BoulderTrap.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/BoulderTrap.cs)

### Concept: Proximity-Triggered Hazard

Boulders are a **sparse penalty** source. In training mode (`taskRewardMode = false`), lives are tracked but losing them only ends the episode via `GameManager.state == GameOver`, not via a negative reward signal. In a more standard design you might add `AddReward(-1f)` on hit.

---

### Line-by-line

```csharp
// Lines 5-8: Config
public float triggerRadius = 2.2f;  // horizontal proximity to trigger drop
public float resetDelay = 6f;       // seconds before boulder resets
public float dropSpeed = 25f;       // m/s initial downward velocity
public float hideRadius = 2.5f;     // visibility occlusion distance
```

```csharp
// Lines 17-24: Start – kinematic by default
rb.useGravity = false;
rb.isKinematic = true;   // boulder hovers stationary until triggered
home = transform.position;
```

```csharp
// Lines 65-70: DistToSegment – visibility logic
static float DistToSegment(Vector3 point, Vector3 a, Vector3 b)
{
    Vector3 ab = b - a;
    float t = Clamp01(Dot(point - a, ab) / Max(ab.sqrMagnitude, ...));
    return Distance(point, a + ab * t);   // closest point on segment
}
```

Formula: **point-to-segment distance**. The boulder hides itself if it lies too close to the line segment from camera to player (so it doesn't obstruct the view).

$$d = \|\vec{p} - (\vec{a} + t(\vec{b} - \vec{a}))\|, \quad t = \text{clamp}\!\left(\frac{(\vec{p}-\vec{a})\cdot(\vec{b}-\vec{a})}{|\vec{b}-\vec{a}|^2},\ 0,\ 1\right)$$

```csharp
// Lines 72-97: FixedUpdate – trigger and reset
Vector3 d = p.position - home; d.y = 0f;
if (d.magnitude < triggerRadius)
{
    rb.isKinematic = false; rb.useGravity = true;
    rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
    rb.linearVelocity = Vector3.down * dropSpeed;  // instant downward snap
    dropped = true; dropTime = Time.time;
}
// After resetDelay seconds:
else if (Time.time - dropTime > resetDelay) ResetNow();
```
Trigger checks only horizontal distance (y component zeroed) so the boulder drops regardless of elevation difference.

```csharp
// Lines 120-133: OnCollisionEnter
if (c.collider.CompareTag("Player") && c.relativeVelocity.magnitude > 1.5f)
    GameManager.Instance.LoseLife("boulder");
```
Velocity threshold prevents slow-rolling contact from counting as a hit.

---

## 9. [`LaserGrid.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/LaserGrid.cs)

### Concept: Timed Periodic Hazard

The laser grid cycles between **off → warning → on** using a simple modulo clock. The agent must learn the timing pattern (implicitly, through many episodes) or avoid lasers entirely.

---

### Line-by-line

```csharp
// Lines 5-8: Timing config
public float onTime  = 2f;    // seconds the laser is lethal
public float offTime = 2f;    // seconds the laser is safe
public float warnTime = 0.6f; // seconds of visual warning before going on
public float phase   = 0f;    // offset so different lasers are out of sync
```

```csharp
// Lines 21-37: Update() – state machine via modulo clock
float cycle = onTime + offTime;       // total period (4 s default)
float t = (Time.time + phase) % cycle; // phase-shifted position in cycle

int want;
if (t < onTime)              want = 2;  // lethal (ON)
else if (t > cycle - warnTime) want = 1; // warning (visible, harmless)
else                           want = 0; // off
```

The time position in the cycle:

$$t_{\text{cycle}} = (t_{\text{world}} + \phi) \bmod (t_{\text{on}} + t_{\text{off}})$$

States: ON if $t_{\text{cycle}} < t_{\text{on}}$, WARN if $t_{\text{cycle}} > (t_{\text{on}}+t_{\text{off}}) - t_{\text{warn}}$, else OFF.

```csharp
// Lines 44-48: Collision detection while ON
if (state == 2 && GameManager.Instance != null && PlayerInside())
    GameManager.Instance.LoseLife("laser");
```

```csharp
// Lines 50-61: PlayerInside() – box overlap test
Collider[] hits = Physics.OverlapBox(transform.position, damageHalfExtents, Quaternion.identity);
foreach (h in hits) if (h.CompareTag("Player")) return true;
```
Uses a box-shaped volume (`damageHalfExtents = (2.8, 1.3, 0.4)` m) rather than the mesh collider for consistent game logic.

---

## 10. [`MovingPlatform.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/MovingPlatform.cs)

### Concept: Kinematic Dynamic Obstacle

```csharp
public Vector3 direction = Vector3.up;   // movement axis
public float distance = 2f;              // amplitude (metres)
public float speed = 1.5f;              // oscillation speed (m/s)
```

```csharp
void FixedUpdate()
{
    float d = Mathf.PingPong(Time.time * speed, distance);
    rb.MovePosition(startPos + direction.normalized * d);
}
```

`Mathf.PingPong(t, L)` maps $t$ to a triangle wave in $[0, L]$:

$$d(t) = L - |((t \bmod 2L) - L)|$$

This creates smooth back-and-forth motion without discontinuities. `rb.MovePosition` keeps the platform in the physics simulation so it properly pushes the player via collision response.

---

## 11. [`Pendulum.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7\RL\Project\BUGSY_Parkour\Assets\Scripts\Pendulum.cs)

### Concept: Sinusoidal Rotational Hazard

```csharp
public Vector3 axis  = Vector3.forward;  // rotation axis
public float angle   = 50f;             // half-swing amplitude (degrees)
public float speed   = 1.5f;            // angular frequency (rad/s)
public float phase   = 0f;              // offset for desync

void FixedUpdate()
{
    float a = angle * Mathf.Sin(Time.time * speed + phase);
    rb.MoveRotation(startRot * Quaternion.AngleAxis(a, axis));
}
```

The instantaneous angle follows a **sinusoidal pendulum**:

$$\theta(t) = A \sin(\omega t + \phi)$$

where $A = 50°$, $\omega = 1.5$ rad/s, $\phi$ = phase offset. Unlike a real physical pendulum (which has non-linear dynamics), this uses a simple sine for predictable, controllable motion.

---

## 12. [`Rotator.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/Rotator.cs)

### Concept: Constant Angular Velocity Obstacle

```csharp
public Vector3 axis  = Vector3.up;  // rotation axis
public float speed   = 60f;         // degrees/second

void FixedUpdate()
{
    Quaternion delta = Quaternion.AngleAxis(speed * Time.fixedDeltaTime, axis);
    rb.MoveRotation(rb.rotation * delta);
}
```

Each physics step rotates by:

$$\Delta\theta = \omega \cdot \Delta t_{\text{fixed}}$$

where $\omega = 60°/s$ and $\Delta t_{\text{fixed}} = 0.01$ s → 0.6° per step. Using `rb.MoveRotation` (not `transform.Rotate`) keeps it inside the physics solver so it interacts correctly with other Rigidbodies.

---

## 13. [`CameraFollow.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/CameraFollow.cs)

### Concept: Smart Third-Person Camera (Viewer Utility)

Not directly part of RL, but important for monitoring training. The camera:
1. Orbits behind the agent (yaw locked to agent heading via `BugsyAgent.Update`)
2. Automatically rises above walls when line-of-sight is blocked (blend variable)
3. Smoothly descends when the view clears (exponential smoothing)

```csharp
// Smooth blend-down (exponential decay)
blend = Mathf.Lerp(blend, best, 1f - Mathf.Exp(-6f * Time.unscaledDeltaTime));
```

Exponential smoothing:

$$b_{t+1} = b_t + (1 - e^{-6\Delta t})(b^* - b_t)$$

where $b^*$ is the target blend. The camera snaps **up** instantly (when a wall appears) but slides **down** smoothly (when the view clears).

---

---

## 14. [`bugsy_ppo.yaml`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/bugsy_ppo.yaml)

### Concept: ML-Agents Training Configuration

This YAML file is passed to the `mlagents-learn` CLI to configure the PPO trainer. Every field directly maps to a hyperparameter in the PPO algorithm.

---

### Line-by-line

```yaml
behaviors:
  BugsyAgent:          # must match BehaviorParameters.BehaviorName in Unity
    trainer_type: ppo  # Proximal Policy Optimisation
```
The `behaviors` key maps to Unity's `BehaviorParameters.BehaviorName`. The trainer type must be `ppo` (or `sac` for Soft Actor-Critic).

```yaml
    hyperparameters:
      batch_size: 256
```
**Minibatch size** — the number of experience samples used per gradient update. Larger batches give more stable gradient estimates but use more memory.

$$\nabla_\theta \mathcal{L} \approx \frac{1}{|\mathcal{B}|} \sum_{t \in \mathcal{B}} \nabla_\theta \mathcal{L}_t$$

```yaml
      buffer_size: 4096
```
**Rollout buffer** — experiences are collected until this many steps are gathered, then the policy is updated. With `batch_size = 256` and `num_epoch = 3`, each buffer produces:

$$\frac{4096}{256} \times 3 = 48 \text{ gradient updates per rollout}$$

```yaml
      learning_rate: 0.0003
```
The Adam optimiser step size $\alpha = 3 \times 10^{-4}$. This is a standard value for PPO in continuous-action benchmarks; suitable for discrete actions too.

```yaml
      beta: 0.005
```
**Entropy coefficient** $\beta$ — the entropy bonus added to the loss to encourage exploration:

$$\mathcal{L} = \mathcal{L}^{\text{CLIP}} + \beta \cdot \mathcal{H}[\pi_\theta]$$

where $\mathcal{H}[\pi_\theta] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$.

Higher $\beta$ → more random policy → wider exploration. 0.005 is relatively low, meaning the agent is expected to converge to a deterministic policy; intrinsic rewards from `CoverageTracker` handle exploration instead.

```yaml
      epsilon: 0.2
```
**PPO clip ratio** $\varepsilon$ — limits how much the policy can change per update:

$$\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon), \quad r_t = \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)}$$

$\varepsilon = 0.2$ is the standard PPO value; it means the policy ratio can't move outside $[0.8, 1.2]$ in a single update.

```yaml
      lambd: 0.95
```
**GAE-λ** (Generalised Advantage Estimation smoothing parameter):

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

$\lambda = 0.95$ gives a bias-variance tradeoff: closer to 1 = lower bias but higher variance advantage estimates.

```yaml
      num_epoch: 3
```
How many complete passes over the buffer per rollout. More epochs squeeze more learning from each batch, but risk overfitting the policy to a stale buffer.

```yaml
      learning_rate_schedule: linear
```
Decays the learning rate linearly from `learning_rate` to 0 over `max_steps`:

$$\alpha_t = \alpha_0 \left(1 - \frac{t}{N}\right)$$

This is common in PPO: large steps early (fast learning), small steps late (fine-tuning).

```yaml
    network_settings:
      normalize: true      # running mean/std normalisation of observations
      hidden_units: 128    # neurons per hidden layer
      num_layers: 2        # two hidden layers → ~128×128 MLP
```
The policy and value networks share a **2-layer MLP** with 128 hidden units each. `normalize: true` applies running-statistics normalisation to the 43-D observation vector — equivalent to input standardisation:

$$\tilde{o} = \frac{o - \mu}{\sigma + \epsilon}$$

where $\mu, \sigma$ are updated online. This is essential because the 25 visited-cell flags (binary) and position values (0–1) have very different scales.

```yaml
    reward_signals:
      extrinsic:
        strength: 1.0
        gamma: 0.995      # discount factor
```
**Discount factor** $\gamma = 0.995$ — very close to 1, meaning the agent is nearly myopic to the long-term future. This is appropriate for a 10,000-step episode budget where the goal (finish zone) is far away. The discounted return is:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

With $\gamma = 0.995$ and a 10,000-step episode, the effective horizon is approximately:

$$\frac{1}{1 - \gamma} = \frac{1}{0.005} = 200 \text{ steps}$$

```yaml
    keep_checkpoints: 5     # rolling window of saved .onnx / .pt models
    checkpoint_interval: 50000  # save every 50k environment steps
    max_steps: 300000       # stop after 300k total steps (~16 h)
    time_horizon: 256       # segment length for GAE computation
    summary_freq: 10000     # write TensorBoard stats every 10k steps
```

`time_horizon = 256` means GAE is computed over 256-step segments of the rollout. This matches `buffer_size / 16 = 256`, so each of the 16 parallel environments (if used) contributes exactly one segment per rollout.

---

## 15. [`make_figures.py`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/make_figures.py)

### Concept: Experimental Evaluation & Statistical Analysis

This Python script reads the CSV logs from `CoverageTracker` (and optional TensorBoard exports) and produces publication-ready figures comparing all agent variants. It embeds several statistical concepts.

---

### Key functions explained

```python
REACHABLE = 492  # maximum cells reachable in this maze (computed by BugsyReachableCells)
```
The ground-truth maze size. All coverage percentages are relative to this.

```python
def boot_ci(x, n=5000):
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])
```
**Bootstrap 95% confidence interval** for the mean. Samples `x` with replacement 5000 times and takes the 2.5th and 97.5th percentiles. This is non-parametric (no normality assumption), making it robust for small episode counts.

```python
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)
```
**Wilson score interval** for a proportion (success rate). Better than the naive $p \pm z\sqrt{p(1-p)/n}$ because it stays within [0,1] and handles small $n$ or extreme $p$ correctly:

$$CI = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

Used for the success-rate bars (Figure 3) where success = episode covered ≥49 or ≥25 cells.

```python
def ema(v, w=0.6):
    last = v[0]
    for x in v:
        last = last * w + (1-w) * x
    return np.array(out)
```
**Exponential Moving Average** with smoothing weight $w = 0.6$:

$$\hat{v}_t = w \cdot \hat{v}_{t-1} + (1-w) \cdot v_t$$

Applied to TensorBoard curves to filter noise while preserving trend shape. Used in all training-curve figures (Figures 6–10).

```python
def steps_to_threshold(steps, sm, thr, K=3):
    for i in range(len(sm) - K + 1):
        if np.all(sm[i:i+K] >= thr):
            return int(steps[i])
    return None
```
**Convergence criterion**: first step where the smoothed metric stays ≥ threshold for 3 consecutive logged points. This guards against single-episode spikes — the policy must stably exceed the bar.

### Figures generated

| Figure | What it shows | Key statistic |
|---|---|---|
| fig1 | Cumulative unique cells over 100 episodes | Lifetime coverage |
| fig2 | Mean cells/episode (bar) + final cumulative % (bar) | Bootstrap CI |
| fig3 | Success rate at ≥49 and ≥25 cells | Wilson CI |
| fig4 | Generalization gap: seen vs unseen maze | Bootstrap CI |
| fig5 | Box plot of cells/episode distribution | Quartiles + mean |
| fig6 | Training cells curve (main 3 variants) | EMA smoothed |
| fig7 | Development history v4→v8 | EMA smoothed |
| fig8/9 | Policy entropy during training | EMA smoothed |
| fig10 | Convergence speed (steps to 40 cells) | Steps-to-threshold |
| fig11 | Cumulative regret vs random baseline | Cumulative sum |

**Cumulative regret** (Figure 11):

$$\text{Regret}_T = \sum_{t=1}^{T} (\text{REACHABLE} - C_t)$$

where $C_t$ is the cells visited in episode $t$. The plot shows the regret *saved* relative to the random baseline:

$$\Delta\text{Regret}_T = \text{Regret}_T^{\text{random}} - \text{Regret}_T^{\text{agent}}$$

---

## 16. [`BugsyMazeBuilder.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/all%20materials/Editor/BugsyMazeBuilder.cs)

### Concept: Procedural Maze Generation (Recursive Backtracker)

This Unity Editor script builds the entire level from code via the **BUGSY → Build Maze Level** menu item. It uses a classic algorithm to guarantee a fully connected maze.

---

### Line-by-line (key parts)

```csharp
const int Cols = 11, Rows = 11;  // 11×11 grid of cells
const float C = 6f;              // 6 m per cell
const int Seed = 21;             // fixed seed → same maze every build
const int ExtraLoops = 2;        // add 2 extra passages for multiple routes
```

```csharp
// Wall data structures
static bool[,] hWall;   // horizontal walls: [col, row+1] = true means wall present
static bool[,] vWall;   // vertical walls:   [col+1, row] = true means wall present
```
The maze is encoded as two boolean grids:
- `hWall[i, j]` — is there a wall on the **top** edge of cell (i, j−1)?
- `vWall[i, j]` — is there a wall on the **right** edge of cell (i−1, j)?

```csharp
static void GenerateMaze(System.Random rnd)
{
    // 1. Fill all walls
    hWall all = true; vWall all = true;
    // 2. Recursive backtracker (DFS)
    visited[StartCol, StartRow] = true;
    stack.Push(start);
    while (stack.Count > 0)
    {
        cur = stack.Peek();
        options = unvisited_neighbours(cur);
        if (options.Count == 0) { stack.Pop(); continue; }
        pick = random(options);
        Carve(cur, pick);           // remove wall between cur and pick
        visited[pick] = true;
        stack.Push(pick);
    }
    // 3. Add ExtraLoops extra openings
    // 4. Cut entrance (top) and exit (bottom) gaps
}
```

**Recursive Backtracker Algorithm** (also called DFS maze generation):
1. Start from a random cell, mark it visited.
2. Randomly pick an unvisited neighbour, carve the wall between them, push neighbour onto stack.
3. If no unvisited neighbours, backtrack (pop).
4. Repeat until stack is empty.

**Guarantee:** every cell is visited exactly once → the result is a **spanning tree** of the grid graph → a **perfect maze** (exactly one path between any two cells). `ExtraLoops = 2` then removes 2 random remaining walls to add cycles (multiple routes).

```csharp
static int ShortestPath()
{
    // BFS from start to finish cell
    Queue<Vector2Int> q;
    dist[StartCol, StartRow] = 0;
    while (q.Count > 0) {
        p = q.Dequeue();
        for each open neighbour n:
            dist[n] = dist[p] + 1;
            q.Enqueue(n);
    }
    return dist[FinishCol, FinishRow];
}
```
BFS (Breadth-First Search) on the maze graph computes the **shortest path** (in cells) from entrance to exit. Reported in the console after building: "Shortest route: X cells." This is the minimum task length the agent must learn.

```csharp
static void PlaceObstacles(System.Random rnd)
{
    for each cell (i, j):
        if straightZ or straightX corridor:
            if rnd.NextDouble() > 0.85: skip  // 85% chance of obstacle
            PlaceStraight(rnd.Next(15), ...)   // choose from 15 obstacle types
        else (corner, junction, dead end):
            if rnd.NextDouble() > 0.70: skip  // 70% chance
            PlaceOpen(rnd.Next(7), ...)        // choose from 7 types
}
```
**Obstacle placement** is probabilistic and cell-topology-aware:
- Straight corridors get obstacles that span the width (hurdles, slide bars, moving beams, etc.)
- Corners/junctions/dead-ends get sweepers, pillars, pendulums, or boulders

The 15 straight obstacle types include: hurdle, slide bar, moving beam, gate, side slider, low/high sweeper, windmill, slalom, step block, combo, tunnel, pendulum, laser grid, and boulder.

---

## 17. [`BugsyAgentSetup.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/all%20materials/Editor/BugsyAgentSetup.cs)

### Concept: Editor Automation — One-Click Agent Configuration

```csharp
[MenuItem("BUGSY/Add Agent To Player")]
public static void AddAgent()
```
Adds all necessary ML-Agents components to the Player object with correct settings via the Unity Editor menu.

---

### Line-by-line

```csharp
bp.BrainParameters.VectorObservationSize = 43;
bp.BrainParameters.NumStackedVectorObservations = 1;
bp.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(new int[] { 2, 2, 2, 3 });
```
Sets the observation space to exactly the 43 numbers built by `CollectObservations`, and the action space to 4 discrete axes with sizes [2, 2, 2, 3] (matching `OnActionReceived`). Stacking = 1 means no frame stacking.

```csharp
dr.DecisionPeriod = 5;           // agent decides every 5 physics steps
dr.TakeActionsBetweenDecisions = true;  // repeat last action between decisions
```
**Decision period = 5** means the neural network runs at `100 Hz / 5 = 20 Hz`. `TakeActionsBetweenDecisions = true` repeats the last discrete action in the intermediate physics steps. This is a temporal abstraction: the agent makes coarse decisions while physics runs at full resolution.

```csharp
ray.DetectableTags = new List<string> { "Wall", "Obstacle", "Platform" };
ray.RaysPerDirection = 5;      // 5 rays on each side of forward = 11 total
ray.MaxRayDegrees = 90f;       // ±90° fan
ray.SphereCastRadius = 0.5f;
ray.RayLength = 15f;
```
A **RayPerceptionSensor3D** is added — this casts 11 rays in a fan pattern and returns distance + tag for each. However, based on `CollectObservations`, the 43-D vector sensor is the primary input; this ray sensor adds additional observations on top. Together they form the full observation the network sees.

---

## 18. [`BugsyTrainingMode.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/all%20materials/Editor/BugsyTrainingMode.cs)

### Concept: One-Click Training vs Play Mode Switcher

```csharp
[MenuItem("BUGSY/Training Mode On")]
[MenuItem("BUGSY/Training Mode Off")]
```
Two menu items that apply a consistent set of changes for switching between **evaluation/play** and **training** modes:

| Setting | Training ON | Training OFF (Play) |
|---|---|---|
| `GameManager.trainingMode` | `true` | `false` |
| `BehaviorParameters.BehaviorType` | `Default` (neural net) | `HeuristicOnly` (keyboard/scripted) |
| `DecisionRequester.DecisionPeriod` | `10` (10 Hz decisions) | `5` (20 Hz decisions) |
| `BugsyAgent.MaxStep` | `10000` | `2000` |

**Why different decision periods?** During training (10 Hz), slower decisions reduce computational load and let ML-Agents batch more efficiently. During play (20 Hz), faster decisions give snappier control feel for the scripted baseline or human player.

**Why different MaxStep?** During training, 10,000 steps × 0.01 s = 100 s per episode max. During play, 2,000 × 0.01 s = 20 s — enough for human/demo play but not so long it hangs.

---

## 19. [`BugsyReachableCells.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/all%20materials/Scripts/BugsyReachableCells.cs)

### Concept: Maze Analysis via BFS — Ground Truth for Evaluation

```csharp
[MenuItem("BUGSY/Count Reachable Cells")]
static void Count()
```
Runs a **Breadth-First Search** from the agent's current position through the `CoverageTracker` grid, using physics raycasts to determine which cells are connected (no wall between them). The result (`REACHABLE = 492`) is the denominator used in `make_figures.py`.

---

### Line-by-line

```csharp
static Vector3 Center(CoverageTracker ct, int gx, int gz)
{
    float x = (gx + 0.5f) * ct.gridSize - ct.origin;
    float z = (gz + 0.5f) * ct.gridSize - ct.origin;
    return new Vector3(x, 1f, z);
}
```
Converts grid indices back to world-space cell centres:

$$x_{\text{world}} = \left(g_x + 0.5\right) \times \text{gridSize} - \text{origin}$$

The inverse of `CellX()` in `CoverageTracker`.

```csharp
static bool Blocked(Vector3 from, Vector3 to)
{
    RaycastHit[] hits = Physics.RaycastAll(from, dir.normalized, len, ...);
    foreach (h in hits) if (h.collider.CompareTag("Wall")) return true;
    return false;
}
```
A ray between two cell centres is blocked if it hits a wall-tagged collider. This is the **maze connectivity test** — equivalent to checking whether the maze passage exists between two cells.

```csharp
// BFS loop
Queue<Vector2Int> q;
q.Enqueue(start);
while (q.Count > 0)
{
    c = q.Dequeue(); count++;
    for each of 4 neighbours n:
        if not seen[n] && not Blocked(Center(c), Center(n)):
            seen[n] = true; q.Enqueue(n);
}
Debug.Log("REACHABLE CELLS = " + count);
```
Standard **BFS flood fill** on the grid. Time complexity $O(N^2)$ for an $N \times N$ grid (here 30×30 = 900 cells max). The maze has 492 reachable cells out of a theoretical maximum of 11×11 = 121 maze cells × (6/3)² = 4 tracker cells each = 484 tracker cells — the slight excess comes from the entry/exit porches.

---

## 20. Key RL Concepts & Equations

### Proximal Policy Optimisation (PPO)

PPO is the training algorithm used by Unity ML-Agents. It optimises a stochastic policy $\pi_\theta$ by maximising a clipped surrogate objective:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t \right) \right]$$

where:
- $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio
- $\hat{A}_t$ is the advantage estimate
- $\varepsilon$ (typically 0.2) clips the update to prevent destructively large policy steps

### Advantage Estimation (GAE)

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

where $\gamma$ = discount factor (e.g. 0.99) and $\lambda$ = GAE smoothing parameter (e.g. 0.95).

### Intrinsic Motivation (Count-Based Novelty)

$$r_t = r_t^{\text{ext}} + \beta \cdot r_t^{\text{int}}$$

In BUGSY's default mode, $r_t^{\text{ext}} = 0$ (no extrinsic task reward) and:

$$r_t^{\text{int}} = \begin{cases} 0.1 & \text{first visit to cell } (g_x, g_z) \text{ this episode} \\ 0 & \text{otherwise} \end{cases}$$

### Observation Normalisation

All position observations are divided by `MazeSize = 66`:

$$\tilde{x} = \frac{x}{66}, \quad \tilde{z} = \frac{z}{66}$$

This maps position into [0, 1], which matches the typical activation ranges of neural network hidden layers and improves training stability.

### Heading Encoding

Raw Euler angle $\theta$ is discontinuous at 0°/360°. Encoding as a unit-circle point:

$$\vec{h} = (\sin\theta, \cos\theta)$$

is continuous and the neural network can interpolate between any two headings without a discontinuity jump.

### Episode Termination Conditions Summary

| Condition | Mechanism | RL effect |
|---|---|---|
| Agent reaches `FinishZone` | `GameManager.Win()` → `EndEpisode()` | +1 reward (task mode only) |
| Agent loses all 3 lives | `GameManager.GameOver` → `EndEpisode()` | Episode ends, no penalty |
| Agent stuck (no new cell 8 s) | `stuckSteps >= 800` in `FixedUpdate` | Episode ends, negative implicit signal |
| Max steps exceeded | ML-Agents Academy config | Episode ends |

### Ablation Design

| Mode | Novelty Reward | Visited Map Obs | Finish Reward |
|---|---|---|---|
| Default (exploration PPO) | ✅ +0.1/cell | ✅ 25 obs | ❌ |
| `taskRewardMode = true` | ❌ | ❌ (zeroed) | ✅ +1 |
| `hideVisitedMap = true` | ✅ | ❌ (zeroed) | ❌ |
| `randomActions = true` | N/A (heuristic baseline) | N/A | N/A |

---

---

## Complete File Inventory

| File | Location | Type | Role |
|---|---|---|---|
| [`BugsyAgent.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/BugsyAgent.cs) | Assets/Scripts | RL Core | Agent observations, actions, episode logic |
| [`CoverageTracker.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/CoverageTracker.cs) | Assets/Scripts | RL Core | Intrinsic reward + visited-cell observations |
| [`WallFollowerBrain.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/WallFollowerBrain.cs) | Assets/Scripts | Baseline | Scripted right-hand-rule maze solver |
| [`PlayerController.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/PlayerController.cs) | Assets/Scripts | Actuator | Physics character controller |
| [`GameManager.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/GameManager.cs) | Assets/Scripts | Lifecycle | Lives, state machine, restart |
| [`FinishZone.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/FinishZone.cs) | Assets/Scripts | Environment | Win trigger |
| [`BoulderTrap.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/BoulderTrap.cs) | Assets/Scripts | Hazard | Proximity-drop boulder |
| [`LaserGrid.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/LaserGrid.cs) | Assets/Scripts | Hazard | Timed on/off laser |
| [`MovingPlatform.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/MovingPlatform.cs) | Assets/Scripts | Obstacle | PingPong platform |
| [`Pendulum.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/Pendulum.cs) | Assets/Scripts | Obstacle | Sinusoidal pendulum |
| [`Rotator.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/Rotator.cs) | Assets/Scripts | Obstacle | Constant rotation |
| [`CameraFollow.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/Assets/Scripts/CameraFollow.cs) | Assets/Scripts | Viewer | Smart 3rd-person camera |
| [`bugsy_ppo.yaml`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/bugsy_ppo.yaml) | Root | Config | PPO hyperparameters |
| [`make_figures.py`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/make_figures.py) | Root | Analysis | Evaluation plots + statistics |
| [`BugsyMazeBuilder.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/all%20materials/Editor/BugsyMazeBuilder.cs) | all materials/Editor | Tool | Procedural maze + obstacle generation |
| [`BugsyAgentSetup.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/all%20materials/Editor/BugsyAgentSetup.cs) | all materials/Editor | Tool | One-click agent component setup |
| [`BugsyTrainingMode.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/all%20materials/Editor/BugsyTrainingMode.cs) | all materials/Editor | Tool | Training ↔ play mode switcher |
| [`BugsyReachableCells.cs`](file:///c:/Users/Kezia/OneDrive/Desktop/sem/S7/RL/Project/BUGSY_Parkour/all%20materials/Scripts/BugsyReachableCells.cs) | all materials/Scripts | Analysis | BFS cell reachability counter |

---

*Generated for BUGSY Parkour RL Project — Semester 7*
