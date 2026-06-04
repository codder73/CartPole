# 🎪 CartPole RL Lab
### Reinforcement Learning + Imitation Learning — Built in React, Runs in the Browser

> Train an AI agent to balance a pole on a moving cart — from zero knowledge to master balancer — entirely in your browser. No Python. No servers. No GPU. Just JavaScript, physics, and pure Q-Learning.

---

## 📸 What It Looks Like

```
┌─────────────────────────────────────────────────────────┐
│  🤖 AI MODE          SCORE: 247                         │
│                                                         │
│         ╱                                               │
│        ╱  ← glowing pole (green = balanced)             │
│       ╱                                                 │
│  ════[████████]════════════════════════  ← track        │
│         cart                                            │
│                                                         │
│  Episode: 312  Score: 247  Best: 891  ε: 0.043         │
└─────────────────────────────────────────────────────────┘
```

---

## 🧠 What This Project Teaches

This is not just a game. Every feature directly maps to a real concept used in production AI systems:

| Concept | Where You See It |
|---|---|
| Environment & State Space | The 4-number world description |
| Q-Learning (Tabular RL) | The agent's brain updating every step |
| Bellman Equation | The math running 1200 times/second |
| ε-Greedy Exploration | Epsilon bar decaying from 1.0 → 0.01 |
| Reward Shaping | Angle + position penalties guiding learning |
| Imitation Learning | Teach Mode — you play, AI copies |
| Behavioural Cloning | AlphaGo's first training stage, reproduced here |
| Persistent Model Storage | Save/Load Q-table via localStorage |
| Policy Exploitation | Showcase Mode — epsilon locked at 0 |

---

## ⚙️ Technical Specifications

### Physics Engine

| Parameter | Value | Meaning |
|---|---|---|
| `GRAVITY` | 9.8 m/s² | Real-world gravity |
| `CART_MASS` | 1.0 kg | Mass of the cart |
| `POLE_MASS` | 0.1 kg | Mass of the pole |
| `POLE_HALF_LENGTH` | 0.5 m | Half the pole length (1m total) |
| `FORCE_MAG` | 10.0 N | Push force per action |
| `DT` | 0.02 s | Time step (50 steps/second) |
| Termination angle | ±12° (±0.209 rad) | Pole fall threshold |
| Termination position | ±2.4 units | Cart off-track threshold |

The physics use the exact **Euler integration** equations from the classic CartPole-v1 OpenAI Gym environment. The pole dynamics are computed via the Lagrangian mechanics formula accounting for the coupled cart-pole system.

### Q-Learning Agent

| Hyperparameter | Value | Role |
|---|---|---|
| `alpha` (learning rate) | 0.3 | How strongly each experience updates Q-values |
| `gamma` (discount factor) | 0.99 | How much future rewards matter |
| `epsilon` (initial) | 1.0 | 100% random exploration at start |
| `epsilon` (minimum) | 0.01 | 1% random exploration floor |
| `epsilon` decay rate | × 0.998 per step | Gradual shift from explore → exploit |
| `alpha` (imitation) | 0.8 | Stronger update for human-demonstrated actions |

### State Discretization

The continuous 4D state space is discretized into bins for the Q-table:

| State Variable | Range | Bins | Resolution |
|---|---|---|---|
| Cart position `x` | [-2.4, 2.4] | 10 | 0.48 units/bin |
| Cart velocity `ẋ` | [-4.0, 4.0] | 10 | 0.8 units/bin |
| Pole angle `θ` | [-0.21, 0.21] rad | 10 | 0.042 rad/bin (~2.4°) |
| Pole angular velocity `θ̇` | [-4.0, 4.0] | 10 | 0.8 units/bin |

**Total possible states:** 10⁴ = **10,000 unique states**
**Total Q-table entries:** up to 20,000 (2 actions × 10,000 states)

### Reward Function

```
r(s, done) = {
  -10                          if done (pole fell or cart off-track)
  1 - 2|θ| - 0.1|x|           otherwise
}
```

- Base survival reward: **+1** per timestep
- Angle penalty: **-2|θ|** — strongly discourages leaning
- Position penalty: **-0.1|x|** — softly encourages staying centered
- Failure penalty: **-10** — large negative signal on termination

### Bellman Update (per step)

```
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') − Q(s, a)]
```

Where:
- `Q(s, a)` = current value estimate for state s, action a
- `r` = shaped reward received
- `γ × max_a' Q(s', a')` = discounted best future value
- `α` = learning rate (0.3 normal, 0.8 imitation)

---

## 🗂️ Project Structure

```
cartpole-rl-local.jsx        ← entire project (single file)
│
├── Physics Engine            cartpoleStep(), randomState()
├── Q-Learning Core           createAgent(), agentAct(), agentLearn()
├── Imitation Learning        imitationLearn()
├── Reward Shaping            shapeReward()
├── State Discretization      disc(), stateKey()
├── Storage (localStorage)    saveAgent(), loadAgent(), clearSave()
├── Live Explanation Engine   getLiveExplanation()
├── Canvas Renderer           drawCartPole()
├── Concept Definitions       CONCEPTS[]
└── React App Component       CartPoleApp (default export)
```

---

## 🚀 Getting Started

### Prerequisites

- Node.js v18+ — [download here](https://nodejs.org)
- A modern browser (Chrome, Firefox, Edge, Safari)

### Installation

```bash
# 1. Create a new Vite + React project
npm create vite@latest cartpole-rl -- --template react
cd cartpole-rl

# 2. Install dependencies
npm install

# 3. Drop in the component
#    Copy cartpole-rl-local.jsx into the src/ folder

# 4. Wire it up — replace src/App.jsx with:
```

```jsx
// src/App.jsx
import CartPoleApp from './cartpole-rl-local'
export default function App() { return <CartPoleApp /> }
```

```bash
# 5. Run it
npm run dev
```

Open `http://localhost:5173` — done. ✅

### Zero-Install Options

| Platform | Link | Notes |
|---|---|---|
| StackBlitz | [stackblitz.com/fork/react](https://stackblitz.com/fork/react) | Paste file, instant run, shareable URL |
| CodeSandbox | [codesandbox.io](https://codesandbox.io) | New React sandbox, paste file |

---

## 🎮 How to Use

### Mode 1 — AI Trains (Autonomous)

1. Click **▶ START**
2. Watch the agent fail repeatedly at first (this is normal — it's exploring)
3. Use **speed buttons (1x / 3x / 8x / 20x)** to fast-forward training
4. By episode ~100–200 scores should start climbing noticeably
5. When BEST score is consistently 300+, the agent has learned well
6. Click **💾 SAVE** to preserve the trained brain

### Mode 2 — Teach Mode (Imitation Learning)

1. Click **👨‍🏫 TEACH MODE**
2. Click **▶ START**
3. Use keyboard **← → arrow keys** (or **A / D**) to control the cart
4. Try to keep the pole balanced — your moves seed the Q-table in real time
5. Even 5–10 good rounds dramatically accelerates later AI training
6. Switch back to **🤖 AI TRAINS** to let the agent refine from your demonstrations

> This is the same technique AlphaGo used — supervised learning on expert human games before self-play RL.

### Mode 3 — Showcase (Free Run)

1. After training and saving, click **🏆 SHOWCASE**
2. The agent runs freely with its saved brain — epsilon locked at **0.000**
3. No learning, no randomness — pure skill demonstration
4. Cart glows **gold**, pole turns **green** when near-perfect balance
5. Hit **⛶** to watch in fullscreen

---

## 💾 Save / Load System

The agent's brain is stored in **`localStorage`** under the key `cartpole_qtable_v2`.

| What Gets Saved | Type |
|---|---|
| Full Q-table | Object (up to ~20,000 key-value pairs) |
| Current epsilon | Float (0.01 – 1.0) |
| Episode count | Integer |
| Best score | Integer |
| Score history (last 30) | Array |
| Save timestamp | String |

**Persistence:** Survives browser close, tab close, computer restart. Cleared only by clicking 🗑 or clearing browser data.

**Storage size:** A fully trained Q-table is typically 100–400 KB in localStorage — well within the 5MB browser limit.

---

## ⌨️ Keyboard Controls

| Key | Action |
|---|---|
| `←` or `A` | Push cart LEFT (Teach Mode only) |
| `→` or `D` | Push cart RIGHT (Teach Mode only) |
| `Escape` | Exit fullscreen |

---

## 📊 Live Stats Explained

| Stat | Meaning |
|---|---|
| **EPISODE** | Number of complete runs (reset to reset) |
| **SCORE** | Steps survived in the current episode |
| **BEST** | Highest score achieved across all episodes |
| **ε (EXPLORE)** | Current epsilon — probability of random action |
| **Q-Table entries** | How many state-action pairs have been visited |
| **Imitation demos** | Human-demonstrated steps recorded |

---

## 📚 RL Concepts Reference

| Concept | Formula | In Code |
|---|---|---|
| **State Space** | s ∈ ℝ⁴ | `[x, xd, th, thd]` |
| **Action Space** | A = {0, 1} | `0` = LEFT, `1` = RIGHT |
| **Reward** | r = 1 − 2\|θ\| − 0.1\|x\| | `shapeReward()` |
| **Q-Value** | Q(s,a) ∈ ℝ | `agent.qTable` |
| **Bellman** | Q ← Q + α[r + γ maxQ(s') − Q] | `agentLearn()` |
| **ε-Greedy** | π(s) = argmax Q(s,a) | `agentAct()` |
| **Imitation** | α = 0.8, expert label | `imitationLearn()` |
| **Termination** | \|θ\| > 12° or \|x\| > 2.4 | `cartpoleStep()` |

---

## 🏆 Connection to AlphaGo

AlphaGo (DeepMind, 2016) used a two-phase training strategy:

**Phase 1 — Supervised Learning:** Trained on 30 million moves from human grandmasters. The network learned to predict what a strong human player would do. This is exactly what **Teach Mode** does — human demonstrations directly update the Q-table with a high learning rate (α = 0.8).

**Phase 2 — Reinforcement Learning:** Self-play games where AlphaGo played against itself and updated its policy based on wins/losses. This is exactly what **AI Trains mode** does — the agent plays against the physics and updates from reward signals.

The difference in scale is enormous (neural networks vs Q-table, 19×19 Go vs 4 numbers). But the fundamental learning loop is identical.

---

## 🔧 Customization

Want to experiment? Key constants to change at the top of the file:

```js
// Make learning faster/slower
alpha: 0.3        // try 0.1 (slower) or 0.5 (faster, less stable)

// Change future reward weighting
gamma: 0.99       // try 0.9 (more short-sighted) or 1.0 (fully long-term)

// Change state resolution
const BINS = 10   // try 6 (coarser) or 14 (finer, slower to train)

// Make exploration decay faster
epsilon * 0.998   // try 0.995 (faster decay) or 0.9995 (slower)

// Change failure penalty
return -10        // try -1 (softer) or -50 (harsh)
```

---

## 🛠️ Built With

| Technology | Version | Purpose |
|---|---|---|
| React | 18+ | UI framework |
| HTML5 Canvas API | — | Real-time physics rendering |
| localStorage API | — | Persistent Q-table storage |
| Vite | 5+ | Build tool / dev server |
| Vanilla JavaScript | ES2020+ | Physics engine + RL algorithm |

Zero external dependencies beyond React itself.

---

## 📄 License

MIT — free to use, modify, learn from, and share.

---

## 🙏 Acknowledgements

- **Richard Bellman (1957)** — the Bellman equation at the heart of Q-learning
- **Watkins & Dayan (1992)** — original Q-learning paper
- **OpenAI Gym** — the CartPole-v1 environment this is based on
- **DeepMind / AlphaGo** — inspiration for the imitation learning feature

---

*"The agent does not read about physics — it lives through it, ten thousand times, until balance becomes second nature."*
