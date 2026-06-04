import { useState, useEffect, useRef, useCallback } from "react";

// ─── PHYSICS ────────────────────────────────────────────────────────────────
const GRAVITY = 9.8, CART_MASS = 1.0, POLE_MASS = 0.1;
const POLE_HALF = 0.5, FORCE = 10.0, DT = 0.02;
const TOTAL_MASS = CART_MASS + POLE_MASS;

function cartpoleStep(state, action) {
  const [x, xd, th, thd] = state;
  const f = action === 1 ? FORCE : -FORCE;
  const cosT = Math.cos(th), sinT = Math.sin(th);
  const tmp = (f + POLE_MASS * POLE_HALF * thd * thd * sinT) / TOTAL_MASS;
  const thAcc = (GRAVITY * sinT - cosT * tmp) / (POLE_HALF * (4/3 - POLE_MASS * cosT * cosT / TOTAL_MASS));
  const xAcc = tmp - POLE_MASS * POLE_HALF * thAcc * cosT / TOTAL_MASS;
  const nx = x + DT * xd, nxd = xd + DT * xAcc;
  const nth = th + DT * thd, nthd = thd + DT * thAcc;
  const done = Math.abs(nx) > 2.4 || Math.abs(nth) > (12 * Math.PI / 180);
  return { nextState: [nx, nxd, nth, nthd], done };
}
function randomState() { return Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.1); }

// ─── Q-LEARNING ──────────────────────────────────────────────────────────────
const BINS = 10;
function disc(v, lo, hi) { return Math.min(BINS-1, Math.floor(((Math.max(lo,Math.min(hi,v))-lo)/(hi-lo))*BINS)); }
function stateKey([x,xd,th,thd]) { return `${disc(x,-2.4,2.4)},${disc(xd,-4,4)},${disc(th,-0.21,0.21)},${disc(thd,-4,4)}`; }
function createAgent() { return { qTable:{}, epsilon:1.0, alpha:0.3, gamma:0.99 }; }
function getQ(agent,key,a) { return agent.qTable[`${key}_${a}`]??0; }
function agentAct(agent, state, forceGreedy=false) {
  if (!forceGreedy && Math.random() < agent.epsilon) return Math.random()<0.5?0:1;
  const k = stateKey(state);
  return getQ(agent,k,0)>=getQ(agent,k,1)?0:1;
}
function shapeReward(state, done) {
  if (done) return -10;
  const [x,,th] = state;
  return 1 - Math.abs(th)*2 - Math.abs(x)*0.1;
}
function agentLearn(agent, state, action, reward, nextState, done) {
  const k=stateKey(state), nk=stateKey(nextState);
  const cQ=getQ(agent,k,action);
  const maxNQ=done?0:Math.max(getQ(agent,nk,0),getQ(agent,nk,1));
  const nQ=cQ+agent.alpha*(reward+agent.gamma*maxNQ-cQ);
  return { ...agent, qTable:{...agent.qTable,[`${k}_${action}`]:nQ}, epsilon:Math.max(0.01,agent.epsilon*0.998) };
}
function imitationLearn(agent, state, action, nextState, done) {
  const k=stateKey(state), nk=stateKey(nextState);
  const reward=done?-10:1-Math.abs(state[2])*2-Math.abs(state[0])*0.1;
  const cQ=getQ(agent,k,action);
  const maxNQ=done?0:Math.max(getQ(agent,nk,0),getQ(agent,nk,1));
  const nQ=cQ+0.8*(reward+agent.gamma*maxNQ-cQ);
  return { ...agent, qTable:{...agent.qTable,[`${k}_${action}`]:nQ} };
}

// ─── STORAGE ─────────────────────────────────────────────────────────────────
const STORAGE_KEY = "cartpole_qtable_v2";
function saveAgent(agent, episode, best, history) {
  try {
    const data = { qTable: agent.qTable, epsilon: agent.epsilon, episode, best, history, savedAt: new Date().toLocaleTimeString() };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    // Also mirror to sessionStorage as fallback
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    return data.savedAt;
  } catch(e) { return null; }
}
function loadAgent() {
  try {
    // Try sessionStorage first (always works in artifacts)
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      if (data?.qTable && Object.keys(data.qTable).length > 0) return data;
    }
  } catch(e) {}
  return null;
}
function clearSave() {
  try { localStorage.removeItem(STORAGE_KEY); window.storage?.delete(STORAGE_KEY); } catch(e){}
}

// ─── LIVE EXPLANATION ────────────────────────────────────────────────────────
function getLiveExplanation(state, action, epsilon, episode, score, mode, justDied, isShowcase) {
  if (!state) return null;
  const [x,xd,th] = state;
  const deg = (th*180/Math.PI).toFixed(1);
  const isLeft = action===0;
  if (isShowcase) {
    if (Math.abs(th)<0.02) return { color:"#34d399", icon:"🏆", title:"Master Balance — Pure Exploitation", concept:"Trained Policy", text:`Pole is nearly perfect at ${deg}°. This is your SAVED trained brain running freely — epsilon is locked at 0 so it always picks the best known action. Pure exploitation, zero randomness. Score: ${score}.` };
    if (Math.abs(th)>0.10) return { color:"#f87171", icon:"⚡", title:"Urgent Recovery", concept:"Reactive Policy", text:`Pole at ${deg}°! The trained Q-table immediately fires the correct recovery action — ${isLeft?"◀ LEFT":"▶ RIGHT"}. This response was learned over hundreds of episodes and is now instant.` };
    return { color:"#a78bfa", icon:"🎭", title:"Showcase Mode — Watch the Expert", concept:"Saved Q-Table Playback", text:`Your trained agent is performing freely with its saved brain. No learning, no exploration — just the pure skill it built during training. Pole at ${deg}°, score: ${score}.` };
  }
  if (justDied) return { color:"#f87171", icon:"💀", title:"Episode Ended — Failure Signal", concept:"Reward Shaping + Termination", text:`Pole fell at ${deg}°. Agent receives -10 penalty. The Bellman update propagates this failure backwards through recent Q-table entries.` };
  if (mode==="manual") return { color:"#34d399", icon:"👨‍🏫", title:"Imitation Learning — You Are the Expert", concept:"Supervised RL / Behavioural Cloning", text:`Like AlphaGo learning from grandmasters, the agent watches YOUR actions and updates its Q-table with α=0.8. Pole at ${deg}°. You pressed ${isLeft?"◀ LEFT":"▶ RIGHT"}. Agent records: "human chose this — it must be good."` };
  if (Math.abs(th)<0.03) return { color:"#34d399", icon:"⚖️", title:"Pole Near Balanced — Exploitation", concept:"Policy Exploitation (ε-greedy)", text:`Pole angle only ${deg}° — almost perfect. ε=${epsilon.toFixed(3)} is low, agent exploiting Q-table. Chose ${isLeft?"◀ LEFT":"▶ RIGHT"} because Q(s,${isLeft?"left":"right"}) is highest.` };
  if (Math.abs(th)>0.15) return { color:"#f87171", icon:"🚨", title:"Critical Angle — Urgent Correction", concept:"State Space + Reward Signal", text:`Pole dangerously tilted at ${deg}°! Reward penalty: ${(-Math.abs(th)*2).toFixed(2)}. Agent pushed ${isLeft?"◀ LEFT":"▶ RIGHT"} to recover.` };
  if (episode<30) return { color:"#fbbf24", icon:"🎲", title:"Early Training — Random Exploration", concept:"Exploration vs Exploitation", text:`Episode ${episode}. ε=${epsilon.toFixed(3)} is high — mostly random actions. This is necessary! The agent is mapping the state space like a baby learning to walk by falling.` };
  return { color:"#a78bfa", icon:"🧠", title:"Bellman Update Firing", concept:"Q-Learning / Bellman Equation", text:`Pole at ${deg}°, velocity ${xd.toFixed(2)}. Agent chose ${isLeft?"◀ LEFT":"▶ RIGHT"}. Q(s,a) ← Q + 0.3×[r + 0.99×maxQ(s') − Q]. Score: ${score}.` };
}

// ─── DRAW ─────────────────────────────────────────────────────────────────────
function drawCartPole(canvas, state, score, mode, humanAction, isShowcase, isFullscreen) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);

  const bg = ctx.createLinearGradient(0,0,0,H);
  if (isShowcase) { bg.addColorStop(0,"#040414"); bg.addColorStop(1,"#08082a"); }
  else { bg.addColorStop(0,"#07071a"); bg.addColorStop(1,"#0d0d28"); }
  ctx.fillStyle=bg; ctx.fillRect(0,0,W,H);

  // Grid
  ctx.strokeStyle="rgba(255,255,255,0.025)"; ctx.lineWidth=1;
  const gx = isFullscreen?80:50, gy = isFullscreen?80:50;
  for(let i=0;i<W;i+=gx){ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i,H);ctx.stroke();}
  for(let i=0;i<H;i+=gy){ctx.beginPath();ctx.moveTo(0,i);ctx.lineTo(W,i);ctx.stroke();}

  const [x,,th] = state;
  const scale = W/5.2;
  const cx=W/2+x*scale, cy=H*0.60;
  const cW=isFullscreen?90:64, cH=isFullscreen?40:32, pL=isFullscreen?170:120;

  // Showcase glow effect on track
  if (isShowcase) {
    const trackGlow = ctx.createLinearGradient(15,0,W-30,0);
    trackGlow.addColorStop(0,"rgba(0,212,255,0)");
    trackGlow.addColorStop(0.5,"rgba(0,212,255,0.25)");
    trackGlow.addColorStop(1,"rgba(0,212,255,0)");
    ctx.fillStyle=trackGlow; ctx.fillRect(15,cy+cH/2-2,W-30,8);
  }

  // Track
  ctx.fillStyle="rgba(0,212,255,0.12)"; ctx.fillRect(15,cy+cH/2,W-30,5);
  ctx.shadowColor="#00d4ff"; ctx.shadowBlur=12;
  ctx.fillStyle="#00d4ff"; ctx.fillRect(15,cy+cH/2,W-30,2);
  ctx.shadowBlur=0;

  // Mode badge
  const badgeColor = isShowcase?"#fbbf24":mode==="manual"?"#34d399":"#a78bfa";
  const badgeText = isShowcase?"🏆 SHOWCASE":"" + (mode==="manual"?"👨‍🏫 TEACH":"🤖 AI MODE");
  ctx.fillStyle=`${badgeColor}22`;
  ctx.beginPath(); ctx.roundRect(8,8,isFullscreen?160:120,isFullscreen?50:38,6); ctx.fill();
  ctx.strokeStyle=`${badgeColor}55`; ctx.lineWidth=1; ctx.stroke();
  ctx.fillStyle=badgeColor; ctx.font=`bold ${isFullscreen?13:10}px monospace`;
  ctx.fillText(isShowcase?"🏆 SHOWCASE MODE":mode==="manual"?"👨‍🏫 TEACH MODE":"🤖 AI MODE",16,isFullscreen?28:22);
  ctx.fillStyle="#fff"; ctx.font=`bold ${isFullscreen?20:14}px monospace`;
  ctx.fillText(`SCORE: ${score}`,16,isFullscreen?46:38);

  // Cart shadow
  ctx.fillStyle="rgba(0,212,255,0.08)";
  ctx.beginPath(); ctx.ellipse(cx,cy+cH/2+10,cW*0.55,9,0,0,Math.PI*2); ctx.fill();

  // Cart — gold in showcase mode
  const c1=isShowcase?"#3a2a0a":mode==="manual"?"#1a3a2a":"#1a3a6a";
  const c2=isShowcase?"#b8860b":mode==="manual"?"#0e8050":"#0e5fa8";
  const c3=isShowcase?"#2a1a05":mode==="manual"?"#0a2a1a":"#0a2a50";
  const cg=ctx.createLinearGradient(cx-cW/2,cy-cH/2,cx+cW/2,cy+cH/2);
  cg.addColorStop(0,c1); cg.addColorStop(0.5,c2); cg.addColorStop(1,c3);
  ctx.fillStyle=cg;
  ctx.beginPath(); ctx.roundRect(cx-cW/2,cy-cH/2,cW,cH,7); ctx.fill();
  const borderCol=isShowcase?"#fbbf24":mode==="manual"?"#34d399":"#00d4ff";
  ctx.strokeStyle=borderCol; ctx.lineWidth=isFullscreen?2:1.5;
  ctx.shadowColor=borderCol; ctx.shadowBlur=isFullscreen?20:12; ctx.stroke(); ctx.shadowBlur=0;

  // Manual arrow
  if (mode==="manual" && humanAction!==null) {
    const arrowX=humanAction===0?cx-cW/2-22:cx+cW/2+8;
    ctx.fillStyle="#34d399"; ctx.shadowColor="#34d399"; ctx.shadowBlur=10;
    ctx.font=`bold ${isFullscreen?28:20}px monospace`;
    ctx.fillText(humanAction===0?"◀":"▶",arrowX,cy+5); ctx.shadowBlur=0;
  }

  // Wheels
  const wheelR=isFullscreen?12:9;
  [-cW*0.3,cW*0.3].forEach(ox=>{
    ctx.fillStyle="#09182e"; ctx.strokeStyle=borderCol;
    ctx.lineWidth=2; ctx.shadowColor=borderCol; ctx.shadowBlur=6;
    ctx.beginPath(); ctx.arc(cx+ox,cy+cH/2+3,wheelR,0,Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.shadowBlur=0; ctx.fillStyle=borderCol;
    ctx.beginPath(); ctx.arc(cx+ox,cy+cH/2+3,wheelR*0.28,0,Math.PI*2); ctx.fill();
  });

  // Pole
  const px=cx+pL*Math.sin(th), py=(cy-cH/2)-pL*Math.cos(th);
  const ang=Math.abs(th);
  const pr=Math.min(255,Math.floor(ang*1400)), pg=Math.max(0,Math.floor(255-ang*1400));
  const poleColor=isShowcase&&ang<0.05?`rgb(50,255,100)`:`rgb(${pr},${pg},40)`;
  ctx.strokeStyle=poleColor; ctx.lineWidth=isFullscreen?12:9; ctx.lineCap="round";
  ctx.shadowColor=poleColor; ctx.shadowBlur=isFullscreen?25:18;
  ctx.beginPath(); ctx.moveTo(cx,cy-cH/2); ctx.lineTo(px,py); ctx.stroke();
  ctx.shadowBlur=0;
  ctx.fillStyle=poleColor; ctx.shadowColor=poleColor; ctx.shadowBlur=isFullscreen?30:22;
  ctx.beginPath(); ctx.arc(px,py,isFullscreen?11:8,0,Math.PI*2); ctx.fill();
  ctx.shadowBlur=0;

  // Showcase: particle trail effect on pole tip
  if (isShowcase && ang<0.08) {
    ctx.fillStyle="rgba(50,255,100,0.15)";
    ctx.beginPath(); ctx.arc(px,py,isFullscreen?30:20,0,Math.PI*2); ctx.fill();
  }
}

// ─── CONCEPTS ────────────────────────────────────────────────────────────────
const CONCEPTS = [
  { id:"env",tag:"ENVIRONMENT",color:"#00d4ff",icon:"🌍",title:"The Environment",desc:"The world the agent lives in. CartPole has a cart on a track with a pole on top. It responds to actions and returns observations + rewards.",formula:"State = [x, ẋ, θ, θ̇]" },
  { id:"state",tag:"STATE SPACE",color:"#a78bfa",icon:"👁️",title:"State / Observation",desc:"What the agent sees: cart position (x), cart velocity (ẋ), pole angle (θ), pole angular velocity (θ̇). These 4 numbers describe the entire world.",formula:"s ∈ ℝ⁴" },
  { id:"action",tag:"ACTION SPACE",color:"#34d399",icon:"🎮",title:"Actions",desc:"What the agent can do — only 2 choices: push LEFT or RIGHT. Discrete, binary, but enough to master balance.",formula:"A = {0: ◀ Left, 1: ▶ Right}" },
  { id:"reward",tag:"REWARD",color:"#fbbf24",icon:"🏆",title:"Reward Signal",desc:"Feedback signal. +1 per timestep alive, minus angle penalty and position penalty. Failure = -10. Agent maximises total reward.",formula:"r = 1 − 2|θ| − 0.1|x|" },
  { id:"policy",tag:"ε-GREEDY",color:"#f87171",icon:"🧠",title:"Policy & Exploration",desc:"With prob ε the agent picks randomly (explore). Otherwise it picks the best known action (exploit). ε decays from 1 → 0.01 as training progresses.",formula:"π(s) = argmax_a Q(s,a)" },
  { id:"qtable",tag:"Q-LEARNING",color:"#fb923c",icon:"📊",title:"Q-Table",desc:"A lookup table: Q[state][action] = expected future reward. Updated every step using Bellman. Grows as the agent visits new states.",formula:"Q(s,a) ← Q + α[r + γ maxQ(s') − Q]" },
  { id:"imitation",tag:"IMITATION",color:"#e879f9",icon:"👨‍🏫",title:"Imitation Learning",desc:"Like AlphaGo learning from grandmaster games — YOU play and the agent copies your strategy with a high learning rate (α=0.8), seeding better Q-values instantly.",formula:"α_imitation = 0.8 >> α_normal" },
  { id:"bellman",tag:"BELLMAN",color:"#22d3ee",icon:"⚡",title:"Bellman Equation",desc:"Value of current state = immediate reward + discounted best future value. γ (gamma=0.99) means future rewards matter almost as much as immediate ones.",formula:"V(s) = r + γ · max_a V(s')" },
];

// ─── APP ──────────────────────────────────────────────────────────────────────
export default function CartPoleApp() {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const agentRef = useRef(createAgent());
  const stateRef = useRef(randomState());
  const runningRef = useRef(false);
  const modeRef = useRef("ai");
  const animRef = useRef(null);
  const speedRef = useRef(1);
  const humanActionDisplayRef = useRef(null);
  const manualIntervalRef = useRef(null);
  const isShowcaseRef = useRef(false);
  const isFullscreenRef = useRef(false);

  const epRef=useRef(0), scoreRef=useRef(0), bestRef=useRef(0), stepsRef=useRef(0), imitRef=useRef(0);

  const [mode, setMode] = useState("ai");
  const [isRunning, setIsRunning] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [score, setScore] = useState(0);
  const [best, setBest] = useState(0);
  const [epsilon, setEpsilon] = useState(1.0);
  const [history, setHistory] = useState([]);
  const [activeConcept, setActiveConcept] = useState(0);
  const [lastAction, setLastAction] = useState(null);
  const [speed, setSpeed] = useState(1);
  const [liveExp, setLiveExp] = useState(null);
  const [qSize, setQSize] = useState(0);
  const [imitCount, setImitCount] = useState(0);
  const [isShowcase, setIsShowcase] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null); // null | "saved" | "loaded" | "cleared"
  const [savedAt, setSavedAt] = useState(null);
  const [hasSave, setHasSave] = useState(false);

  // Check for existing save on mount
  useEffect(() => {
    const saved = loadAgent();
    if (saved) { setHasSave(true); setSavedAt(saved.savedAt); }
  }, []);

  // ── SAVE ──
  const handleSave = useCallback(() => {
    const t = saveAgent(agentRef.current, epRef.current, bestRef.current, history);
    if (t) { setSaveStatus("saved"); setSavedAt(t); setHasSave(true); setTimeout(()=>setSaveStatus(null),2500); }
  }, [history]);

  // ── LOAD ──
  const handleLoad = useCallback(() => {
    const data = loadAgent();
    if (!data) return;
    handlePause();
    agentRef.current = { ...createAgent(), qTable: data.qTable, epsilon: data.epsilon };
    epRef.current = data.episode||0; bestRef.current = data.best||0; scoreRef.current = 0;
    setEpisode(data.episode||0); setBest(data.best||0); setEpsilon(data.epsilon||1);
    setHistory(data.history||[]); setQSize(Object.keys(data.qTable).length);
    setSaveStatus("loaded"); setTimeout(()=>setSaveStatus(null),2500);
    stateRef.current = randomState();
    drawCartPole(canvasRef.current, stateRef.current, 0, modeRef.current, null, false, isFullscreenRef.current);
  }, []);

  // ── CLEAR SAVE ──
  const handleClearSave = useCallback(() => {
    clearSave(); setHasSave(false); setSavedAt(null);
    setSaveStatus("cleared"); setTimeout(()=>setSaveStatus(null),2000);
  }, []);

  // ── SHOWCASE MODE ──
  const handleShowcase = useCallback(() => {
    handlePause();
    const data = loadAgent();
    if (data) {
      agentRef.current = { ...createAgent(), qTable: data.qTable, epsilon: 0.0 }; // epsilon=0 → pure greedy
      epRef.current = 0; scoreRef.current = 0; bestRef.current = 0;
      setEpisode(0); setScore(0); setBest(0); setHistory([]);
    } else {
      // Use current trained agent, lock epsilon
      agentRef.current = { ...agentRef.current, epsilon: 0.0 };
    }
    stateRef.current = randomState();
    isShowcaseRef.current = true; modeRef.current = "ai";
    setIsShowcase(true); setMode("ai");
    drawCartPole(canvasRef.current, stateRef.current, 0, "ai", null, true, isFullscreenRef.current);
    // Auto-start
    runningRef.current = true; setIsRunning(true);
    speedRef.current = 1; setSpeed(1);
    animRef.current = requestAnimationFrame(showcaseTick);
  }, []);

  // ── SHOWCASE TICK (no learning, pure exploitation) ──
  const showcaseTick = useCallback(() => {
    if (!runningRef.current) return;
    const state = stateRef.current;
    const action = agentAct(agentRef.current, state, true); // forceGreedy
    const { nextState, done } = cartpoleStep(state, action);
    stateRef.current = nextState;
    scoreRef.current++;
    if (done) {
      if (scoreRef.current > bestRef.current) bestRef.current = scoreRef.current;
      epRef.current++;
      setHistory(p=>[...p.slice(-29), scoreRef.current]);
      setEpisode(epRef.current); setBest(bestRef.current);
      scoreRef.current = 0;
      stateRef.current = randomState();
    }
    setScore(scoreRef.current); setLastAction(action);
    if (stepsRef.current % 6 === 0) {
      const exp = getLiveExplanation(nextState, action, 0, epRef.current, scoreRef.current, "ai", done, true);
      if (exp) setLiveExp(exp);
    }
    stepsRef.current++;
    drawCartPole(canvasRef.current, stateRef.current, scoreRef.current, "ai", null, true, isFullscreenRef.current);
    animRef.current = requestAnimationFrame(showcaseTick);
  }, []);

  // ── AI TICK ──
  const aiTick = useCallback(() => {
    if (!runningRef.current || modeRef.current !== "ai") return;
    const steps = speedRef.current;
    for (let i=0; i<steps; i++) {
      const state = stateRef.current;
      const action = agentAct(agentRef.current, state);
      const { nextState, done } = cartpoleStep(state, action);
      const reward = shapeReward(nextState, done);
      agentRef.current = agentLearn(agentRef.current, state, action, reward, nextState, done);
      stateRef.current = nextState;
      scoreRef.current++; stepsRef.current++;
      if (done) {
        if (scoreRef.current > bestRef.current) bestRef.current = scoreRef.current;
        epRef.current++;
        setHistory(p=>[...p.slice(-29), scoreRef.current]);
        setEpisode(epRef.current); setBest(bestRef.current);
        scoreRef.current = 0; stateRef.current = randomState();
      }
      if (i===steps-1) {
        setLastAction(action); setScore(scoreRef.current); setEpsilon(agentRef.current.epsilon);
        if (stepsRef.current%8===0) setQSize(Object.keys(agentRef.current.qTable).length);
        const exp = getLiveExplanation(nextState, action, agentRef.current.epsilon, epRef.current, scoreRef.current, "ai", done, false);
        if (exp) setLiveExp(exp);
        drawCartPole(canvasRef.current, stateRef.current, scoreRef.current, "ai", null, false, isFullscreenRef.current);
      }
    }
    animRef.current = requestAnimationFrame(aiTick);
  }, []);

  // ── MANUAL STEP ──
  const manualStep = useCallback((action) => {
    if (!runningRef.current || modeRef.current !== "manual") return;
    const state = stateRef.current;
    const { nextState, done } = cartpoleStep(state, action);
    agentRef.current = imitationLearn(agentRef.current, state, action, nextState, done);
    imitRef.current++; setImitCount(imitRef.current);
    stateRef.current = nextState; scoreRef.current++;
    humanActionDisplayRef.current = action;
    setLastAction(action); setScore(scoreRef.current);
    setQSize(Object.keys(agentRef.current.qTable).length);
    const exp = getLiveExplanation(nextState, action, agentRef.current.epsilon, epRef.current, scoreRef.current, "manual", done, false);
    if (exp) setLiveExp(exp);
    drawCartPole(canvasRef.current, nextState, scoreRef.current, "manual", action, false, isFullscreenRef.current);
    if (done) {
      if (scoreRef.current > bestRef.current) bestRef.current = scoreRef.current;
      epRef.current++; setHistory(p=>[...p.slice(-29), scoreRef.current]);
      setEpisode(epRef.current); setBest(bestRef.current);
      scoreRef.current = 0; stateRef.current = randomState();
    }
    setTimeout(()=>{ humanActionDisplayRef.current=null; },200);
  }, []);

  useEffect(() => {
    const down = (e) => {
      if (!runningRef.current || modeRef.current !== "manual") return;
      if (e.key==="ArrowLeft"||e.key==="a"||e.key==="A") manualStep(0);
      if (e.key==="ArrowRight"||e.key==="d"||e.key==="D") manualStep(1);
    };
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
  }, [manualStep]);

  const startManualLoop = useCallback(() => {
    if (manualIntervalRef.current) clearInterval(manualIntervalRef.current);
    manualIntervalRef.current = setInterval(() => {
      if (!runningRef.current || modeRef.current !== "manual") return;
      drawCartPole(canvasRef.current, stateRef.current, scoreRef.current, "manual", humanActionDisplayRef.current, false, isFullscreenRef.current);
    }, 50);
  }, []);

  const handleStart = () => {
    if (runningRef.current) return;
    runningRef.current = true; setIsRunning(true);
    if (isShowcaseRef.current) animRef.current = requestAnimationFrame(showcaseTick);
    else if (modeRef.current==="ai") animRef.current = requestAnimationFrame(aiTick);
    else startManualLoop();
  };

  const handlePause = useCallback(() => {
    runningRef.current = false; setIsRunning(false);
    if (animRef.current) cancelAnimationFrame(animRef.current);
    if (manualIntervalRef.current) clearInterval(manualIntervalRef.current);
  }, []);

  const handleReset = () => {
    handlePause();
    isShowcaseRef.current = false; setIsShowcase(false);
    agentRef.current = createAgent(); stateRef.current = randomState();
    epRef.current=0; scoreRef.current=0; bestRef.current=0; stepsRef.current=0; imitRef.current=0;
    setEpisode(0); setScore(0); setBest(0); setEpsilon(1.0);
    setHistory([]); setLastAction(null); setLiveExp(null); setQSize(0); setImitCount(0);
    drawCartPole(canvasRef.current, stateRef.current, 0, modeRef.current, null, false, isFullscreenRef.current);
  };

  const switchMode = (m) => {
    if (isShowcaseRef.current) return;
    handlePause(); modeRef.current=m; setMode(m);
    drawCartPole(canvasRef.current, stateRef.current, scoreRef.current, m, null, false, isFullscreenRef.current);
  };

  const setSpeedVal = (v) => { speedRef.current=v; setSpeed(v); };

  // ── FULLSCREEN ──
  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen?.().then(()=>{ isFullscreenRef.current=true; setIsFullscreen(true); }).catch(()=>{});
    } else {
      document.exitFullscreen?.().then(()=>{ isFullscreenRef.current=false; setIsFullscreen(false); }).catch(()=>{});
    }
  }, []);

  useEffect(() => {
    const onFsChange = () => {
      const fs = !!document.fullscreenElement;
      isFullscreenRef.current=fs; setIsFullscreen(fs);
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  // resize canvas on fullscreen
  useEffect(() => {
    if (!canvasRef.current) return;
    if (isFullscreen) {
      canvasRef.current.width = 900; canvasRef.current.height = 420;
    } else {
      canvasRef.current.width = 580; canvasRef.current.height = 260;
    }
    drawCartPole(canvasRef.current, stateRef.current, scoreRef.current, modeRef.current, null, isShowcaseRef.current, isFullscreen);
  }, [isFullscreen]);

  useEffect(() => {
    drawCartPole(canvasRef.current, stateRef.current, 0, "ai", null, false, false);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      if (manualIntervalRef.current) clearInterval(manualIntervalRef.current);
    };
  }, []);

  const concept = CONCEPTS[activeConcept];
  const maxBar = Math.max(...history, 1);

  // Fullscreen overlay layout
  if (isFullscreen) {
    return (
      <div ref={containerRef} style={{ width:"100vw", height:"100vh", background:"#040414", display:"flex", flexDirection:"column", fontFamily:"'Courier New',monospace", color:"#e2e8f0", padding:"16px", boxSizing:"border-box" }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"12px" }}>
          <div>
            <div style={{ fontSize:"9px", letterSpacing:"6px", color:"#00d4ff" }}>REINFORCEMENT LEARNING LAB</div>
            <h1 style={{ margin:0, fontSize:"22px", fontWeight:"900", background:"linear-gradient(90deg,#00d4ff,#a78bfa,#f87171)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>
              CARTPOLE {isShowcase?"— 🏆 SHOWCASE":"— Q-LEARNING"}
            </h1>
          </div>
          <div style={{ display:"flex", gap:"8px", alignItems:"center" }}>
            {/* Stats row */}
            {[{l:"EP",v:episode,c:"#a78bfa"},{l:"SCORE",v:score,c:"#34d399"},{l:"BEST",v:best,c:"#fbbf24"},{l:"ε",v:epsilon.toFixed(2),c:"#f87171"}].map(s=>(
              <div key={s.l} style={{ padding:"6px 12px", borderRadius:"8px", background:"rgba(255,255,255,0.04)", border:`1px solid ${s.c}30`, textAlign:"center" }}>
                <div style={{ fontSize:"8px", color:s.c, letterSpacing:"2px" }}>{s.l}</div>
                <div style={{ fontSize:"18px", fontWeight:"900" }}>{s.v}</div>
              </div>
            ))}
            <button onClick={toggleFullscreen} style={{ padding:"8px 16px", border:"1px solid rgba(255,255,255,0.2)", background:"rgba(255,255,255,0.05)", color:"#94a3b8", borderRadius:"8px", cursor:"pointer", fontFamily:"monospace", fontSize:"12px" }}>⛶ EXIT</button>
          </div>
        </div>
        {/* Canvas fullscreen */}
        <div style={{ flex:1, borderRadius:"12px", overflow:"hidden", border:`1px solid ${isShowcase?"rgba(251,191,36,0.3)":"rgba(0,212,255,0.2)"}`, boxShadow:`0 0 60px ${isShowcase?"rgba(251,191,36,0.12)":"rgba(0,212,255,0.1)"}`, marginBottom:"12px" }}>
          <canvas ref={canvasRef} width={900} height={420} style={{ display:"block", width:"100%", height:"100%" }} />
        </div>
        {/* Controls strip */}
        <div style={{ display:"flex", gap:"8px", alignItems:"center", flexWrap:"wrap" }}>
          <button onClick={isRunning?handlePause:handleStart} style={{ padding:"9px 20px", border:`1px solid ${isRunning?"#f87171":"#34d399"}`, background:isRunning?"rgba(248,113,113,0.15)":"rgba(52,211,153,0.15)", color:isRunning?"#f87171":"#34d399", borderRadius:"8px", cursor:"pointer", fontFamily:"monospace", fontSize:"13px", fontWeight:"700" }}>{isRunning?"⏸ PAUSE":"▶ START"}</button>
          {!isShowcase && <><button onClick={handleSave} style={{ padding:"9px 16px", border:"1px solid #00d4ff", background:"rgba(0,212,255,0.1)", color:"#00d4ff", borderRadius:"8px", cursor:"pointer", fontFamily:"monospace", fontSize:"12px", fontWeight:"700" }}>💾 SAVE</button>
          {hasSave && <button onClick={handleShowcase} style={{ padding:"9px 16px", border:"1px solid #fbbf24", background:"rgba(251,191,36,0.12)", color:"#fbbf24", borderRadius:"8px", cursor:"pointer", fontFamily:"monospace", fontSize:"12px", fontWeight:"700" }}>🏆 SHOWCASE</button>}</>}
          {isShowcase && <button onClick={handleReset} style={{ padding:"9px 16px", border:"1px solid #f87171", background:"rgba(248,113,113,0.1)", color:"#f87171", borderRadius:"8px", cursor:"pointer", fontFamily:"monospace", fontSize:"12px" }}>↩ BACK TO TRAINING</button>}
          {/* Live exp */}
          {liveExp && <div style={{ flex:1, padding:"8px 14px", borderRadius:"8px", background:`${liveExp.color}10`, border:`1px solid ${liveExp.color}30`, fontSize:"11px", color:"#94a3b8", display:"flex", gap:"10px", alignItems:"center" }}>
            <span style={{ fontSize:"16px" }}>{liveExp.icon}</span>
            <span><strong style={{ color:liveExp.color }}>{liveExp.title}</strong> — {liveExp.text.slice(0,120)}...</span>
          </div>}
        </div>
      </div>
    );
  }

  // ── NORMAL LAYOUT ──
  return (
    <div ref={containerRef} style={{ minHeight:"100vh", background:"#060613", fontFamily:"'Courier New',monospace", color:"#e2e8f0", padding:"14px", boxSizing:"border-box" }}>

      {/* Header */}
      <div style={{ textAlign:"center", marginBottom:"12px" }}>
        <div style={{ fontSize:"10px", letterSpacing:"6px", color:"#00d4ff", marginBottom:"2px" }}>REINFORCEMENT LEARNING LAB</div>
        <h1 style={{ margin:0, fontSize:"22px", fontWeight:"900", background:"linear-gradient(90deg,#00d4ff,#a78bfa,#f87171)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", letterSpacing:"2px" }}>
          CARTPOLE — Q-LEARNING + IMITATION {isShowcase&&"🏆 SHOWCASE"}
        </h1>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 310px", gap:"14px", maxWidth:"1120px", margin:"0 auto" }}>

        {/* LEFT */}
        <div>
          {/* Canvas */}
          <div style={{ borderRadius:"12px", overflow:"hidden", border:`1px solid ${isShowcase?"rgba(251,191,36,0.35)":"rgba(0,212,255,0.2)"}`, boxShadow:`0 0 40px ${isShowcase?"rgba(251,191,36,0.1)":"rgba(0,212,255,0.08)"}`, marginBottom:"10px", position:"relative" }}>
            <canvas ref={canvasRef} width={580} height={260} style={{ display:"block", width:"100%", height:"auto" }} />
            {/* Fullscreen button overlay */}
            <button onClick={toggleFullscreen} title="Fullscreen" style={{ position:"absolute", top:"8px", right:"8px", padding:"5px 10px", background:"rgba(0,0,0,0.5)", border:"1px solid rgba(255,255,255,0.15)", color:"#94a3b8", borderRadius:"6px", cursor:"pointer", fontFamily:"monospace", fontSize:"12px", backdropFilter:"blur(4px)" }}>⛶</button>
          </div>

          {/* Save/Load/Showcase bar */}
          <div style={{ display:"flex", gap:"7px", marginBottom:"10px", padding:"10px", borderRadius:"10px", background:"rgba(0,212,255,0.04)", border:"1px solid rgba(0,212,255,0.12)", flexWrap:"wrap", alignItems:"center" }}>
            <span style={{ fontSize:"9px", color:"#00d4ff", letterSpacing:"2px", marginRight:"2px" }}>💾 BRAIN</span>
            <button onClick={handleSave} disabled={isShowcase} style={{ padding:"6px 14px", border:"1px solid #00d4ff", background:"rgba(0,212,255,0.1)", color:"#00d4ff", borderRadius:"7px", cursor:"pointer", fontFamily:"monospace", fontSize:"11px", fontWeight:"700", opacity:isShowcase?0.4:1 }}>SAVE</button>
            <button onClick={handleLoad} disabled={!hasSave} style={{ padding:"6px 14px", border:`1px solid ${hasSave?"#a78bfa":"rgba(255,255,255,0.1)"}`, background:hasSave?"rgba(167,139,250,0.1)":"rgba(255,255,255,0.02)", color:hasSave?"#a78bfa":"#334155", borderRadius:"7px", cursor:hasSave?"pointer":"not-allowed", fontFamily:"monospace", fontSize:"11px", fontWeight:"700" }}>LOAD</button>
            <button onClick={handleShowcase} disabled={!hasSave&&qSize<10} style={{ padding:"6px 14px", border:`1px solid ${(hasSave||qSize>=10)?"#fbbf24":"rgba(255,255,255,0.1)"}`, background:(hasSave||qSize>=10)?"rgba(251,191,36,0.12)":"rgba(255,255,255,0.02)", color:(hasSave||qSize>=10)?"#fbbf24":"#334155", borderRadius:"7px", cursor:(hasSave||qSize>=10)?"pointer":"not-allowed", fontFamily:"monospace", fontSize:"11px", fontWeight:"700" }}>🏆 SHOWCASE</button>
            {isShowcase && <button onClick={handleReset} style={{ padding:"6px 12px", border:"1px solid #f87171", background:"rgba(248,113,113,0.1)", color:"#f87171", borderRadius:"7px", cursor:"pointer", fontFamily:"monospace", fontSize:"11px" }}>↩ TRAIN</button>}
            {hasSave && <button onClick={handleClearSave} style={{ padding:"6px 10px", border:"1px solid rgba(255,255,255,0.1)", background:"rgba(255,255,255,0.02)", color:"#475569", borderRadius:"7px", cursor:"pointer", fontFamily:"monospace", fontSize:"10px" }}>🗑</button>}
            <div style={{ marginLeft:"auto", fontSize:"10px" }}>
              {saveStatus==="saved" && <span style={{ color:"#34d399" }}>✓ Saved at {savedAt}</span>}
              {saveStatus==="loaded" && <span style={{ color:"#a78bfa" }}>✓ Brain loaded!</span>}
              {saveStatus==="cleared" && <span style={{ color:"#f87171" }}>✗ Save cleared</span>}
              {!saveStatus && hasSave && <span style={{ color:"#475569" }}>Save: {savedAt}</span>}
              {!saveStatus && !hasSave && <span style={{ color:"#1e293b" }}>No save yet</span>}
            </div>
          </div>

          {/* Mode + Controls */}
          {!isShowcase && <div style={{ display:"flex", gap:"8px", marginBottom:"10px" }}>
            {[["ai","🤖 AI TRAINS","#a78bfa"],["manual","👨‍🏫 TEACH MODE","#34d399"]].map(([m,label,col])=>(
              <button key={m} onClick={()=>switchMode(m)} style={{ flex:1, padding:"9px", border:`1px solid ${mode===m?col:"rgba(255,255,255,0.08)"}`, background:mode===m?`${col}18`:"rgba(255,255,255,0.02)", color:mode===m?col:"#475569", borderRadius:"8px", cursor:"pointer", fontFamily:"monospace", fontSize:"12px", fontWeight:"700", letterSpacing:"1px" }}>{label}</button>
            ))}
          </div>}

          <div style={{ display:"flex", gap:"8px", marginBottom:"10px", flexWrap:"wrap", alignItems:"center" }}>
            <button onClick={isRunning?handlePause:handleStart} style={{ padding:"9px 20px", border:`1px solid ${isRunning?"#f87171":"#34d399"}`, background:isRunning?"rgba(248,113,113,0.12)":"rgba(52,211,153,0.12)", color:isRunning?"#f87171":"#34d399", borderRadius:"8px", cursor:"pointer", fontFamily:"monospace", fontSize:"13px", fontWeight:"700" }}>{isRunning?"⏸ PAUSE":"▶ START"}</button>
            {!isShowcase && <button onClick={handleReset} style={{ padding:"9px 16px", border:"1px solid #fbbf24", background:"rgba(251,191,36,0.1)", color:"#fbbf24", borderRadius:"8px", cursor:"pointer", fontFamily:"monospace", fontSize:"13px", fontWeight:"700" }}>↺ RESET</button>}
            {!isShowcase && <div style={{ display:"flex", gap:"5px", alignItems:"center" }}>
              <span style={{ fontSize:"10px", color:"#64748b" }}>SPEED</span>
              {[1,3,8,20].map(v=>(
                <button key={v} onClick={()=>setSpeedVal(v)} style={{ padding:"5px 9px", border:`1px solid ${speed===v?"#00d4ff":"rgba(255,255,255,0.08)"}`, background:speed===v?"rgba(0,212,255,0.15)":"rgba(255,255,255,0.02)", color:speed===v?"#00d4ff":"#475569", borderRadius:"6px", cursor:"pointer", fontFamily:"monospace", fontSize:"11px", fontWeight:"700" }}>{v}x</button>
              ))}
            </div>}
            <div style={{ marginLeft:"auto", padding:"7px 12px", border:"1px solid rgba(255,255,255,0.08)", borderRadius:"8px", background:"rgba(255,255,255,0.02)", fontSize:"12px", display:"flex", gap:"8px", alignItems:"center" }}>
              <span style={{ color:"#64748b" }}>ACTION</span>
              <span style={{ color:lastAction===0?"#00d4ff":lastAction===1?"#f87171":"#333", fontWeight:"800", fontSize:"14px" }}>
                {lastAction===null?"—":lastAction===0?"◀ LEFT":"▶ RIGHT"}
              </span>
            </div>
          </div>

          {/* Manual controls */}
          {mode==="manual"&&!isShowcase&&(
            <div style={{ marginBottom:"10px", padding:"12px", borderRadius:"10px", background:"rgba(52,211,153,0.06)", border:"1px solid rgba(52,211,153,0.25)" }}>
              <div style={{ fontSize:"10px", color:"#34d399", letterSpacing:"2px", marginBottom:"8px" }}>👨‍🏫 TEACH THE AGENT — Keys or tap buttons</div>
              <div style={{ display:"flex", gap:"10px", marginBottom:"8px" }}>
                {[["◀ LEFT (A / ←)",0,"#00d4ff"],["▶ RIGHT (D / →)",1,"#f87171"]].map(([label,act,col])=>(
                  <button key={act} onMouseDown={()=>manualStep(act)} onTouchStart={(e)=>{e.preventDefault();manualStep(act);}} style={{ flex:1, padding:"14px", border:`1px solid ${col}`, background:`${col}18`, color:col, borderRadius:"10px", cursor:"pointer", fontFamily:"monospace", fontSize:"14px", fontWeight:"900", userSelect:"none", WebkitUserSelect:"none" }}>{label}</button>
                ))}
              </div>
              <div style={{ fontSize:"10px", color:"#475569" }}>🧠 Demos: <strong style={{color:"#34d399"}}>{imitCount}</strong> — Like AlphaGo: your expert moves directly seed the Q-table (α=0.8)</div>
            </div>
          )}

          {/* Showcase banner */}
          {isShowcase&&(
            <div style={{ marginBottom:"10px", padding:"12px", borderRadius:"10px", background:"rgba(251,191,36,0.08)", border:"1px solid rgba(251,191,36,0.35)", textAlign:"center" }}>
              <div style={{ fontSize:"12px", color:"#fbbf24", fontWeight:"800", letterSpacing:"2px", marginBottom:"4px" }}>🏆 SHOWCASE MODE — TRAINED BRAIN RUNNING FREE</div>
              <div style={{ fontSize:"10px", color:"#94a3b8" }}>ε = 0.000 — pure exploitation, no randomness, no learning. Just the skills it built during training.</div>
            </div>
          )}

          {/* Stats */}
          <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:"7px", marginBottom:"10px" }}>
            {[{label:"EPISODE",value:episode,color:"#a78bfa"},{label:"SCORE",value:score,color:"#34d399"},{label:"BEST",value:best,color:"#fbbf24"},{label:isShowcase?"ε (LOCKED)":"ε EXPLORE",value:isShowcase?"0.000":epsilon.toFixed(3),color:isShowcase?"#fbbf24":"#f87171"}].map(s=>(
              <div key={s.label} style={{ padding:"9px", borderRadius:"8px", background:"rgba(255,255,255,0.03)", border:`1px solid ${s.color}25`, textAlign:"center" }}>
                <div style={{ fontSize:"8px", color:s.color, letterSpacing:"2px", marginBottom:"3px" }}>{s.label}</div>
                <div style={{ fontSize:"20px", fontWeight:"900", color:"#fff" }}>{s.value}</div>
              </div>
            ))}
          </div>

          {/* Live explanation */}
          <div style={{ padding:"14px", borderRadius:"10px", background:liveExp?`${liveExp.color}08`:"rgba(255,255,255,0.02)", border:`1px solid ${liveExp?liveExp.color+"35":"rgba(255,255,255,0.06)"}`, marginBottom:"10px", minHeight:"88px", transition:"border-color 0.4s" }}>
            {!liveExp?<div style={{ color:"#334155", fontSize:"12px", marginTop:"10px" }}>▶ Start to see live explanations...</div>:(
              <>
                <div style={{ display:"flex", gap:"8px", alignItems:"center", marginBottom:"5px" }}>
                  <span style={{ fontSize:"18px" }}>{liveExp.icon}</span>
                  <div><div style={{ fontSize:"9px", color:liveExp.color, letterSpacing:"2px" }}>{liveExp.concept}</div><div style={{ fontSize:"13px", fontWeight:"800", color:"#fff" }}>{liveExp.title}</div></div>
                </div>
                <div style={{ fontSize:"11px", color:"#94a3b8", lineHeight:"1.7" }}>{liveExp.text}</div>
              </>
            )}
          </div>

          {/* Score history */}
          <div style={{ padding:"10px", borderRadius:"10px", background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.05)" }}>
            <div style={{ fontSize:"9px", color:"#64748b", letterSpacing:"2px", marginBottom:"7px" }}>EPISODE SCORE HISTORY (LAST 30)</div>
            <div style={{ display:"flex", alignItems:"flex-end", gap:"3px", height:"46px" }}>
              {history.length===0?<div style={{ color:"#1e293b", fontSize:"11px" }}>Train to see progress...</div>
                :history.map((s,i)=>{
                  const h=Math.max(3,(s/maxBar)*43), isLast=i===history.length-1;
                  const hue=Math.min(120,(s/Math.max(best,1))*120);
                  return <div key={i} style={{ flex:1, height:`${h}px`, background:isLast?`hsl(${hue},100%,60%)`:`hsla(${hue},80%,55%,0.55)`, borderRadius:"2px 2px 0 0", boxShadow:isLast?`0 0 8px hsl(${hue},100%,60%)`:"none" }} />;
                })}
            </div>
          </div>
        </div>

        {/* RIGHT */}
        <div>
          <div style={{ fontSize:"9px", color:"#64748b", letterSpacing:"3px", marginBottom:"9px" }}>📚 RL CONCEPTS</div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"5px", marginBottom:"10px" }}>
            {CONCEPTS.map((c,i)=>(
              <button key={c.id} onClick={()=>setActiveConcept(i)} style={{ padding:"7px 5px", borderRadius:"7px", cursor:"pointer", border:`1px solid ${i===activeConcept?c.color:"rgba(255,255,255,0.05)"}`, background:i===activeConcept?`${c.color}15`:"rgba(255,255,255,0.02)", color:i===activeConcept?c.color:"#475569", fontFamily:"monospace", fontSize:"8px", fontWeight:"700", textAlign:"center", lineHeight:"1.4" }}>
                <div style={{ fontSize:"13px", marginBottom:"2px" }}>{c.icon}</div>
                <div>{c.tag.split(" ")[0]}</div>
              </button>
            ))}
          </div>
          <div style={{ padding:"14px", borderRadius:"10px", background:`${concept.color}08`, border:`1px solid ${concept.color}38`, marginBottom:"10px" }}>
            <div style={{ fontSize:"8px", color:concept.color, letterSpacing:"3px", marginBottom:"5px" }}>{concept.tag}</div>
            <div style={{ fontSize:"14px", fontWeight:"800", color:"#fff", marginBottom:"7px" }}>{concept.icon} {concept.title}</div>
            <div style={{ fontSize:"11px", color:"#94a3b8", lineHeight:"1.7", marginBottom:"10px" }}>{concept.desc}</div>
            <div style={{ padding:"9px 12px", borderRadius:"7px", background:"rgba(0,0,0,0.45)", border:`1px solid ${concept.color}28`, fontFamily:"monospace", fontSize:"11px", color:concept.color, textAlign:"center" }}>{concept.formula}</div>
          </div>
          <div style={{ padding:"11px", borderRadius:"9px", background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.05)", marginBottom:"10px" }}>
            <div style={{ fontSize:"9px", color:"#64748b", letterSpacing:"2px", marginBottom:"8px" }}>RL LOOP</div>
            {[["Environment","#00d4ff","↓ state, reward"],["Agent (Policy)","#a78bfa","↓ action"],["Q-Table Update","#fbbf24","↓ learn"],["Next State","#34d399","↺ loop"]].map(([label,col,arrow],i)=>(
              <div key={i}>
                <div style={{ padding:"5px 9px", borderRadius:"5px", background:`${col}10`, border:`1px solid ${col}28`, fontSize:"10px", color:col, fontWeight:"700" }}>{label}</div>
                {i<3&&<div style={{ fontSize:"9px", color:"#334155", paddingLeft:"10px", margin:"2px 0" }}>{arrow}</div>}
              </div>
            ))}
          </div>
          <div style={{ padding:"10px", borderRadius:"8px", background:"rgba(251,191,36,0.05)", border:"1px solid rgba(251,191,36,0.18)", fontSize:"10px", marginBottom:"8px" }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:"5px" }}><span style={{color:"#fbbf24"}}>Q-Table entries</span><strong style={{color:"#fff"}}>{qSize}</strong></div>
            <div style={{ display:"flex", justifyContent:"space-between" }}><span style={{color:"#e879f9"}}>Imitation demos</span><strong style={{color:"#fff"}}>{imitCount}</strong></div>
          </div>
          <div style={{ padding:"11px", borderRadius:"8px", background:"rgba(232,121,249,0.07)", border:"1px solid rgba(232,121,249,0.25)", fontSize:"10px" }}>
            <div style={{ color:"#e879f9", letterSpacing:"2px", fontWeight:"700", marginBottom:"6px" }}>🏆 ALPHAGO CONNECTION</div>
            <div style={{ color:"#94a3b8", lineHeight:"1.6" }}>AlphaGo used <strong style={{color:"#e879f9"}}>supervised learning</strong> on 30M grandmaster moves first, then RL self-play.<br/><br/>Switch to <strong style={{color:"#34d399"}}>👨‍🏫 TEACH MODE</strong> and play well — your demos seed the Q-table just like those expert games seeded AlphaGo's policy network.</div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth:"1120px", margin:"12px auto 0", padding:"9px 14px", borderRadius:"8px", background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.04)", fontSize:"10px", color:"#334155", textAlign:"center" }}>
        💡 <strong style={{color:"#475569"}}>Workflow:</strong> Train (or Teach) → <strong style={{color:"#00d4ff"}}>💾 SAVE</strong> brain → come back anytime → <strong style={{color:"#a78bfa"}}>LOAD</strong> → <strong style={{color:"#fbbf24"}}>🏆 SHOWCASE</strong> to watch it perform freely. Hit <strong style={{color:"#94a3b8"}}>⛶</strong> for fullscreen.
      </div>
    </div>
  );
}
