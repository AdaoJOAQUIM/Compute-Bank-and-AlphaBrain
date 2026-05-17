/**
 * AlphaBrain v10.0 — PWA App Logic
 * Axiom L: crystallize NL → SparseVec
 * Axiom F: SGR retrieve from W
 * Axiom Q: eval() retrieved S-expressions (homoiconic W)
 * Axiom K: pheromone tick loop (adiabatic invariant)
 */

'use strict';

const {
  SparseVec, SparseW, SeedLexicon, SExpr, parseSexpr,
  Interpreter, Env, CrystallizationStore,
  AlphaNode, AlphaNetwork, sgrRetrieve
} = window.AlphaBrainRuntime;

// ── Runtime State ─────────────────────────────────────────────────────────────
const GENESIS = Array.from({length: 32}, (_, i) => i); // deterministic genesis
const net = AlphaNetwork.threeNode();
const store = new CrystallizationStore(GENESIS);
const interp = new Interpreter(500);
const globalEnv = new Env();
const mainW = net.getNode(0).w;

let tickCount = 0;
let evalsCount = 0;
let lastFuelUsed = 0;
let sexprsStored = 0;
let behaviorEvents = 0;

// Track dwell time for Axiom N
window._dwellMs = 0;
window._contextTokens = '';
let _lastInteraction = Date.now();

// ── Tick Loop (Axiom K — adiabatic invariant) ─────────────────────────────────
function tick() {
  net.pheromeoneTick(0.1);
  tickCount++;

  // Update Φ₁ (CPU pressure) from ops
  const node0 = net.getNode(0);
  if (node0) {
    document.getElementById('phi1-metric').textContent = node0.pheromone.phi1.toFixed(3);
    document.getElementById('phi2-metric').textContent = node0.pheromone.phi2.toFixed(3);
    document.getElementById('phi5-metric').textContent = node0.pheromone.phi5.toFixed(3);
    document.getElementById('adiabatic-ok').textContent = node0.pheromone.adiabaticOk() ? 'YES' : 'NO';
    document.getElementById('phi1-display').textContent = `Φ₁: ${node0.pheromone.phi1.toFixed(3)}`;
    document.getElementById('phi2-display').textContent = `Φ₂: ${node0.pheromone.phi2.toFixed(3)}`;
  }
  document.getElementById('tick-counter').textContent = `tick: ${tickCount}`;
}

// Adiabatic: 100ms tick = 10 Hz, well within CPU budget
setInterval(tick, 100);

// ── NL Input Handler ──────────────────────────────────────────────────────────
window.handleInput = function() {
  const input = document.getElementById('nl-input');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  input.style.height = 'auto';
  addMessage(text, 'user');

  // Axiom L: crystallize NL into sparse vector
  const tokens = text.toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(t => t.length > 1 && !['the','a','an','is','are','was','to','of','in','for'].includes(t));

  if (tokens.length === 0) {
    addMessage('(no meaningful tokens found — please describe what you want)', 'system');
    return;
  }

  window._contextTokens = tokens.join(' ');
  window._dwellMs = Date.now() - _lastInteraction;
  _lastInteraction = Date.now();

  // Step 1: Crystallize
  const vector = store.observe(tokens);
  updateMetrics();

  // Step 2: Axiom H — SGR retrieve before compute
  const node = net.getNode(0);
  node.storePattern(vector, 1);
  const noisy = vector.noisy(0.05, Math.random);
  const result = sgrRetrieve(noisy, 0, net.nodes, 5);

  document.getElementById('last-hops').textContent = result.hops;
  document.getElementById('last-fidelity').textContent = result.fidelity.toFixed(3);

  // Step 3: Check if retrieved pattern is an S-expression
  const responseLines = [];
  responseLines.push(`Crystallized: [${tokens.join(', ')}]`);
  responseLines.push(`SGR: ${result.hops} hops, fidelity=${result.fidelity.toFixed(3)}`);

  // Step 4: Axiom Q — try to match known S-expression rules
  const rule = matchSexprRule(tokens, text);
  let sexprResult = null;
  if (rule) {
    try {
      const fuel = { v: 500 };
      const expr = parseSexpr(rule.code);
      const evalResult = interp.eval(expr, globalEnv, mainW, fuel);
      lastFuelUsed = 500 - fuel.v;
      evalsCount++;
      sexprsStored++;
      sexprResult = { code: rule.code, result: evalResult.toString(), fuel: lastFuelUsed, label: rule.label };
      responseLines.push(`Axiom Q: eval() rule — "${rule.label}"`);
    } catch (e) {
      responseLines.push(`Axiom Q: eval error — ${e.message}`);
    }
  }

  // Step 5: Generate natural language response
  const nlResponse = generateResponse(tokens, text, result, store.intents);
  responseLines.push('');
  responseLines.push(nlResponse);

  addMessage(responseLines.join('\n'), 'system', sexprResult, store.intentAccuracy > 0.5);
  updateIntentList();
  updateMetrics();
};

// ── S-expression Rule Matcher (Axiom Q) ───────────────────────────────────────
const SEXPR_RULES = [
  {
    keywords: ['remember', 'save', 'store', 'memorize'],
    label: 'crystallize-remember',
    code: '(define remember-rule (lambda (ctx) (crystallize ctx)))'
  },
  {
    keywords: ['when', 'dwell', 'time', 'long', 'minutes', 'hours'],
    label: 'dwell-trigger',
    code: '(when (> (dwell-time) 300000) (crystallize (context-tokens)))'
  },
  {
    keywords: ['find', 'search', 'locate', 'look'],
    label: 'find-rule',
    code: '(define find-rule (lambda (query) (list "searching" query)))'
  },
  {
    keywords: ['morning', 'daily', 'schedule', 'routine'],
    label: 'morning-routine',
    code: '(define morning (lambda () (list "calendar" "emails" "notes")))'
  },
  {
    keywords: ['deploy', 'production', 'release', 'publish'],
    label: 'deploy-intent',
    code: '(define deploy-rule (lambda () (crystallize (list "deploy" "production"))))'
  },
];

function matchSexprRule(tokens, text) {
  const lower = text.toLowerCase();
  for (const rule of SEXPR_RULES) {
    if (rule.keywords.some(kw => lower.includes(kw) || tokens.includes(kw))) {
      return rule;
    }
  }
  return null;
}

// ── Response Generator ────────────────────────────────────────────────────────
function generateResponse(tokens, text, sgrResult, intents) {
  const confirmed = intents.filter(i => i.confirmed);
  const total = intents.length;

  if (text.match(/\?$/) || text.toLowerCase().startsWith('what') || text.toLowerCase().startsWith('how')) {
    return `I've crystallized your query into W. ${confirmed.length}/${total} intents confirmed so far. ` +
      `SGR retrieved with ${sgrResult.fidelity > 0.7 ? 'high' : 'moderate'} fidelity — ` +
      `this is ${sgrResult.fidelity > 0.7 ? 'a known pattern' : 'a new pattern being learned'}.`;
  }

  if (tokens.some(t => ['remember', 'save', 'store'].includes(t))) {
    return `Crystallizing intent into W. This pattern will strengthen with repetition. ` +
      `After ${Math.max(0, 5 - (intents[intents.length-1]?.count || 0))} more similar observations, this intent will be confirmed.`;
  }

  if (confirmed.length > 0) {
    return `Pattern recognized (fidelity: ${sgrResult.fidelity.toFixed(2)}). ` +
      `${confirmed.length} confirmed intent${confirmed.length > 1 ? 's' : ''} in W. ` +
      `Language is crystallizing from your usage — no pre-trained models.`;
  }

  return `New pattern stored in W. I'm learning your vocabulary from usage. ` +
    `${total} intent${total !== 1 ? 's' : ''} observed so far, ` +
    `${confirmed.length} confirmed. Language crystallizes after ~5 repetitions.`;
}

// ── UI Helpers ────────────────────────────────────────────────────────────────
function addMessage(text, role, sexprResult = null, crystallized = false) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `msg ${role}${crystallized ? ' crystallized' : ''}`;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;

  if (sexprResult) {
    const sx = document.createElement('div');
    sx.className = 'sexpr';
    const lbl = document.createElement('div');
    lbl.className = 'sexpr-label';
    lbl.textContent = `S-expr · ${sexprResult.label} · fuel: ${sexprResult.fuel}`;
    const code = document.createElement('div');
    code.textContent = sexprResult.code;
    const res = document.createElement('div');
    res.style.cssText = 'color:#50b868;margin-top:4px;font-size:0.6rem';
    res.textContent = `→ ${sexprResult.result}`;
    sx.appendChild(lbl); sx.appendChild(code); sx.appendChild(res);
    bubble.appendChild(sx);
  }

  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = `${role} · tick ${tickCount}`;
  if (role === 'system') meta.textContent += ` · W:${mainW.entryCount} entries`;

  div.appendChild(bubble);
  div.appendChild(meta);
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function updateMetrics() {
  const node0 = net.getNode(0);
  if (!node0) return;
  document.getElementById('patterns-count').textContent = node0.patternsStored;
  document.getElementById('w-entries').textContent = mainW.entryCount;
  document.getElementById('sexprs-count').textContent = sexprsStored;
  document.getElementById('evals-count').textContent = evalsCount;
  document.getElementById('last-fuel').textContent = lastFuelUsed > 0 ? lastFuelUsed : '—';
}

function updateIntentList() {
  const list = document.getElementById('intent-list');
  if (store.intents.length === 0) return;
  list.innerHTML = '';
  for (const intent of store.intents) {
    const div = document.createElement('div');
    div.className = `intent-item${intent.confirmed ? ' confirmed' : ''}`;
    div.textContent = `[${intent.tokens.slice(0, 3).join(', ')}] ×${intent.count}`;
    list.appendChild(div);
  }
}

// ── Textarea auto-resize ───────────────────────────────────────────────────────
const textarea = document.getElementById('nl-input');
textarea.addEventListener('input', () => {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(120, textarea.scrollHeight) + 'px';
});
textarea.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleInput(); }
});

// ── Service Worker registration ───────────────────────────────────────────────
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js')
    .then(reg => {
      document.getElementById('sw-status').textContent = 'Service Worker: active';
      document.getElementById('sw-status').style.color = 'var(--grn)';
    })
    .catch(() => {
      document.getElementById('sw-status').textContent = 'Service Worker: unavailable (file://)';
    });
} else {
  document.getElementById('sw-status').textContent = 'Service Worker: not supported';
}

// ── Behavioral telemetry (Axiom N) ────────────────────────────────────────────
// Passive click/dwell/scroll crystallization
let _pageLoadTime = Date.now();
document.addEventListener('click', e => {
  const text = e.target.textContent?.trim().slice(0, 50) || '';
  if (text.length > 3) {
    const tokens = text.toLowerCase().split(/\s+/).filter(t => t.length > 1);
    if (tokens.length > 0) {
      store.observe(tokens);
      behaviorEvents++;
    }
  }
});

// Dwell tracking
setInterval(() => {
  window._dwellMs = Date.now() - _lastInteraction;
}, 1000);
