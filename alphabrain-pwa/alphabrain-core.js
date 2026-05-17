/**
 * AlphaBrain v10.0 — Core JS runtime
 *
 * Implements in JavaScript (client-side, no server, no npm):
 *   - SparseVec (K=13 active dims, bipolar ±1.0)
 *   - SparseW   (Map of "i,j"->f32, lower-triangle only)
 *   - SeedLexicon (FNV-based n-gram hash; deterministic)
 *   - SGR       (semantic gradient routing on W)
 *   - Pheromone (Φ₁..Φ₅, Laplacian diffusion)
 *   - EmbeddedInterpreter (minimal Scheme eval, fuel-metered)
 *   - CrystallizationStore
 *
 * No external dependencies. Runs on Raspberry Pi browser.
 * Axioms: A,B,C,D,E,F,G,H,J,K,L,M
 */

'use strict';

// ── Constants ────────────────────────────────────────────────────────────────
const D = 256;
const K = 13;
const ETA = 0.01;
const LAMBDA = 0.001;
const K_B = 1.380649e-23;
const T_ROOM = 293.15;
const LANDAUER_MIN = K_B * T_ROOM * Math.LN2;

// ── SparseVec ────────────────────────────────────────────────────────────────
class SparseVec {
  constructor(indices, values) {
    this.indices = new Uint16Array(indices); // sorted, len == K
    this.values  = new Float32Array(values); // ±1.0
  }

  static random(rngFn) {
    // Sample K distinct indices from [0, D)
    const pool = Array.from({length: D}, (_, i) => i);
    const idx = [];
    for (let i = 0; i < K; i++) {
      const j = i + Math.floor(rngFn() * (D - i));
      [pool[i], pool[j]] = [pool[j], pool[i]];
      idx.push(pool[i]);
    }
    idx.sort((a, b) => a - b);
    const vals = idx.map(() => rngFn() < 0.5 ? 1.0 : -1.0);
    return new SparseVec(idx, vals);
  }

  noisy(noise, rngFn) {
    const vals = Array.from(this.values).map(v => rngFn() < noise ? -v : v);
    return new SparseVec(this.indices, vals);
  }

  bitSimilarity(other) {
    let matches = 0, intersection = 0;
    let i = 0, j = 0;
    while (i < this.indices.length && j < other.indices.length) {
      if (this.indices[i] === other.indices[j]) {
        intersection++;
        if ((this.values[i] > 0) === (other.values[j] > 0)) matches++;
        i++; j++;
      } else if (this.indices[i] < other.indices[j]) { i++; }
      else { j++; }
    }
    const union = this.indices.length + other.indices.length - intersection;
    return union === 0 ? 0 : matches / union;
  }

  activeOverlap(other) {
    let common = 0, i = 0, j = 0;
    while (i < this.indices.length && j < other.indices.length) {
      if (this.indices[i] === other.indices[j]) { common++; i++; j++; }
      else if (this.indices[i] < other.indices[j]) { i++; }
      else { j++; }
    }
    return common / K;
  }

  equals(other) {
    if (this.indices.length !== other.indices.length) return false;
    for (let i = 0; i < this.indices.length; i++) {
      if (this.indices[i] !== other.indices[i]) return false;
      if (Math.abs(this.values[i] - other.values[i]) > 1e-6) return false;
    }
    return true;
  }
}

// ── SparseW ──────────────────────────────────────────────────────────────────
class SparseW {
  constructor() {
    this.entries = new Map(); // "i,j" (i<j) -> f32
    this.nPatterns = 0;
  }

  _key(i, j) { return i < j ? `${i},${j}` : `${j},${i}`; }

  get(i, j) {
    if (i === j) return 0;
    return this.entries.get(this._key(i, j)) || 0;
  }

  hebbianStore(xi, scale) {
    for (let ii = 0; ii < xi.indices.length; ii++) {
      for (let jj = ii + 1; jj < xi.indices.length; jj++) {
        const i = xi.indices[ii], j = xi.indices[jj];
        const delta = scale * xi.values[ii] * xi.values[jj];
        const key = this._key(i, j);
        this.entries.set(key, (this.entries.get(key) || 0) + delta);
      }
    }
    this.nPatterns++;
  }

  plasticityStep(psi) {
    // Decay
    for (const [k, v] of this.entries) this.entries.set(k, v * (1 - LAMBDA));
    // Reinforce active pairs
    for (let ii = 0; ii < psi.indices.length; ii++) {
      for (let jj = ii + 1; jj < psi.indices.length; jj++) {
        const i = psi.indices[ii], j = psi.indices[jj];
        const key = this._key(i, j);
        this.entries.set(key, (this.entries.get(key) || 0) + ETA * psi.values[ii] * psi.values[jj]);
      }
    }
  }

  energy(psi) {
    let e = 0;
    for (let ii = 0; ii < psi.indices.length; ii++) {
      for (let jj = ii + 1; jj < psi.indices.length; jj++) {
        e += this.get(psi.indices[ii], psi.indices[jj]) * psi.values[ii] * psi.values[jj];
      }
    }
    return -0.5 * e / K;
  }

  hopfieldStep(psi) {
    const vals = new Float32Array(psi.indices.length);
    for (let ii = 0; ii < psi.indices.length; ii++) {
      let sum = 0;
      for (let jj = 0; jj < psi.indices.length; jj++) {
        if (ii !== jj) sum += this.get(psi.indices[ii], psi.indices[jj]) * psi.values[jj];
      }
      vals[ii] = sum >= 0 ? 1.0 : -1.0;
    }
    return new SparseVec(psi.indices, vals);
  }

  hopfieldRelax(psi, maxSteps = 20) {
    let cur = psi;
    for (let s = 0; s < maxSteps; s++) {
      const nxt = this.hopfieldStep(cur);
      if (nxt.equals(cur)) return nxt;
      cur = nxt;
    }
    return cur;
  }

  merge(other, selfW, otherW) {
    const total = selfW + otherW;
    const wa = selfW / total, wb = otherW / total;
    const merged = new SparseW();
    for (const [k, v] of this.entries) merged.entries.set(k, (merged.entries.get(k) || 0) + wa * v);
    for (const [k, v] of other.entries) merged.entries.set(k, (merged.entries.get(k) || 0) + wb * v);
    for (const [k, v] of merged.entries) if (Math.abs(v) < 1e-9) merged.entries.delete(k);
    merged.nPatterns = Math.floor((this.nPatterns + other.nPatterns + 1) / 2);
    return merged;
  }

  get entryCount() { return this.entries.size; }
}

// ── SeedLexicon ──────────────────────────────────────────────────────────────
// FNV-1a based (no crypto needed in browser, deterministic across JS engines)
class SeedLexicon {
  constructor(genesis) {
    this.genesis = genesis; // Uint8Array[32]
    this.cache = new Map();
  }

  _fnv1a(data) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < data.length; i++) {
      h ^= data[i];
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h;
  }

  _hashToken(token, n) {
    // n-grams of token combined with genesis seed
    const chars = [...token];
    const ngrams = [];
    if (chars.length < n) {
      ngrams.push(token);
    } else {
      for (let i = 0; i <= chars.length - n; i++) {
        ngrams.push(chars.slice(i, i + n).join(''));
      }
    }
    const encoded = new TextEncoder().encode(
      this.genesis.join(',') + '|' + ngrams.join('\xb7')
    );
    return this._fnv1a(encoded);
  }

  seed(token) {
    if (this.cache.has(token)) return this.cache.get(token);

    // Weighted merge of 3 hash levels
    const h2 = this._hashToken(token, 2);
    const h3 = this._hashToken(token, 3);
    const h4 = this._hashToken(token, 4);

    // Use merged hash as LCG seed for deterministic sampling
    const merged = (h2 * 0.20 + h3 * 0.50 + h4 * 0.30) >>> 0;

    // LCG: deterministic, fast
    let state = merged || 1;
    const lcg = () => {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      return state / 0x100000000;
    };

    // Sample K distinct indices from [0, D)
    const pool = Array.from({length: D}, (_, i) => i);
    const idx = [];
    for (let i = 0; i < K; i++) {
      const j = i + Math.floor(lcg() * (D - i));
      [pool[i], pool[j]] = [pool[j], pool[i]];
      idx.push(pool[i]);
    }
    idx.sort((a, b) => a - b);
    const vals = idx.map(() => lcg() < 0.5 ? 1.0 : -1.0);
    const v = new SparseVec(idx, vals);
    this.cache.set(token, v);
    return v;
  }

  crystallize(tokens) {
    const idxVals = new Map();
    for (const t of tokens) {
      const v = this.seed(t);
      for (let i = 0; i < v.indices.length; i++) {
        const key = v.indices[i];
        idxVals.set(key, (idxVals.get(key) || 0) + v.values[i]);
      }
    }
    // Top-K by absolute value, tie-break by index
    let pairs = Array.from(idxVals.entries())
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]) || a[0] - b[0]);
    pairs = pairs.slice(0, K);
    pairs.sort((a, b) => a[0] - b[0]);
    const indices = pairs.map(p => p[0]);
    const values  = pairs.map(p => p[1] >= 0 ? 1.0 : -1.0);
    // Pad if needed (tokens too short)
    while (indices.length < K) { indices.push(0); values.push(1.0); }
    return new SparseVec(indices.slice(0, K), values.slice(0, K));
  }
}

// ── Pheromone ────────────────────────────────────────────────────────────────
class Pheromone {
  constructor() {
    this.phi = new Float64Array(5); // Φ₁..Φ₅
    this.diffusion = 0.1;
    this.decay = [0.05, 0.02, 0.10, 0.08, 0.03];
  }

  step(neighborPhis, sources, dt) {
    const n = neighborPhis.length;
    for (let ch = 0; ch < 5; ch++) {
      const meanNbr = n > 0
        ? neighborPhis.reduce((s, p) => s + p[ch], 0) / n
        : 0;
      const lap = meanNbr - this.phi[ch];
      this.phi[ch] += dt * (this.diffusion * lap + sources[ch] - this.decay[ch] * this.phi[ch]);
      this.phi[ch] = Math.max(0, Math.min(1, this.phi[ch]));
    }
  }

  get phi1() { return this.phi[0]; }
  get phi2() { return this.phi[1]; }
  get phi5() { return this.phi[4]; }
  adiabaticOk(theta = 0.85) { return this.phi[0] < theta; }
  consolidateNow() { return this.phi[1] < 0.20; }
}

// ── SGR ──────────────────────────────────────────────────────────────────────
function sgrRetrieve(query, startNodeId, nodes, maxHops = 10) {
  let current = startNodeId;
  const path = [current];

  for (let hop = 0; hop < maxHops; hop++) {
    const node = nodes.get(current);
    if (!node) break;
    const curEnergy = node.w.energy(query);

    let bestId = null, bestE = curEnergy;
    for (const nbr of node.neighbors) {
      const nbrNode = nodes.get(nbr);
      if (!nbrNode) continue;
      const e = nbrNode.w.energy(query);
      if (e < bestE) { bestE = e; bestId = nbr; }
    }

    if (bestId === null) break;
    current = bestId;
    path.push(current);
  }

  const dest = nodes.get(current);
  const recovered = dest ? dest.w.hopfieldRelax(query) : query;
  const fidelity = query.bitSimilarity(recovered);

  return { recovered, hops: path.length - 1, path, fidelity };
}

// ── EmbeddedInterpreter ──────────────────────────────────────────────────────
// Minimal Scheme eval in JS. Fuel-metered. Safe.
class SExpr {
  constructor(type, value) { this.type = type; this.value = value; }
  static nil() { return new SExpr('nil', null); }
  static bool(v) { return new SExpr('bool', v); }
  static int(v) { return new SExpr('int', v); }
  static float(v) { return new SExpr('float', v); }
  static sym(s) { return new SExpr('sym', s); }
  static str(s) { return new SExpr('str', s); }
  static cons(a, b) { return new SExpr('cons', [a, b]); }
  static list(items) {
    let r = SExpr.nil();
    for (let i = items.length - 1; i >= 0; i--) r = SExpr.cons(items[i], r);
    return r;
  }
  toString() {
    switch (this.type) {
      case 'nil': return '()';
      case 'bool': return this.value ? '#t' : '#f';
      case 'int': return String(this.value);
      case 'float': return String(this.value);
      case 'sym': return this.value;
      case 'str': return `"${this.value}"`;
      case 'cons': return `(${this._listStr()})`;
      case 'lambda': return '#<lambda>';
      default: return `#<${this.type}>`;
    }
  }
  _listStr() {
    const parts = [];
    let cur = this;
    while (cur.type === 'cons') { parts.push(cur.value[0].toString()); cur = cur.value[1]; }
    if (cur.type !== 'nil') parts.push('. ' + cur.toString());
    return parts.join(' ');
  }
}

function parseSexpr(s) {
  s = s.trim();
  let pos = 0;
  function skip() { while (pos < s.length && /\s/.test(s[pos])) pos++; }
  function read() {
    skip();
    if (pos >= s.length) throw new Error('EOF');
    if (s[pos] === '(') {
      pos++;
      const items = [];
      while (true) {
        skip();
        if (s[pos] === ')') { pos++; break; }
        items.push(read());
      }
      return SExpr.list(items);
    }
    if (s[pos] === "'") { pos++; return SExpr.list([SExpr.sym('quote'), read()]); }
    if (s[pos] === '"') {
      pos++;
      let str = '';
      while (pos < s.length && s[pos] !== '"') str += s[pos++];
      pos++;
      return SExpr.str(str);
    }
    let tok = '';
    while (pos < s.length && !/[\s()"]/.test(s[pos])) tok += s[pos++];
    if (tok === '#t') return SExpr.bool(true);
    if (tok === '#f') return SExpr.bool(false);
    if (/^-?\d+$/.test(tok)) return SExpr.int(parseInt(tok));
    if (/^-?\d*\.\d+$/.test(tok)) return SExpr.float(parseFloat(tok));
    return SExpr.sym(tok);
  }
  return read();
}

const BUILTINS = new Set(['+','-','*','/','<','>','=','not','car','cdr','cons','null?','list',
  'dwell-time','context-tokens','crystallize','display','newline']);

class Interpreter {
  constructor(fuelPerCall = 500) { this.fuelPerCall = fuelPerCall; }

  eval(expr, env, w, fuel) {
    if (fuel.v <= 0) throw new Error('fuel exhausted');
    fuel.v--;

    switch (expr.type) {
      case 'nil': case 'bool': case 'int': case 'float': case 'str': return expr;
      case 'lambda': return expr;
      case 'sym': {
        const v = env.lookup(expr.value);
        if (v !== undefined) return v;
        if (BUILTINS.has(expr.value)) return expr;
        throw new Error(`unbound: ${expr.value}`);
      }
      case 'cons': {
        const head = expr.value[0];
        const tail = expr.value[1];
        // Special forms
        if (head.type === 'sym') {
          const s = head.value;
          if (s === 'quote') return this._car(tail);
          if (s === 'define') {
            const name = this._symName(this._car(tail));
            const val = this.eval(this._cadr(tail), env, w, fuel);
            env.define(name, val);
            return SExpr.nil();
          }
          if (s === 'lambda') {
            const params = this._listToSyms(this._car(tail));
            const body = this._cadr(tail);
            return new SExpr('lambda', { params, body, env: env.snapshot() });
          }
          if (s === 'let') {
            const bindings = this._car(tail);
            const body = this._cadr(tail);
            const inner = env.push();
            let b = bindings;
            while (b.type === 'cons') {
              const pair = b.value[0];
              inner.define(this._symName(this._car(pair)), this.eval(this._cadr(pair), env, w, fuel));
              b = b.value[1];
            }
            return this.eval(body, inner, w, fuel);
          }
          if (s === 'when') {
            const cond = this.eval(this._car(tail), env, w, fuel);
            if (this._truthy(cond)) return this.eval(this._cadr(tail), env, w, fuel);
            return SExpr.nil();
          }
          if (s === 'if') {
            const cond = this.eval(this._car(tail), env, w, fuel);
            return this._truthy(cond)
              ? this.eval(this._cadr(tail), env, w, fuel)
              : this.eval(this._caddr(tail) || SExpr.nil(), env, w, fuel);
          }
          if (s === 'and') {
            const a = this.eval(this._car(tail), env, w, fuel);
            return this._truthy(a) ? this.eval(this._cadr(tail), env, w, fuel) : SExpr.bool(false);
          }
          if (s === 'or') {
            const a = this.eval(this._car(tail), env, w, fuel);
            return this._truthy(a) ? a : this.eval(this._cadr(tail), env, w, fuel);
          }
          if (s === 'eval') {
            const inner = this.eval(this._car(tail), env, w, fuel);
            return this.eval(inner, env, w, fuel);
          }
          if (s === 'begin') {
            let result = SExpr.nil();
            let cur = tail;
            while (cur.type === 'cons') { result = this.eval(cur.value[0], env, w, fuel); cur = cur.value[1]; }
            return result;
          }
        }
        // Function application
        const fn = head.type === 'sym' && !BUILTINS.has(head.value)
          ? this.eval(head, env, w, fuel)
          : (head.type === 'sym' ? head : this.eval(head, env, w, fuel));
        const args = this._evalList(tail, env, w, fuel);
        return this._apply(fn, args, env, w, fuel);
      }
      default: return expr;
    }
  }

  _apply(fn, args, env, w, fuel) {
    if (fn.type === 'lambda') {
      const inner = Env.fromSnapshot(fn.value.env).push();
      fn.value.params.forEach((p, i) => inner.define(p, args[i] || SExpr.nil()));
      return this.eval(fn.value.body, inner, w, fuel);
    }
    if (fn.type === 'sym') return this._builtin(fn.value, args, w);
    throw new Error(`not callable: ${fn.type}`);
  }

  _builtin(name, args, w) {
    const num = a => {
      if (a.type === 'int' || a.type === 'float') return a.value;
      throw new Error('expected number');
    };
    switch (name) {
      case '+': return SExpr.float(args.reduce((s, a) => s + num(a), 0));
      case '-': return args.length === 1 ? SExpr.float(-num(args[0])) : SExpr.float(num(args[0]) - args.slice(1).reduce((s,a) => s + num(a), 0));
      case '*': return SExpr.float(args.reduce((p, a) => p * num(a), 1));
      case '/': { const b = num(args[1]); if (b===0) throw new Error('div/0'); return SExpr.float(num(args[0])/b); }
      case '>': return SExpr.bool(num(args[0]) > num(args[1]));
      case '<': return SExpr.bool(num(args[0]) < num(args[1]));
      case '=': return SExpr.bool(Math.abs(num(args[0]) - num(args[1])) < 1e-12);
      case 'not': return SExpr.bool(!this._truthy(args[0]));
      case 'car': return this._car(args[0]);
      case 'cdr': return this._cdr(args[0]);
      case 'cons': return SExpr.cons(args[0], args[1]);
      case 'null?': return SExpr.bool(args[0].type === 'nil');
      case 'list': return SExpr.list(args);
      case 'display': { console.log(args[0].toString()); return SExpr.nil(); }
      case 'newline': { console.log(); return SExpr.nil(); }
      case 'dwell-time': return SExpr.int(window._dwellMs || 0);
      case 'context-tokens': return SExpr.str(window._contextTokens || '');
      case 'crystallize': { w.nPatterns++; return SExpr.bool(true); }
      default: throw new Error(`unknown builtin: ${name}`);
    }
  }

  _car(e) { if (e.type === 'cons') return e.value[0]; throw new Error('car of non-pair'); }
  _cdr(e) { if (e.type === 'cons') return e.value[1]; throw new Error('cdr of non-pair'); }
  _cadr(e) { return this._car(this._cdr(e)); }
  _caddr(e) { try { return this._car(this._cdr(this._cdr(e))); } catch { return null; } }
  _symName(e) { if (e.type === 'sym') return e.value; throw new Error('expected symbol'); }
  _listToSyms(e) {
    const params = [];
    let cur = e;
    while (cur.type === 'cons') { params.push(this._symName(cur.value[0])); cur = cur.value[1]; }
    return params;
  }
  _truthy(e) { return !(e.type === 'bool' && !e.value) && e.type !== 'nil'; }
  _evalList(e, env, w, fuel) {
    const args = [];
    let cur = e;
    while (cur.type === 'cons') { args.push(this.eval(cur.value[0], env, w, fuel)); cur = cur.value[1]; }
    return args;
  }
}

class Env {
  constructor(frames) { this.frames = frames || [new Map()]; }
  lookup(name) {
    for (let i = this.frames.length - 1; i >= 0; i--) {
      if (this.frames[i].has(name)) return this.frames[i].get(name);
    }
    return undefined;
  }
  define(name, val) { this.frames[this.frames.length - 1].set(name, val); }
  push() { return new Env([...this.frames, new Map()]); }
  snapshot() { return this.frames.map(f => new Map(f)); }
  static fromSnapshot(snap) { return new Env(snap.map(f => new Map(f))); }
}

// ── CrystallizationStore ─────────────────────────────────────────────────────
class CrystallizationStore {
  constructor(genesis) {
    this.lexicon = new SeedLexicon(new Uint8Array(genesis));
    this.intents = [];
    this.interactions = 0;
  }

  observe(tokens) {
    this.interactions++;
    const vector = this.lexicon.crystallize(tokens);
    let matchIdx = -1;
    for (let i = 0; i < this.intents.length; i++) {
      if (this.intents[i].vector.bitSimilarity(vector) > 0.5) { matchIdx = i; break; }
    }
    if (matchIdx >= 0) {
      this.intents[matchIdx].count++;
      if (this.intents[matchIdx].count >= 5) this.intents[matchIdx].confirmed = true;
    } else {
      this.intents.push({ tokens, vector, count: 1, confirmed: false });
    }
    return vector;
  }

  get intentAccuracy() {
    if (this.intents.length === 0) return 0;
    return this.intents.filter(i => i.confirmed).length / this.intents.length;
  }
}

// ── AlphaNode / AlphaNetwork ─────────────────────────────────────────────────
class AlphaNode {
  constructor(id, deviceClass) {
    this.id = id;
    this.deviceClass = deviceClass; // 'laptop'|'phone'|'cloud'
    this.w = new SparseW();
    this.pheromone = new Pheromone();
    this.neighbors = [];
    this.opsThisTick = 0;
    this.patternsStored = 0;
  }

  storePattern(xi, k) {
    const scale = 1.0 / (k * K);
    this.w.hebbianStore(xi, scale);
    this.patternsStored++;
    this.opsThisTick += K * K;
  }

  adapt(psi) {
    this.w.plasticityStep(psi);
    this.opsThisTick += K * K;
  }

  phi1Source(cpuBudget) {
    return Math.min(1, this.opsThisTick / cpuBudget);
  }

  resetTick() { this.opsThisTick = 0; }
}

class AlphaNetwork {
  constructor() {
    this.nodes = new Map();
    this.tick = 0;
    this.dead = new Set();
  }

  static threeNode() {
    const net = new AlphaNetwork();
    const l = new AlphaNode(0, 'laptop');
    const p = new AlphaNode(1, 'phone');
    const c = new AlphaNode(2, 'cloud');
    l.neighbors = [1, 2]; p.neighbors = [0, 2]; c.neighbors = [0, 1];
    net.nodes.set(0, l); net.nodes.set(1, p); net.nodes.set(2, c);
    return net;
  }

  aliveIds() { return [...this.nodes.keys()].filter(id => !this.dead.has(id)).sort(); }
  getNode(id) { return this.dead.has(id) ? null : this.nodes.get(id); }
  kill(id) { this.dead.add(id); }

  pheromeoneTick(dt = 0.1) {
    const snap = new Map();
    for (const [id, node] of this.nodes) snap.set(id, [...node.pheromone.phi]);
    for (const [id, node] of this.nodes) {
      if (this.dead.has(id)) continue;
      const nbrPhis = node.neighbors.filter(n => !this.dead.has(n)).map(n => snap.get(n));
      const src1 = node.phi1Source(10000);
      node.pheromone.step(nbrPhis, [src1, 0, 0, 0, 0], dt);
      node.resetTick();
    }
    this.tick++;
  }
}

// ── Global runtime instance ──────────────────────────────────────────────────
window.AlphaBrainRuntime = {
  SparseVec, SparseW, SeedLexicon, SExpr, parseSexpr,
  Interpreter, Env, Pheromone, CrystallizationStore,
  AlphaNode, AlphaNetwork, sgrRetrieve,
  D, K, ETA, LAMBDA, LANDAUER_MIN
};
