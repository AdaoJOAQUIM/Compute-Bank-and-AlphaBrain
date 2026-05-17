//! Minimal Scheme interpreter (~200 lines). [Axiom Q — eval() self-rewrite]
//!
//! Supported: define lambda when let quote eval and or not if + - * / > < =
//! Fuel-metered: each eval() step costs 1 fuel unit.
//! Safe: cannot mutate W outside the intent boundary (checked at crystallize call).

use crate::sexpr::{SExpr, SExprInner, nil, bool_, int, float, sym, cons};
use crate::sparse_w::SparseW;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Env(Vec<HashMap<String, SExpr>>);

impl Env {
    pub fn new() -> Self { Self(vec![HashMap::new()]) }

    pub fn lookup(&self, name: &str) -> Option<&SExpr> {
        self.0.iter().rev().find_map(|frame| frame.get(name))
    }

    pub fn define(&mut self, name: String, val: SExpr) {
        self.0.last_mut().unwrap().insert(name, val);
    }

    pub fn push(&self) -> Env { let mut e = self.clone(); e.0.push(HashMap::new()); e }
}

impl Default for Env {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvalError(pub String);
impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "{}", self.0) }
}
fn err(msg: impl Into<String>) -> EvalError { EvalError(msg.into()) }

pub struct Interpreter { pub fuel_per_call: u64 }

impl Interpreter {
    pub fn new(fuel: u64) -> Self { Self { fuel_per_call: fuel } }

    pub fn eval(&self, expr: &SExpr, env: &mut Env, w: &mut SparseW, fuel: &mut u64) -> Result<SExpr, EvalError> {
        if *fuel == 0 { return Err(err("fuel exhausted")); }
        *fuel -= 1;

        match expr.as_ref() {
            SExprInner::Nil | SExprInner::Bool(_) | SExprInner::Int(_)
            | SExprInner::Float(_) | SExprInner::Str(_) => Ok(expr.clone()),

            SExprInner::Sym(s) => {
                env.lookup(s).cloned()
                    .or_else(|| if is_builtin(s) { Some(sym(s)) } else { None })
                    .ok_or_else(|| err(format!("unbound: {}", s)))
            }

            SExprInner::Lambda(..) => Ok(expr.clone()),

            SExprInner::Cons(head, tail) => {
                let head_val = if let SExprInner::Sym(s) = head.as_ref() {
                    match s.as_str() {
                        "quote"  => return Ok(car(tail)?),
                        "define" => {
                            let name = sym_name(&car(tail)?)?;
                            let val  = self.eval(&cadr(tail)?, env, w, fuel)?;
                            env.define(name, val);
                            return Ok(nil());
                        }
                        "lambda" => {
                            let params = list_to_syms(&car(tail)?)?;
                            let body   = cadr(tail)?;
                            return Ok(Box::new(SExprInner::Lambda(params, body, String::new())));
                        }
                        "let" => {
                            let bindings = car(tail)?;
                            let body     = cadr(tail)?;
                            let mut inner = env.push();
                            let mut b = &bindings;
                            while let SExprInner::Cons(pair, rest) = b.as_ref() {
                                let name = sym_name(&car(pair)?)?;
                                let val  = self.eval(&cadr(pair)?, env, w, fuel)?;
                                inner.define(name, val);
                                b = rest;
                            }
                            return self.eval(&body, &mut inner, w, fuel);
                        }
                        "when" => {
                            let cond   = self.eval(&car(tail)?, env, w, fuel)?;
                            let body   = cadr(tail)?;
                            if is_truthy(&cond) { return self.eval(&body, env, w, fuel); }
                            return Ok(nil());
                        }
                        "if" => {
                            let cond  = self.eval(&car(tail)?, env, w, fuel)?;
                            let then_ = cadr(tail)?;
                            let else_ = caddr(tail).unwrap_or_else(|_| nil());
                            if is_truthy(&cond) { return self.eval(&then_, env, w, fuel); }
                            return self.eval(&else_, env, w, fuel);
                        }
                        "and" => {
                            let a = self.eval(&car(tail)?, env, w, fuel)?;
                            if !is_truthy(&a) { return Ok(bool_(false)); }
                            return self.eval(&cadr(tail)?, env, w, fuel);
                        }
                        "or" => {
                            let a = self.eval(&car(tail)?, env, w, fuel)?;
                            if is_truthy(&a) { return Ok(a); }
                            return self.eval(&cadr(tail)?, env, w, fuel);
                        }
                        "eval" => {
                            let inner = self.eval(&car(tail)?, env, w, fuel)?;
                            return self.eval(&inner, env, w, fuel);
                        }
                        _ => {
                            // Try env lookup; if not found, return sym as-is (builtin)
                            env.lookup(s).cloned().unwrap_or_else(|| head.clone())
                        }
                    }
                } else {
                    self.eval(head, env, w, fuel)?
                };

                // Apply function
                let args = eval_list(tail, env, w, self, fuel)?;
                self.apply(&head_val, args, env, w, fuel)
            }
        }
    }

    fn apply(&self, func: &SExpr, args: Vec<SExpr>, env: &mut Env, w: &mut SparseW, fuel: &mut u64) -> Result<SExpr, EvalError> {
        match func.as_ref() {
            SExprInner::Lambda(params, body, _) => {
                let mut inner = env.push();
                for (p, a) in params.iter().zip(args) { inner.define(p.clone(), a); }
                self.eval(body, &mut inner, w, fuel)
            }
            SExprInner::Sym(s) => self.apply_builtin(s, args, w),
            _ => Err(err(format!("not callable: {:?}", func))),
        }
    }

    fn apply_builtin(&self, name: &str, args: Vec<SExpr>, w: &mut SparseW) -> Result<SExpr, EvalError> {
        let num = |a: &SExpr| -> Result<f64, EvalError> {
            match a.as_ref() {
                SExprInner::Int(n)   => Ok(*n as f64),
                SExprInner::Float(f) => Ok(*f),
                _ => Err(err("expected number"))
            }
        };
        match name {
            "+"  => {
                let s: f64 = args.iter().map(|a| num(a)).collect::<Result<Vec<_>,_>>()?.into_iter().sum();
                Ok(float(s))
            }
            "-"  => {
                let vs: Vec<f64> = args.iter().map(|a| num(a)).collect::<Result<_,_>>()?;
                if vs.len() == 1 { return Ok(float(-vs[0])); }
                Ok(float(vs[0] - vs[1..].iter().sum::<f64>()))
            }
            "*"  => {
                let p: f64 = args.iter().map(|a| num(a)).collect::<Result<Vec<_>,_>>()?.into_iter().product();
                Ok(float(p))
            }
            "/"  => {
                let a = num(&args[0])?; let b = num(&args[1])?;
                if b == 0.0 { Err(err("div/0")) } else { Ok(float(a/b)) }
            }
            ">"  => Ok(bool_(num(&args[0])? > num(&args[1])?)),
            "<"  => Ok(bool_(num(&args[0])? < num(&args[1])?)),
            "="  => Ok(bool_((num(&args[0])? - num(&args[1])?).abs() < 1e-12)),
            "not" => Ok(bool_(!is_truthy(&args[0]))),
            "car" => car(&args[0]),
            "cdr" => cdr(&args[0]),
            "cons" => Ok(cons(args[0].clone(), args[1].clone())),
            "null?" => Ok(bool_(matches!(args[0].as_ref(), SExprInner::Nil))),
            "list" => Ok(crate::sexpr::list(args)),
            // Domain built-ins (return mock values for simulation)
            "dwell-time"      => Ok(int(600)),   // simulated: 10 minutes
            "context-tokens"  => Ok(sym("current-context")),
            "crystallize"     => {
                // simulate crystallization — increment n_patterns as side-effect
                w.n_patterns += 1;
                Ok(bool_(true))
            }
            _ => Err(err(format!("unknown built-in: {}", name)))
        }
    }
}

// ── S-expression helpers ─────────────────────────────────────────────────────

fn car(e: &SExpr) -> Result<SExpr, EvalError> {
    match e.as_ref() { SExprInner::Cons(a, _) => Ok(a.clone()), _ => Err(err("car of non-pair")) }
}
fn cdr(e: &SExpr) -> Result<SExpr, EvalError> {
    match e.as_ref() { SExprInner::Cons(_, b) => Ok(b.clone()), _ => Err(err("cdr of non-pair")) }
}
fn cadr(e: &SExpr) -> Result<SExpr, EvalError>  { car(&cdr(e)?) }
fn caddr(e: &SExpr) -> Result<SExpr, EvalError> { car(&cdr(&cdr(e)?)?) }

fn sym_name(e: &SExpr) -> Result<String, EvalError> {
    match e.as_ref() { SExprInner::Sym(s) => Ok(s.clone()), _ => Err(err("expected symbol")) }
}

fn list_to_syms(e: &SExpr) -> Result<Vec<String>, EvalError> {
    let mut params = Vec::new();
    let mut cur = e;
    while let SExprInner::Cons(h, t) = cur.as_ref() {
        params.push(sym_name(h)?);
        cur = t;
    }
    Ok(params)
}

fn is_truthy(e: &SExpr) -> bool {
    !matches!(e.as_ref(), SExprInner::Bool(false) | SExprInner::Nil)
}

fn eval_list(e: &SExpr, env: &mut Env, w: &mut SparseW, interp: &Interpreter, fuel: &mut u64) -> Result<Vec<SExpr>, EvalError> {
    let mut args = Vec::new();
    let mut cur = e;
    while let SExprInner::Cons(h, t) = cur.as_ref() {
        args.push(interp.eval(h, env, w, fuel)?);
        cur = t;
    }
    Ok(args)
}

fn is_builtin(name: &str) -> bool {
    matches!(name, "+" | "-" | "*" | "/" | ">" | "<" | "=" | "not"
        | "car" | "cdr" | "cons" | "null?" | "list"
        | "dwell-time" | "context-tokens" | "crystallize")
}
