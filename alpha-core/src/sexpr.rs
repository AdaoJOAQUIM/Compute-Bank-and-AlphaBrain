//! S-expression type for homoiconic W storage (Axiom VIII).

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SExprInner {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Sym(String),
    Str(String),
    Cons(SExpr, SExpr),
    /// Stored as (params, body, env-snapshot string)
    Lambda(Vec<String>, SExpr, String),
}

pub type SExpr = Box<SExprInner>;

pub fn nil() -> SExpr { Box::new(SExprInner::Nil) }
pub fn bool_(b: bool) -> SExpr { Box::new(SExprInner::Bool(b)) }
pub fn int(n: i64) -> SExpr { Box::new(SExprInner::Int(n)) }
pub fn float(f: f64) -> SExpr { Box::new(SExprInner::Float(f)) }
pub fn sym(s: &str) -> SExpr { Box::new(SExprInner::Sym(s.to_string())) }
pub fn str_(s: &str) -> SExpr { Box::new(SExprInner::Str(s.to_string())) }
pub fn cons(a: SExpr, b: SExpr) -> SExpr { Box::new(SExprInner::Cons(a, b)) }

/// Construct a list from a Vec of S-expressions.
pub fn list(mut items: Vec<SExpr>) -> SExpr {
    items.reverse();
    let mut result = nil();
    for item in items { result = cons(item, result); }
    result
}

/// Parse a simple S-expression from a string.
pub fn parse(s: &str) -> Result<SExpr, String> {
    let tokens = tokenize(s)?;
    let (expr, _) = parse_tokens(&tokens, 0)?;
    Ok(expr)
}

fn tokenize(s: &str) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' | ')' | '\'' => { tokens.push(c.to_string()); chars.next(); }
            '"' => {
                chars.next();
                let mut buf = String::new();
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '"' { break; }
                    buf.push(c);
                }
                tokens.push(format!("\"{}\"", buf));
            }
            ' ' | '\t' | '\n' | '\r' => { chars.next(); }
            _ => {
                let mut buf = String::new();
                while let Some(&c) = chars.peek() {
                    if c == '(' || c == ')' || c == ' ' || c == '\n' { break; }
                    buf.push(c); chars.next();
                }
                tokens.push(buf);
            }
        }
    }
    Ok(tokens)
}

fn parse_tokens(tokens: &[String], pos: usize) -> Result<(SExpr, usize), String> {
    if pos >= tokens.len() { return Err("unexpected end".into()); }
    match tokens[pos].as_str() {
        "'" => {
            let (inner, next) = parse_tokens(tokens, pos + 1)?;
            Ok((list(vec![sym("quote"), inner]), next))
        }
        "(" => {
            let mut items = Vec::new();
            let mut p = pos + 1;
            while p < tokens.len() && tokens[p] != ")" {
                let (item, next) = parse_tokens(tokens, p)?;
                items.push(item);
                p = next;
            }
            Ok((list(items), p + 1))
        }
        t if t.starts_with('"') => {
            let s = t.trim_matches('"').to_string();
            Ok((str_(&s), pos + 1))
        }
        "#t" | "#true"  => Ok((bool_(true),  pos + 1)),
        "#f" | "#false" => Ok((bool_(false), pos + 1)),
        t => {
            if let Ok(n) = t.parse::<i64>()  { return Ok((int(n), pos + 1)); }
            if let Ok(f) = t.parse::<f64>()  { return Ok((float(f), pos + 1)); }
            Ok((sym(t), pos + 1))
        }
    }
}

impl fmt::Display for SExprInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SExprInner::Nil          => write!(f, "()"),
            SExprInner::Bool(b)      => write!(f, "{}", if *b { "#t" } else { "#f" }),
            SExprInner::Int(n)       => write!(f, "{}", n),
            SExprInner::Float(x)     => write!(f, "{}", x),
            SExprInner::Sym(s)       => write!(f, "{}", s),
            SExprInner::Str(s)       => write!(f, "\"{}\"", s),
            SExprInner::Lambda(p,..) => write!(f, "(lambda {:?} ...)", p),
            SExprInner::Cons(a, b)   => {
                write!(f, "({}", a)?;
                let mut cur = b;
                loop {
                    match cur.as_ref() {
                        SExprInner::Nil       => { write!(f, ")")?; break; }
                        SExprInner::Cons(x,y) => { write!(f, " {}", x)?; cur = y; }
                        other                 => { write!(f, " . {})", other)?; break; }
                    }
                }
                Ok(())
            }
        }
    }
}
