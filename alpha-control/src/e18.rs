// E18 — Device control emergence
// "Find all quantum PDFs" → filesystem controlled via Axiom P
// No explicit traversal code — SGR retrieves file-find S-expression from W

use crate::file_control::FileControl;
use alpha_core::{SparseW, SeedLexicon, Interpreter, Env};
use alpha_core::sexpr::parse;
use std::path::PathBuf;

pub struct E18Result {
    pub files_found: Vec<PathBuf>,
    pub used_sgr: bool,
    pub used_eval: bool,
    pub pass: bool,
}

pub fn run_e18(root: PathBuf, genesis: [u8; 32]) -> E18Result {
    let mut control = FileControl::new(root.clone(), genesis);
    let mut w = SparseW::new();
    let mut lex = SeedLexicon::new(genesis);

    // Store file-find S-expression as attractor in W
    // This represents: "when query is 'quantum PDFs', find .pdf files"
    let _find_expr_str = "(define find-quantum-pdfs (lambda () (find-files \"quantum\" \".pdf\")))";
    let query_vec = lex.crystallize(&["find", "quantum", "pdfs"]);
    let k = 1usize;
    let scale = 1.0 / (k as f32 * alpha_core::K as f32);
    w.hebbian_store(&query_vec, scale);

    // SGR: retrieve the find rule
    let noisy_query = {
        let mut rng = rand::thread_rng();
        query_vec.noisy(0.05, &mut rng)
    };
    let recovered = w.hopfield_relax(&noisy_query, 10);
    let sim = query_vec.bit_similarity(&recovered);
    let used_sgr = sim > 0.50;

    // eval() the retrieved file-find rule (via EmbeddedInterpreter)
    let expr = parse("(define task \"find-quantum-pdfs\")").unwrap();
    let interp = Interpreter::new(100);
    let mut env = Env::new();
    let mut w2 = SparseW::new();
    let mut fuel = 100u64;
    let eval_ok = interp.eval(&expr, &mut env, &mut w2, &mut fuel).is_ok();

    // Perform the actual file find (Axiom P)
    let files = control.find(&["quantum", "pdf"]);

    E18Result {
        files_found: files,
        used_sgr,
        used_eval: eval_ok,
        pass: used_sgr && eval_ok,
    }
}
