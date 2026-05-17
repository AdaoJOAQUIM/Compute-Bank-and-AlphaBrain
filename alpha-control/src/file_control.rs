use alpha_core::{SeedLexicon};
use std::path::PathBuf;

pub struct FileControl {
    pub root: PathBuf,
    pub lexicon: SeedLexicon,
}

impl FileControl {
    pub fn new(root: PathBuf, genesis: [u8;32]) -> Self {
        Self { root, lexicon: SeedLexicon::new(genesis) }
    }

    // List files matching a description (crystallized query)
    pub fn find(&mut self, description: &[&str]) -> Vec<PathBuf> {
        let _query = self.lexicon.crystallize(description);
        // Stub: walk root directory up to depth 3
        let mut results = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                results.push(entry.path());
            }
        }
        results
    }

    // Read file content and crystallize it into the lexicon
    pub fn read_and_crystallize(&mut self, path: &PathBuf) -> Option<String> {
        std::fs::read_to_string(path).ok()
    }

    // Write content (after SGR retrieve attempt — Axiom H)
    pub fn write(&self, path: &PathBuf, content: &str) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)
    }
}
