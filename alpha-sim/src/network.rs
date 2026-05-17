use std::collections::{HashMap, HashSet};
use alpha_core::{NodeId, SparseVec};
use crate::node::{AlphaNode, DeviceClass};

pub struct AlphaNetwork {
    pub nodes: HashMap<NodeId, AlphaNode>,
    pub tick:  u64,
    dead:      HashSet<NodeId>,
}

impl AlphaNetwork {
    /// Three-node network: laptop + phone + optional cloud.
    pub fn three_node() -> Self {
        let mut net = Self { nodes: HashMap::new(), tick: 0, dead: HashSet::new() };
        let laptop = NodeId(0); let phone = NodeId(1); let cloud = NodeId(2);
        let mut l = AlphaNode::new(laptop, DeviceClass::Laptop);
        let mut p = AlphaNode::new(phone,  DeviceClass::Phone);
        let mut c = AlphaNode::new(cloud,  DeviceClass::Cloud);
        l.neighbors = vec![phone, cloud];
        p.neighbors = vec![laptop, cloud];
        c.neighbors = vec![laptop, phone];
        net.nodes.insert(laptop, l);
        net.nodes.insert(phone,  p);
        net.nodes.insert(cloud,  c);
        net
    }

    pub fn full_mesh(n: usize) -> Self {
        let mut net = Self { nodes: HashMap::new(), tick: 0, dead: HashSet::new() };
        let ids: Vec<NodeId> = (0..n).map(|i| NodeId(i as u64)).collect();
        for &id in &ids {
            let class = if id.0 == 0 { DeviceClass::Laptop }
                        else if id.0 == 1 { DeviceClass::Phone }
                        else { DeviceClass::Cloud };
            let mut node = AlphaNode::new(id, class);
            node.neighbors = ids.iter().filter(|&&x| x != id).copied().collect();
            net.nodes.insert(id, node);
        }
        net
    }

    pub fn alive_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.nodes.keys().filter(|id| !self.dead.contains(id)).copied().collect();
        ids.sort();
        ids
    }

    pub fn kill(&mut self, id: NodeId) { self.dead.insert(id); }
    pub fn alive_count(&self) -> usize { self.nodes.len() - self.dead.len() }
    pub fn is_alive(&self, id: NodeId) -> bool { !self.dead.contains(&id) }

    pub fn get_node(&self, id: NodeId) -> Option<&AlphaNode> {
        if self.dead.contains(&id) { None } else { self.nodes.get(&id) }
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut AlphaNode> {
        if self.dead.contains(&id) { None } else { self.nodes.get_mut(&id) }
    }

    pub fn pheromone_tick(&mut self, dt: f64) {
        let snap: HashMap<NodeId, [f64; 5]> = self.nodes.iter()
            .map(|(&id, n)| (id, n.pheromone.phi))
            .collect();
        for (&id, node) in self.nodes.iter_mut() {
            if self.dead.contains(&id) { continue; }
            let nbr_phis: Vec<[f64; 5]> = node.neighbors.iter()
                .filter(|nid| !self.dead.contains(nid))
                .filter_map(|nid| snap.get(nid))
                .copied()
                .collect();
            let source1 = node.phi1_source(10_000);
            let sources = [source1, 0.0, 0.0, 0.0, 0.0];
            node.pheromone.step(&nbr_phis, sources, dt);
            node.reset_tick_counters();
        }
        self.tick += 1;
    }

    /// Store a pattern on all alive nodes.
    pub fn broadcast_store(&mut self, xi: &SparseVec) {
        let alive: Vec<NodeId> = self.alive_ids();
        let k = alive.len();
        for id in alive {
            if let Some(node) = self.get_node_mut(id) {
                node.store_pattern(xi, k.max(1));
            }
        }
    }
}
