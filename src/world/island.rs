// Island detection for physics simulation

use std::collections::HashSet;

/// Represents a group of interacting bodies
#[derive(Debug, Clone)]
pub struct Island {
    /// Body indices in this island
    pub bodies: Vec<u32>,
    /// Contact indices in this island
    pub contacts: Vec<usize>,
    /// Joint indices in this island
    pub joints: Vec<usize>,
    /// Whether island can sleep
    pub can_sleep: bool,
}

impl Island {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            contacts: Vec::new(),
            joints: Vec::new(),
            can_sleep: true,
        }
    }
    
    pub fn add_body(&mut self, body: u32) {
        self.bodies.push(body);
    }
    
    pub fn add_contact(&mut self, contact_idx: usize) {
        self.contacts.push(contact_idx);
    }
    
    pub fn add_joint(&mut self, joint_idx: usize) {
        self.joints.push(joint_idx);
    }
    
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }
    
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }
}

impl Default for Island {
    fn default() -> Self {
        Self::new()
    }
}

/// Contact pair for island building
#[derive(Debug, Clone, Copy)]
pub struct ContactPair {
    pub body_a: u32,
    pub body_b: u32,
}

/// Joint pair for island building  
#[derive(Debug, Clone, Copy)]
pub struct JointPair {
    pub body_a: u32,
    pub body_b: u32,
}

/// Island detector using flood-fill algorithm
pub struct IslandDetector {
    /// Bodies that have been visited
    visited: HashSet<u32>,
    /// Stack for flood-fill
    stack: Vec<u32>,
}

impl IslandDetector {
    pub fn new() -> Self {
        Self {
            visited: HashSet::new(),
            stack: Vec::new(),
        }
    }
    
    /// Find all islands given contacts and joints
    pub fn find_islands(
        &mut self,
        body_count: usize,
        is_static: impl Fn(u32) -> bool,
        contacts: &[ContactPair],
        joints: &[JointPair],
    ) -> Vec<Island> {
        self.visited.clear();
        let mut islands = Vec::new();
        
        // Build adjacency list
        let adjacency = self.build_adjacency(body_count, contacts, joints);
        
        // Find islands starting from each unvisited dynamic body
        for body in 0..body_count as u32 {
            if self.visited.contains(&body) || is_static(body) {
                continue;
            }
            
            let island = self.flood_fill(
                body,
                &adjacency,
                &is_static,
                contacts,
                joints,
            );
            
            if !island.is_empty() {
                islands.push(island);
            }
        }
        
        islands
    }
    
    fn build_adjacency(
        &self,
        body_count: usize,
        contacts: &[ContactPair],
        joints: &[JointPair],
    ) -> Vec<Vec<u32>> {
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); body_count];
        
        for contact in contacts {
            adjacency[contact.body_a as usize].push(contact.body_b);
            adjacency[contact.body_b as usize].push(contact.body_a);
        }
        
        for joint in joints {
            adjacency[joint.body_a as usize].push(joint.body_b);
            adjacency[joint.body_b as usize].push(joint.body_a);
        }
        
        adjacency
    }
    
    fn flood_fill(
        &mut self,
        start: u32,
        adjacency: &[Vec<u32>],
        is_static: &impl Fn(u32) -> bool,
        contacts: &[ContactPair],
        joints: &[JointPair],
    ) -> Island {
        let mut island = Island::new();
        
        self.stack.clear();
        self.stack.push(start);
        
        while let Some(body) = self.stack.pop() {
            if self.visited.contains(&body) {
                continue;
            }
            
            self.visited.insert(body);
            
            // Skip static bodies but continue flood-fill
            if !is_static(body) {
                island.add_body(body);
            }
            
            // Add neighbors to stack
            for &neighbor in &adjacency[body as usize] {
                if !self.visited.contains(&neighbor) {
                    self.stack.push(neighbor);
                }
            }
        }
        
        // Find contacts in this island
        for (idx, contact) in contacts.iter().enumerate() {
            if island.bodies.contains(&contact.body_a) || 
               island.bodies.contains(&contact.body_b) {
                island.add_contact(idx);
            }
        }
        
        // Find joints in this island
        for (idx, joint) in joints.iter().enumerate() {
            if island.bodies.contains(&joint.body_a) ||
               island.bodies.contains(&joint.body_b) {
                island.add_joint(idx);
            }
        }
        
        island
    }
    
    /// Clear internal state
    pub fn clear(&mut self) {
        self.visited.clear();
        self.stack.clear();
    }
}

impl Default for IslandDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_island_new() {
        let island = Island::new();
        assert!(island.is_empty());
        assert_eq!(island.body_count(), 0);
    }

    #[test]
    fn test_island_add_body() {
        let mut island = Island::new();
        island.add_body(0);
        island.add_body(1);
        assert_eq!(island.body_count(), 2);
    }

    #[test]
    fn test_detector_no_contacts() {
        let mut detector = IslandDetector::new();
        let islands = detector.find_islands(3, |_| false, &[], &[]);
        
        // Each body is its own island
        assert_eq!(islands.len(), 3);
    }

    #[test]
    fn test_detector_connected() {
        let mut detector = IslandDetector::new();
        let contacts = vec![
            ContactPair { body_a: 0, body_b: 1 },
            ContactPair { body_a: 1, body_b: 2 },
        ];
        
        let islands = detector.find_islands(3, |_| false, &contacts, &[]);
        
        // All bodies in one island
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].body_count(), 3);
    }

    #[test]
    fn test_detector_two_islands() {
        let mut detector = IslandDetector::new();
        let contacts = vec![
            ContactPair { body_a: 0, body_b: 1 },
            ContactPair { body_a: 2, body_b: 3 },
        ];
        
        let islands = detector.find_islands(4, |_| false, &contacts, &[]);
        
        assert_eq!(islands.len(), 2);
    }

    #[test]
    fn test_detector_static_body() {
        let mut detector = IslandDetector::new();
        let contacts = vec![
            ContactPair { body_a: 0, body_b: 1 },
        ];
        
        // Body 0 is static
        let islands = detector.find_islands(2, |b| b == 0, &contacts, &[]);
        
        // Only body 1 in island
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].body_count(), 1);
        assert!(islands[0].bodies.contains(&1));
    }

    #[test]
    fn test_detector_with_joints() {
        let mut detector = IslandDetector::new();
        let joints = vec![
            JointPair { body_a: 0, body_b: 1 },
        ];
        
        let islands = detector.find_islands(2, |_| false, &[], &joints);
        
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].joints.len(), 1);
    }

    #[test]
    fn test_detector_contacts_in_island() {
        let mut detector = IslandDetector::new();
        let contacts = vec![
            ContactPair { body_a: 0, body_b: 1 },
            ContactPair { body_a: 1, body_b: 2 },
        ];
        
        let islands = detector.find_islands(3, |_| false, &contacts, &[]);
        
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].contacts.len(), 2);
    }
}
