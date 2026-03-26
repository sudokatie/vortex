//! Broad-phase collision detection using sweep-and-prune.

use std::collections::HashSet;

use crate::world::BodyHandle;
use super::Aabb;

/// Trait for broad-phase collision detection algorithms.
pub trait BroadPhase {
    /// Insert a new body with its AABB.
    fn insert(&mut self, handle: BodyHandle, aabb: Aabb);

    /// Remove a body.
    fn remove(&mut self, handle: BodyHandle);

    /// Update a body's AABB.
    fn update(&mut self, handle: BodyHandle, aabb: Aabb);

    /// Query all potentially colliding pairs.
    fn query_pairs(&self) -> Vec<(BodyHandle, BodyHandle)>;

    /// Clear all entries.
    fn clear(&mut self);
}

/// Entry in the sweep-and-prune structure.
#[derive(Debug, Clone)]
struct SapEntry {
    handle: BodyHandle,
    aabb: Aabb,
}

/// Sweep-and-prune broad phase.
/// 
/// Sorts bodies along one axis and only checks overlaps for bodies
/// that overlap on that axis.
#[derive(Debug, Default)]
pub struct SweepAndPrune {
    entries: Vec<SapEntry>,
    /// Which axis to sort on (0=X, 1=Y, 2=Z)
    axis: usize,
}

impl SweepAndPrune {
    /// Create a new sweep-and-prune structure.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            axis: 0,
        }
    }

    /// Set which axis to sort on (0=X, 1=Y, 2=Z).
    pub fn set_axis(&mut self, axis: usize) {
        self.axis = axis.min(2);
    }

    fn get_min(&self, aabb: &Aabb) -> f32 {
        match self.axis {
            0 => aabb.min.x,
            1 => aabb.min.y,
            _ => aabb.min.z,
        }
    }

    fn get_max(&self, aabb: &Aabb) -> f32 {
        match self.axis {
            0 => aabb.max.x,
            1 => aabb.max.y,
            _ => aabb.max.z,
        }
    }
}

impl BroadPhase for SweepAndPrune {
    fn insert(&mut self, handle: BodyHandle, aabb: Aabb) {
        self.entries.push(SapEntry { handle, aabb });
    }

    fn remove(&mut self, handle: BodyHandle) {
        self.entries.retain(|e| e.handle != handle);
    }

    fn update(&mut self, handle: BodyHandle, aabb: Aabb) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.handle == handle) {
            entry.aabb = aabb;
        }
    }

    fn query_pairs(&self) -> Vec<(BodyHandle, BodyHandle)> {
        if self.entries.len() < 2 {
            return Vec::new();
        }

        // Sort entries by min on the chosen axis
        let mut sorted: Vec<_> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            self.get_min(&a.aabb).partial_cmp(&self.get_min(&b.aabb)).unwrap()
        });

        let mut pairs = Vec::new();

        // Sweep through sorted list
        for i in 0..sorted.len() {
            let a = &sorted[i];
            let a_max = self.get_max(&a.aabb);

            // Check against all following entries until we pass a_max
            for j in (i + 1)..sorted.len() {
                let b = &sorted[j];
                let b_min = self.get_min(&b.aabb);

                // If b starts after a ends on this axis, no more overlaps possible
                if b_min > a_max {
                    break;
                }

                // Check full AABB overlap
                if a.aabb.intersects(&b.aabb) {
                    pairs.push((a.handle, b.handle));
                }
            }
        }

        pairs
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Spatial hash grid for uniform distributions.
#[derive(Debug)]
pub struct SpatialHash {
    cell_size: f32,
    cells: std::collections::HashMap<(i32, i32, i32), Vec<BodyHandle>>,
    handle_to_aabb: std::collections::HashMap<BodyHandle, Aabb>,
}

impl SpatialHash {
    /// Create a new spatial hash with the given cell size.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size: cell_size.max(0.1),
            cells: std::collections::HashMap::new(),
            handle_to_aabb: std::collections::HashMap::new(),
        }
    }

    fn cell_coords(&self, point: glam::Vec3) -> (i32, i32, i32) {
        (
            (point.x / self.cell_size).floor() as i32,
            (point.y / self.cell_size).floor() as i32,
            (point.z / self.cell_size).floor() as i32,
        )
    }

    fn cells_for_aabb(&self, aabb: &Aabb) -> Vec<(i32, i32, i32)> {
        let min = self.cell_coords(aabb.min);
        let max = self.cell_coords(aabb.max);

        let mut cells = Vec::new();
        for x in min.0..=max.0 {
            for y in min.1..=max.1 {
                for z in min.2..=max.2 {
                    cells.push((x, y, z));
                }
            }
        }
        cells
    }
}

impl BroadPhase for SpatialHash {
    fn insert(&mut self, handle: BodyHandle, aabb: Aabb) {
        self.handle_to_aabb.insert(handle, aabb);
        for cell in self.cells_for_aabb(&aabb) {
            self.cells.entry(cell).or_default().push(handle);
        }
    }

    fn remove(&mut self, handle: BodyHandle) {
        if let Some(aabb) = self.handle_to_aabb.remove(&handle) {
            for cell in self.cells_for_aabb(&aabb) {
                if let Some(handles) = self.cells.get_mut(&cell) {
                    handles.retain(|h| *h != handle);
                }
            }
        }
    }

    fn update(&mut self, handle: BodyHandle, aabb: Aabb) {
        self.remove(handle);
        self.insert(handle, aabb);
    }

    fn query_pairs(&self) -> Vec<(BodyHandle, BodyHandle)> {
        let mut pairs = HashSet::new();

        for handles in self.cells.values() {
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
                    let a = handles[i];
                    let b = handles[j];

                    // Ensure consistent ordering
                    let pair = if a < b { (a, b) } else { (b, a) };

                    // Only add if AABBs actually overlap
                    if let (Some(aabb_a), Some(aabb_b)) = 
                        (self.handle_to_aabb.get(&a), self.handle_to_aabb.get(&b)) 
                    {
                        if aabb_a.intersects(aabb_b) {
                            pairs.insert(pair);
                        }
                    }
                }
            }
        }

        pairs.into_iter().collect()
    }

    fn clear(&mut self) {
        self.cells.clear();
        self.handle_to_aabb.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use slotmap::SlotMap;

    fn make_handles(n: usize) -> (SlotMap<BodyHandle, ()>, Vec<BodyHandle>) {
        let mut map = SlotMap::with_key();
        let handles: Vec<_> = (0..n).map(|_| map.insert(())).collect();
        (map, handles)
    }

    #[test]
    fn test_sap_empty() {
        let sap = SweepAndPrune::new();
        assert!(sap.query_pairs().is_empty());
    }

    #[test]
    fn test_sap_no_overlap() {
        let mut sap = SweepAndPrune::new();
        let (_map, handles) = make_handles(2);

        sap.insert(handles[0], Aabb::new(Vec3::ZERO, Vec3::ONE));
        sap.insert(handles[1], Aabb::new(Vec3::splat(5.0), Vec3::splat(6.0)));

        assert!(sap.query_pairs().is_empty());
    }

    #[test]
    fn test_sap_overlap() {
        let mut sap = SweepAndPrune::new();
        let (_map, handles) = make_handles(2);

        sap.insert(handles[0], Aabb::new(Vec3::ZERO, Vec3::ONE));
        sap.insert(handles[1], Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5)));

        let pairs = sap.query_pairs();
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn test_sap_remove() {
        let mut sap = SweepAndPrune::new();
        let (_map, handles) = make_handles(2);

        sap.insert(handles[0], Aabb::new(Vec3::ZERO, Vec3::ONE));
        sap.insert(handles[1], Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5)));
        sap.remove(handles[1]);

        assert!(sap.query_pairs().is_empty());
    }

    #[test]
    fn test_sap_update() {
        let mut sap = SweepAndPrune::new();
        let (_map, handles) = make_handles(2);

        sap.insert(handles[0], Aabb::new(Vec3::ZERO, Vec3::ONE));
        sap.insert(handles[1], Aabb::new(Vec3::splat(5.0), Vec3::splat(6.0)));
        assert!(sap.query_pairs().is_empty());

        // Move second body to overlap
        sap.update(handles[1], Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5)));
        assert_eq!(sap.query_pairs().len(), 1);
    }

    #[test]
    fn test_spatial_hash_empty() {
        let hash = SpatialHash::new(1.0);
        assert!(hash.query_pairs().is_empty());
    }

    #[test]
    fn test_spatial_hash_no_overlap() {
        let mut hash = SpatialHash::new(1.0);
        let (_map, handles) = make_handles(2);

        hash.insert(handles[0], Aabb::new(Vec3::ZERO, Vec3::ONE));
        hash.insert(handles[1], Aabb::new(Vec3::splat(10.0), Vec3::splat(11.0)));

        assert!(hash.query_pairs().is_empty());
    }

    #[test]
    fn test_spatial_hash_overlap() {
        let mut hash = SpatialHash::new(1.0);
        let (_map, handles) = make_handles(2);

        hash.insert(handles[0], Aabb::new(Vec3::ZERO, Vec3::ONE));
        hash.insert(handles[1], Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5)));

        let pairs = hash.query_pairs();
        assert_eq!(pairs.len(), 1);
    }
}
