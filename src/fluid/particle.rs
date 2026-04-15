//! Fluid particle and spatial hashing grid for SPH simulation.
//!
//! This module provides:
//! - `FluidParticle`: Individual fluid particle with position, velocity, and SPH properties
//! - `ParticleGrid`: Spatial hash grid for O(n) neighbor queries

use glam::Vec3;
use std::collections::HashMap;

/// A fluid particle for SPH simulation.
///
/// Each particle carries kinematic state (position, velocity, acceleration)
/// and SPH-computed properties (density, pressure).
#[derive(Debug, Clone, Copy)]
pub struct FluidParticle {
    /// Position in world space
    pub position: Vec3,
    /// Velocity
    pub velocity: Vec3,
    /// Acceleration (accumulated forces / mass)
    pub acceleration: Vec3,
    /// Density computed during SPH step
    pub density: f64,
    /// Pressure computed during SPH step
    pub pressure: f64,
    /// Particle mass
    pub mass: f64,
}

impl FluidParticle {
    /// Create a new fluid particle at the given position with the specified mass.
    ///
    /// Velocity, acceleration, density, and pressure are initialized to zero.
    pub fn new(position: Vec3, mass: f64) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            density: 0.0,
            pressure: 0.0,
            mass,
        }
    }

    /// Create a new fluid particle with initial velocity.
    pub fn with_velocity(position: Vec3, velocity: Vec3, mass: f64) -> Self {
        Self {
            position,
            velocity,
            acceleration: Vec3::ZERO,
            density: 0.0,
            pressure: 0.0,
            mass,
        }
    }
}

impl Default for FluidParticle {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            density: 0.0,
            pressure: 0.0,
            mass: 1.0,
        }
    }
}

/// Spatial hash grid for efficient neighbor queries.
///
/// Divides space into a uniform grid where each cell contains indices of particles
/// within that cell. Neighbor queries check only the 27 surrounding cells (3x3x3),
/// providing O(n) complexity for building and querying.
#[derive(Debug, Clone)]
pub struct ParticleGrid {
    /// Cell size (typically equal to smoothing radius)
    cell_size: f64,
    /// Inverse cell size for fast coordinate computation
    inv_cell_size: f64,
    /// Map from grid cell coordinates to particle indices
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl ParticleGrid {
    /// Create a new particle grid with the given cell size.
    ///
    /// The cell size should typically be equal to the SPH smoothing radius
    /// to ensure all neighbors within the support radius are found.
    ///
    /// # Panics
    /// Panics if `cell_size <= 0.0`.
    pub fn new(cell_size: f64) -> Self {
        assert!(cell_size > 0.0, "Cell size must be positive");
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            grid: HashMap::new(),
        }
    }

    /// Convert a world position to grid cell coordinates.
    #[inline]
    pub fn cell_coord(&self, pos: Vec3) -> (i32, i32, i32) {
        (
            (pos.x as f64 * self.inv_cell_size).floor() as i32,
            (pos.y as f64 * self.inv_cell_size).floor() as i32,
            (pos.z as f64 * self.inv_cell_size).floor() as i32,
        )
    }

    /// Insert a particle index at the given position.
    pub fn insert(&mut self, index: usize, position: Vec3) {
        let cell = self.cell_coord(position);
        self.grid.entry(cell).or_insert_with(Vec::new).push(index);
    }

    /// Query all particle indices within the given radius of a position.
    ///
    /// Checks the 27 surrounding cells (3x3x3) centered on the query position's cell.
    /// Returns indices of all particles in those cells that are within `radius` distance.
    pub fn query(&self, position: Vec3, radius: f64) -> Vec<usize> {
        let (cx, cy, cz) = self.cell_coord(position);
        let mut result = Vec::new();

        // Check 3x3x3 neighborhood
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let cell = (cx + dx, cy + dy, cz + dz);
                    if let Some(indices) = self.grid.get(&cell) {
                        result.extend(indices.iter().copied());
                    }
                }
            }
        }

        result
    }

    /// Query neighbors with distance filtering.
    ///
    /// Like `query`, but also filters out particles that are farther than `radius`
    /// from the query position. Requires access to particle positions.
    pub fn query_with_positions(
        &self,
        position: Vec3,
        radius: f64,
        positions: &[Vec3],
    ) -> Vec<usize> {
        let radius_sq = radius * radius;
        let (cx, cy, cz) = self.cell_coord(position);
        let mut result = Vec::new();

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let cell = (cx + dx, cy + dy, cz + dz);
                    if let Some(indices) = self.grid.get(&cell) {
                        for &idx in indices {
                            let diff = positions[idx] - position;
                            if (diff.length_squared() as f64) <= radius_sq {
                                result.push(idx);
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Clear the grid for the next simulation step.
    pub fn clear(&mut self) {
        self.grid.clear();
    }

    /// Get the cell size.
    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// Get the number of occupied cells.
    pub fn num_cells(&self) -> usize {
        self.grid.len()
    }

    /// Get the total number of particle entries (may include duplicates if particles moved).
    pub fn num_entries(&self) -> usize {
        self.grid.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    // ==================== FluidParticle Tests ====================

    #[test]
    fn particle_new_initializes_correctly() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let mass = 0.5;
        let p = FluidParticle::new(pos, mass);

        assert!((p.position - pos).length() < EPSILON);
        assert_eq!(p.velocity, Vec3::ZERO);
        assert_eq!(p.acceleration, Vec3::ZERO);
        assert_eq!(p.density, 0.0);
        assert_eq!(p.pressure, 0.0);
        assert_eq!(p.mass, mass);
    }

    #[test]
    fn particle_with_velocity_initializes_correctly() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let vel = Vec3::new(0.1, 0.2, 0.3);
        let mass = 2.0;
        let p = FluidParticle::with_velocity(pos, vel, mass);

        assert!((p.position - pos).length() < EPSILON);
        assert!((p.velocity - vel).length() < EPSILON);
        assert_eq!(p.acceleration, Vec3::ZERO);
        assert_eq!(p.mass, mass);
    }

    #[test]
    fn particle_default_has_unit_mass() {
        let p = FluidParticle::default();
        assert_eq!(p.position, Vec3::ZERO);
        assert_eq!(p.mass, 1.0);
    }

    // ==================== ParticleGrid Construction Tests ====================

    #[test]
    fn grid_new_with_valid_cell_size() {
        let grid = ParticleGrid::new(1.0);
        assert_eq!(grid.cell_size(), 1.0);
        assert_eq!(grid.num_cells(), 0);
    }

    #[test]
    #[should_panic(expected = "Cell size must be positive")]
    fn grid_panics_on_zero_cell_size() {
        ParticleGrid::new(0.0);
    }

    #[test]
    #[should_panic(expected = "Cell size must be positive")]
    fn grid_panics_on_negative_cell_size() {
        ParticleGrid::new(-1.0);
    }

    // ==================== Cell Coordinate Tests ====================

    #[test]
    fn cell_coord_positive_positions() {
        let grid = ParticleGrid::new(1.0);

        assert_eq!(grid.cell_coord(Vec3::new(0.5, 0.5, 0.5)), (0, 0, 0));
        assert_eq!(grid.cell_coord(Vec3::new(1.5, 2.5, 3.5)), (1, 2, 3));
        assert_eq!(grid.cell_coord(Vec3::new(0.0, 0.0, 0.0)), (0, 0, 0));
        assert_eq!(grid.cell_coord(Vec3::new(0.99, 0.99, 0.99)), (0, 0, 0));
        assert_eq!(grid.cell_coord(Vec3::new(1.0, 1.0, 1.0)), (1, 1, 1));
    }

    #[test]
    fn cell_coord_negative_positions() {
        let grid = ParticleGrid::new(1.0);

        assert_eq!(grid.cell_coord(Vec3::new(-0.5, -0.5, -0.5)), (-1, -1, -1));
        assert_eq!(grid.cell_coord(Vec3::new(-1.5, -2.5, -3.5)), (-2, -3, -4));
        assert_eq!(grid.cell_coord(Vec3::new(-1.0, -1.0, -1.0)), (-1, -1, -1));
    }

    #[test]
    fn cell_coord_with_different_cell_size() {
        let grid = ParticleGrid::new(0.5);

        assert_eq!(grid.cell_coord(Vec3::new(0.25, 0.25, 0.25)), (0, 0, 0));
        assert_eq!(grid.cell_coord(Vec3::new(0.5, 0.5, 0.5)), (1, 1, 1));
        assert_eq!(grid.cell_coord(Vec3::new(1.0, 1.0, 1.0)), (2, 2, 2));
    }

    // ==================== Insert and Query Tests ====================

    #[test]
    fn insert_and_query_single_particle() {
        let mut grid = ParticleGrid::new(1.0);
        let pos = Vec3::new(0.5, 0.5, 0.5);

        grid.insert(0, pos);

        let neighbors = grid.query(pos, 1.0);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&0));
    }

    #[test]
    fn insert_multiple_particles_same_cell() {
        let mut grid = ParticleGrid::new(1.0);

        grid.insert(0, Vec3::new(0.1, 0.1, 0.1));
        grid.insert(1, Vec3::new(0.2, 0.2, 0.2));
        grid.insert(2, Vec3::new(0.9, 0.9, 0.9));

        let neighbors = grid.query(Vec3::new(0.5, 0.5, 0.5), 1.0);
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
    }

    #[test]
    fn query_finds_neighbors_in_adjacent_cells() {
        let mut grid = ParticleGrid::new(1.0);

        // Place particles in different cells around the origin
        grid.insert(0, Vec3::new(0.5, 0.5, 0.5));   // cell (0,0,0)
        grid.insert(1, Vec3::new(1.5, 0.5, 0.5));   // cell (1,0,0)
        grid.insert(2, Vec3::new(-0.5, 0.5, 0.5));  // cell (-1,0,0)
        grid.insert(3, Vec3::new(0.5, 1.5, 0.5));   // cell (0,1,0)
        grid.insert(4, Vec3::new(0.5, 0.5, 1.5));   // cell (0,0,1)

        // Query from center of cell (0,0,0) should find all 5
        let neighbors = grid.query(Vec3::new(0.5, 0.5, 0.5), 2.0);
        assert_eq!(neighbors.len(), 5);
    }

    #[test]
    fn query_does_not_find_distant_particles() {
        let mut grid = ParticleGrid::new(1.0);

        // Place particles far apart
        grid.insert(0, Vec3::new(0.5, 0.5, 0.5));   // cell (0,0,0)
        grid.insert(1, Vec3::new(5.5, 5.5, 5.5));   // cell (5,5,5) - far away

        // Query from cell (0,0,0) should only find particle 0
        let neighbors = grid.query(Vec3::new(0.5, 0.5, 0.5), 1.0);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&0));
        assert!(!neighbors.contains(&1));
    }

    #[test]
    fn query_with_positions_filters_by_distance() {
        let mut grid = ParticleGrid::new(1.0);
        let positions = vec![
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.6, 0.5, 0.5),   // very close
            Vec3::new(1.4, 0.5, 0.5),   // in adjacent cell but within radius
            Vec3::new(1.9, 0.5, 0.5),   // in adjacent cell but outside radius
        ];

        for (i, &pos) in positions.iter().enumerate() {
            grid.insert(i, pos);
        }

        let query_pos = Vec3::new(0.5, 0.5, 0.5);
        let neighbors = grid.query_with_positions(query_pos, 1.0, &positions);

        assert!(neighbors.contains(&0)); // distance 0.0
        assert!(neighbors.contains(&1)); // distance 0.1
        assert!(neighbors.contains(&2)); // distance 0.9
        assert!(!neighbors.contains(&3)); // distance 1.4 > 1.0
    }

    // ==================== Clear Tests ====================

    #[test]
    fn clear_removes_all_entries() {
        let mut grid = ParticleGrid::new(1.0);

        grid.insert(0, Vec3::new(0.5, 0.5, 0.5));
        grid.insert(1, Vec3::new(1.5, 1.5, 1.5));
        grid.insert(2, Vec3::new(2.5, 2.5, 2.5));

        assert!(grid.num_cells() > 0);
        assert_eq!(grid.num_entries(), 3);

        grid.clear();

        assert_eq!(grid.num_cells(), 0);
        assert_eq!(grid.num_entries(), 0);

        let neighbors = grid.query(Vec3::new(0.5, 0.5, 0.5), 1.0);
        assert!(neighbors.is_empty());
    }

    // ==================== 3x3x3 Neighborhood Tests ====================

    #[test]
    fn query_checks_all_27_cells() {
        let mut grid = ParticleGrid::new(1.0);

        // Place one particle in each of the 27 cells around the origin cell
        let mut idx = 0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let pos = Vec3::new(
                        dx as f32 + 0.5,
                        dy as f32 + 0.5,
                        dz as f32 + 0.5,
                    );
                    grid.insert(idx, pos);
                    idx += 1;
                }
            }
        }

        assert_eq!(grid.num_cells(), 27);
        assert_eq!(grid.num_entries(), 27);

        // Query from center should find all 27 particles
        let neighbors = grid.query(Vec3::new(0.5, 0.5, 0.5), 3.0);
        assert_eq!(neighbors.len(), 27);
    }

    #[test]
    fn query_empty_grid_returns_empty() {
        let grid = ParticleGrid::new(1.0);
        let neighbors = grid.query(Vec3::new(0.0, 0.0, 0.0), 1.0);
        assert!(neighbors.is_empty());
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn grid_handles_large_coordinates() {
        let mut grid = ParticleGrid::new(1.0);

        let pos = Vec3::new(1000.5, 2000.5, 3000.5);
        grid.insert(0, pos);

        let neighbors = grid.query(pos, 1.0);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&0));
    }

    #[test]
    fn grid_handles_small_cell_size() {
        let mut grid = ParticleGrid::new(0.01);

        grid.insert(0, Vec3::new(0.005, 0.005, 0.005));
        grid.insert(1, Vec3::new(0.015, 0.005, 0.005));

        // These should be in adjacent cells
        assert_eq!(grid.cell_coord(Vec3::new(0.005, 0.005, 0.005)), (0, 0, 0));
        assert_eq!(grid.cell_coord(Vec3::new(0.015, 0.005, 0.005)), (1, 0, 0));

        let neighbors = grid.query(Vec3::new(0.005, 0.005, 0.005), 0.02);
        assert_eq!(neighbors.len(), 2);
    }
}
