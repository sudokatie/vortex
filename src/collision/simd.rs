//! SIMD-accelerated collision primitives
//!
//! This module provides SIMD-optimized versions of collision detection functions
//! using platform intrinsics when available.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

use glam::Vec3;

// =============================================================================
// SIMD Vec3 operations (4-wide)
// =============================================================================

/// 4-wide SIMD vector for batched Vec3 operations.
/// Stores 4 Vec3s in SoA layout: [x0,x1,x2,x3], [y0,y1,y2,y3], [z0,z1,z2,z3]
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct SimdVec3x4 {
    pub x: [f32; 4],
    pub y: [f32; 4],
    pub z: [f32; 4],
}

impl SimdVec3x4 {
    /// Create from 4 Vec3s.
    #[inline]
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> Self {
        Self {
            x: [v0.x, v1.x, v2.x, v3.x],
            y: [v0.y, v1.y, v2.y, v3.y],
            z: [v0.z, v1.z, v2.z, v3.z],
        }
    }

    /// Create with all lanes set to same Vec3.
    #[inline]
    pub fn splat(v: Vec3) -> Self {
        Self {
            x: [v.x; 4],
            y: [v.y; 4],
            z: [v.z; 4],
        }
    }

    /// Extract Vec3 at index.
    #[inline]
    pub fn get(&self, i: usize) -> Vec3 {
        Vec3::new(self.x[i], self.y[i], self.z[i])
    }

    /// Dot product with another SimdVec3x4, returns 4 scalars.
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub fn dot(&self, other: &Self) -> [f32; 4] {
        unsafe {
            if is_x86_feature_detected!("sse") {
                self.dot_sse(other)
            } else {
                self.dot_scalar(other)
            }
        }
    }

    #[inline]
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    pub fn dot(&self, other: &Self) -> [f32; 4] {
        self.dot_scalar(other)
    }

    #[inline]
    fn dot_scalar(&self, other: &Self) -> [f32; 4] {
        [
            self.x[0] * other.x[0] + self.y[0] * other.y[0] + self.z[0] * other.z[0],
            self.x[1] * other.x[1] + self.y[1] * other.y[1] + self.z[1] * other.z[1],
            self.x[2] * other.x[2] + self.y[2] * other.y[2] + self.z[2] * other.z[2],
            self.x[3] * other.x[3] + self.y[3] * other.y[3] + self.z[3] * other.z[3],
        ]
    }

    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse")]
    unsafe fn dot_sse(&self, other: &Self) -> [f32; 4] {
        let x1 = _mm_loadu_ps(self.x.as_ptr());
        let y1 = _mm_loadu_ps(self.y.as_ptr());
        let z1 = _mm_loadu_ps(self.z.as_ptr());
        
        let x2 = _mm_loadu_ps(other.x.as_ptr());
        let y2 = _mm_loadu_ps(other.y.as_ptr());
        let z2 = _mm_loadu_ps(other.z.as_ptr());
        
        let xx = _mm_mul_ps(x1, x2);
        let yy = _mm_mul_ps(y1, y2);
        let zz = _mm_mul_ps(z1, z2);
        
        let result = _mm_add_ps(_mm_add_ps(xx, yy), zz);
        
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), result);
        out
    }

    /// Cross product, returns 4 Vec3s.
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub fn cross(&self, other: &Self) -> Self {
        unsafe {
            if is_x86_feature_detected!("sse") {
                self.cross_sse(other)
            } else {
                self.cross_scalar(other)
            }
        }
    }

    #[inline]
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    pub fn cross(&self, other: &Self) -> Self {
        self.cross_scalar(other)
    }

    #[inline]
    fn cross_scalar(&self, other: &Self) -> Self {
        Self {
            x: [
                self.y[0] * other.z[0] - self.z[0] * other.y[0],
                self.y[1] * other.z[1] - self.z[1] * other.y[1],
                self.y[2] * other.z[2] - self.z[2] * other.y[2],
                self.y[3] * other.z[3] - self.z[3] * other.y[3],
            ],
            y: [
                self.z[0] * other.x[0] - self.x[0] * other.z[0],
                self.z[1] * other.x[1] - self.x[1] * other.z[1],
                self.z[2] * other.x[2] - self.x[2] * other.z[2],
                self.z[3] * other.x[3] - self.x[3] * other.z[3],
            ],
            z: [
                self.x[0] * other.y[0] - self.y[0] * other.x[0],
                self.x[1] * other.y[1] - self.y[1] * other.x[1],
                self.x[2] * other.y[2] - self.y[2] * other.x[2],
                self.x[3] * other.y[3] - self.y[3] * other.x[3],
            ],
        }
    }

    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse")]
    unsafe fn cross_sse(&self, other: &Self) -> Self {
        let ax = _mm_loadu_ps(self.x.as_ptr());
        let ay = _mm_loadu_ps(self.y.as_ptr());
        let az = _mm_loadu_ps(self.z.as_ptr());
        
        let bx = _mm_loadu_ps(other.x.as_ptr());
        let by = _mm_loadu_ps(other.y.as_ptr());
        let bz = _mm_loadu_ps(other.z.as_ptr());
        
        // cross.x = a.y * b.z - a.z * b.y
        let cx = _mm_sub_ps(_mm_mul_ps(ay, bz), _mm_mul_ps(az, by));
        // cross.y = a.z * b.x - a.x * b.z
        let cy = _mm_sub_ps(_mm_mul_ps(az, bx), _mm_mul_ps(ax, bz));
        // cross.z = a.x * b.y - a.y * b.x
        let cz = _mm_sub_ps(_mm_mul_ps(ax, by), _mm_mul_ps(ay, bx));
        
        let mut result = Self { x: [0.0; 4], y: [0.0; 4], z: [0.0; 4] };
        _mm_storeu_ps(result.x.as_mut_ptr(), cx);
        _mm_storeu_ps(result.y.as_mut_ptr(), cy);
        _mm_storeu_ps(result.z.as_mut_ptr(), cz);
        result
    }

    /// Add two SimdVec3x4.
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub fn add(&self, other: &Self) -> Self {
        unsafe {
            if is_x86_feature_detected!("sse") {
                self.add_sse(other)
            } else {
                self.add_scalar(other)
            }
        }
    }

    #[inline]
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    pub fn add(&self, other: &Self) -> Self {
        self.add_scalar(other)
    }

    #[inline]
    fn add_scalar(&self, other: &Self) -> Self {
        Self {
            x: [self.x[0] + other.x[0], self.x[1] + other.x[1], self.x[2] + other.x[2], self.x[3] + other.x[3]],
            y: [self.y[0] + other.y[0], self.y[1] + other.y[1], self.y[2] + other.y[2], self.y[3] + other.y[3]],
            z: [self.z[0] + other.z[0], self.z[1] + other.z[1], self.z[2] + other.z[2], self.z[3] + other.z[3]],
        }
    }

    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse")]
    unsafe fn add_sse(&self, other: &Self) -> Self {
        let mut result = Self { x: [0.0; 4], y: [0.0; 4], z: [0.0; 4] };
        
        let x = _mm_add_ps(_mm_loadu_ps(self.x.as_ptr()), _mm_loadu_ps(other.x.as_ptr()));
        let y = _mm_add_ps(_mm_loadu_ps(self.y.as_ptr()), _mm_loadu_ps(other.y.as_ptr()));
        let z = _mm_add_ps(_mm_loadu_ps(self.z.as_ptr()), _mm_loadu_ps(other.z.as_ptr()));
        
        _mm_storeu_ps(result.x.as_mut_ptr(), x);
        _mm_storeu_ps(result.y.as_mut_ptr(), y);
        _mm_storeu_ps(result.z.as_mut_ptr(), z);
        result
    }

    /// Subtract two SimdVec3x4.
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub fn sub(&self, other: &Self) -> Self {
        unsafe {
            if is_x86_feature_detected!("sse") {
                self.sub_sse(other)
            } else {
                self.sub_scalar(other)
            }
        }
    }

    #[inline]
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    pub fn sub(&self, other: &Self) -> Self {
        self.sub_scalar(other)
    }

    #[inline]
    fn sub_scalar(&self, other: &Self) -> Self {
        Self {
            x: [self.x[0] - other.x[0], self.x[1] - other.x[1], self.x[2] - other.x[2], self.x[3] - other.x[3]],
            y: [self.y[0] - other.y[0], self.y[1] - other.y[1], self.y[2] - other.y[2], self.y[3] - other.y[3]],
            z: [self.z[0] - other.z[0], self.z[1] - other.z[1], self.z[2] - other.z[2], self.z[3] - other.z[3]],
        }
    }

    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse")]
    unsafe fn sub_sse(&self, other: &Self) -> Self {
        let mut result = Self { x: [0.0; 4], y: [0.0; 4], z: [0.0; 4] };
        
        let x = _mm_sub_ps(_mm_loadu_ps(self.x.as_ptr()), _mm_loadu_ps(other.x.as_ptr()));
        let y = _mm_sub_ps(_mm_loadu_ps(self.y.as_ptr()), _mm_loadu_ps(other.y.as_ptr()));
        let z = _mm_sub_ps(_mm_loadu_ps(self.z.as_ptr()), _mm_loadu_ps(other.z.as_ptr()));
        
        _mm_storeu_ps(result.x.as_mut_ptr(), x);
        _mm_storeu_ps(result.y.as_mut_ptr(), y);
        _mm_storeu_ps(result.z.as_mut_ptr(), z);
        result
    }

    /// Scale by scalar.
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        Self {
            x: [self.x[0] * s, self.x[1] * s, self.x[2] * s, self.x[3] * s],
            y: [self.y[0] * s, self.y[1] * s, self.y[2] * s, self.y[3] * s],
            z: [self.z[0] * s, self.z[1] * s, self.z[2] * s, self.z[3] * s],
        }
    }

    /// Length squared of each vector.
    #[inline]
    pub fn length_squared(&self) -> [f32; 4] {
        self.dot(self)
    }
}

// =============================================================================
// SIMD AABB operations
// =============================================================================

/// 4-wide SIMD AABB for batched intersection tests.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct SimdAabb4 {
    pub min_x: [f32; 4],
    pub min_y: [f32; 4],
    pub min_z: [f32; 4],
    pub max_x: [f32; 4],
    pub max_y: [f32; 4],
    pub max_z: [f32; 4],
}

impl SimdAabb4 {
    /// Test if single AABB intersects any of the 4 AABBs.
    /// Returns bitmask (bit i set if intersection with AABB i).
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub fn intersects_single(&self, min: Vec3, max: Vec3) -> u8 {
        unsafe {
            if is_x86_feature_detected!("sse") {
                self.intersects_single_sse(min, max)
            } else {
                self.intersects_single_scalar(min, max)
            }
        }
    }

    #[inline]
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    pub fn intersects_single(&self, min: Vec3, max: Vec3) -> u8 {
        self.intersects_single_scalar(min, max)
    }

    #[inline]
    fn intersects_single_scalar(&self, min: Vec3, max: Vec3) -> u8 {
        let mut result = 0u8;
        for i in 0..4 {
            let overlaps = 
                min.x <= self.max_x[i] && max.x >= self.min_x[i] &&
                min.y <= self.max_y[i] && max.y >= self.min_y[i] &&
                min.z <= self.max_z[i] && max.z >= self.min_z[i];
            if overlaps {
                result |= 1 << i;
            }
        }
        result
    }

    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse")]
    unsafe fn intersects_single_sse(&self, min: Vec3, max: Vec3) -> u8 {
        // Broadcast single AABB
        let min_x = _mm_set1_ps(min.x);
        let min_y = _mm_set1_ps(min.y);
        let min_z = _mm_set1_ps(min.z);
        let max_x = _mm_set1_ps(max.x);
        let max_y = _mm_set1_ps(max.y);
        let max_z = _mm_set1_ps(max.z);
        
        // Load 4 AABBs
        let self_min_x = _mm_loadu_ps(self.min_x.as_ptr());
        let self_min_y = _mm_loadu_ps(self.min_y.as_ptr());
        let self_min_z = _mm_loadu_ps(self.min_z.as_ptr());
        let self_max_x = _mm_loadu_ps(self.max_x.as_ptr());
        let self_max_y = _mm_loadu_ps(self.max_y.as_ptr());
        let self_max_z = _mm_loadu_ps(self.max_z.as_ptr());
        
        // Test overlap on each axis
        // overlap_x = (min.x <= self.max_x) && (max.x >= self.min_x)
        let x_overlap = _mm_and_ps(
            _mm_cmple_ps(min_x, self_max_x),
            _mm_cmpge_ps(max_x, self_min_x),
        );
        let y_overlap = _mm_and_ps(
            _mm_cmple_ps(min_y, self_max_y),
            _mm_cmpge_ps(max_y, self_min_y),
        );
        let z_overlap = _mm_and_ps(
            _mm_cmple_ps(min_z, self_max_z),
            _mm_cmpge_ps(max_z, self_min_z),
        );
        
        // All axes must overlap
        let overlap = _mm_and_ps(_mm_and_ps(x_overlap, y_overlap), z_overlap);
        
        // Extract mask
        _mm_movemask_ps(overlap) as u8
    }
}

// =============================================================================
// Batched GJK support functions
// =============================================================================

/// Compute 4 sphere support points simultaneously.
#[inline]
pub fn simd_sphere_support(radius: f32, directions: &SimdVec3x4) -> SimdVec3x4 {
    // Normalize directions and scale by radius
    let len_sq = directions.length_squared();
    
    SimdVec3x4 {
        x: [
            if len_sq[0] > 1e-10 { directions.x[0] / len_sq[0].sqrt() * radius } else { 0.0 },
            if len_sq[1] > 1e-10 { directions.x[1] / len_sq[1].sqrt() * radius } else { 0.0 },
            if len_sq[2] > 1e-10 { directions.x[2] / len_sq[2].sqrt() * radius } else { 0.0 },
            if len_sq[3] > 1e-10 { directions.x[3] / len_sq[3].sqrt() * radius } else { 0.0 },
        ],
        y: [
            if len_sq[0] > 1e-10 { directions.y[0] / len_sq[0].sqrt() * radius } else { 0.0 },
            if len_sq[1] > 1e-10 { directions.y[1] / len_sq[1].sqrt() * radius } else { 0.0 },
            if len_sq[2] > 1e-10 { directions.y[2] / len_sq[2].sqrt() * radius } else { 0.0 },
            if len_sq[3] > 1e-10 { directions.y[3] / len_sq[3].sqrt() * radius } else { 0.0 },
        ],
        z: [
            if len_sq[0] > 1e-10 { directions.z[0] / len_sq[0].sqrt() * radius } else { 0.0 },
            if len_sq[1] > 1e-10 { directions.z[1] / len_sq[1].sqrt() * radius } else { 0.0 },
            if len_sq[2] > 1e-10 { directions.z[2] / len_sq[2].sqrt() * radius } else { 0.0 },
            if len_sq[3] > 1e-10 { directions.z[3] / len_sq[3].sqrt() * radius } else { 0.0 },
        ],
    }
}

/// Compute 4 box support points simultaneously.
#[inline]
pub fn simd_box_support(half_extents: Vec3, directions: &SimdVec3x4) -> SimdVec3x4 {
    SimdVec3x4 {
        x: [
            if directions.x[0] >= 0.0 { half_extents.x } else { -half_extents.x },
            if directions.x[1] >= 0.0 { half_extents.x } else { -half_extents.x },
            if directions.x[2] >= 0.0 { half_extents.x } else { -half_extents.x },
            if directions.x[3] >= 0.0 { half_extents.x } else { -half_extents.x },
        ],
        y: [
            if directions.y[0] >= 0.0 { half_extents.y } else { -half_extents.y },
            if directions.y[1] >= 0.0 { half_extents.y } else { -half_extents.y },
            if directions.y[2] >= 0.0 { half_extents.y } else { -half_extents.y },
            if directions.y[3] >= 0.0 { half_extents.y } else { -half_extents.y },
        ],
        z: [
            if directions.z[0] >= 0.0 { half_extents.z } else { -half_extents.z },
            if directions.z[1] >= 0.0 { half_extents.z } else { -half_extents.z },
            if directions.z[2] >= 0.0 { half_extents.z } else { -half_extents.z },
            if directions.z[3] >= 0.0 { half_extents.z } else { -half_extents.z },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_vec3x4_dot() {
        let a = SimdVec3x4::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let b = SimdVec3x4::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 2.0, 3.0),
        );
        
        let result = a.dot(&b);
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 1.0).abs() < 0.001);
        assert!((result[2] - 2.0).abs() < 0.001);
        assert!((result[3] - 14.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_vec3x4_cross() {
        let a = SimdVec3x4::splat(Vec3::X);
        let b = SimdVec3x4::splat(Vec3::Y);
        
        let result = a.cross(&b);
        for i in 0..4 {
            let v = result.get(i);
            assert!((v - Vec3::Z).length() < 0.001);
        }
    }

    #[test]
    fn test_simd_aabb_intersects() {
        let aabbs = SimdAabb4 {
            min_x: [0.0, 10.0, 0.0, -5.0],
            min_y: [0.0, 10.0, 0.0, -5.0],
            min_z: [0.0, 10.0, 0.0, -5.0],
            max_x: [1.0, 11.0, 1.0, -4.0],
            max_y: [1.0, 11.0, 1.0, -4.0],
            max_z: [1.0, 11.0, 1.0, -4.0],
        };
        
        // Test AABB that overlaps [0] and [2]
        let mask = aabbs.intersects_single(
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.5, 1.5, 1.5),
        );
        
        assert!(mask & 0b0001 != 0); // overlaps [0]
        assert!(mask & 0b0010 == 0); // no overlap [1]
        assert!(mask & 0b0100 != 0); // overlaps [2]
        assert!(mask & 0b1000 == 0); // no overlap [3]
    }

    #[test]
    fn test_simd_sphere_support() {
        let dirs = SimdVec3x4::new(
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0).normalize(),
        );
        
        let support = simd_sphere_support(2.0, &dirs);
        
        assert!((support.get(0) - Vec3::new(2.0, 0.0, 0.0)).length() < 0.001);
        assert!((support.get(1) - Vec3::new(0.0, 2.0, 0.0)).length() < 0.001);
        assert!((support.get(2) - Vec3::new(0.0, 0.0, 2.0)).length() < 0.001);
    }

    #[test]
    fn test_simd_box_support() {
        let half = Vec3::new(1.0, 2.0, 3.0);
        let dirs = SimdVec3x4::new(
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        
        let support = simd_box_support(half, &dirs);
        
        assert!((support.get(0) - Vec3::new(1.0, 2.0, 3.0)).length() < 0.001);
        assert!((support.get(1) - Vec3::new(-1.0, 2.0, 3.0)).length() < 0.001);
        assert!((support.get(2) - Vec3::new(1.0, -2.0, -3.0)).length() < 0.001);
        assert!((support.get(3) - Vec3::new(-1.0, -2.0, -3.0)).length() < 0.001);
    }
}
