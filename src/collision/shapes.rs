//! Collision shapes for rigid bodies.

use glam::{Mat3, Vec3};

use super::Aabb;

/// A face of a convex mesh.
#[derive(Debug, Clone)]
pub struct Face {
    /// Vertex indices forming the face.
    pub indices: Vec<usize>,
    /// Face normal.
    pub normal: Vec3,
}

/// A triangle for mesh colliders.
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    /// The three vertices of the triangle.
    pub vertices: [Vec3; 3],
    /// Precomputed normal (optional, computed lazily if zero).
    normal: Vec3,
}

impl Triangle {
    /// Create a new triangle from three vertices.
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2).normalize_or_zero();
        Self {
            vertices: [v0, v1, v2],
            normal,
        }
    }

    /// Create a triangle from a vertex array.
    pub fn from_array(vertices: [Vec3; 3]) -> Self {
        Self::new(vertices[0], vertices[1], vertices[2])
    }

    /// Get the triangle normal.
    pub fn normal(&self) -> Vec3 {
        self.normal
    }

    /// Get the centroid of the triangle.
    pub fn centroid(&self) -> Vec3 {
        (self.vertices[0] + self.vertices[1] + self.vertices[2]) / 3.0
    }

    /// Get the support point in a given direction (for GJK).
    pub fn support(&self, direction: Vec3) -> Vec3 {
        let d0 = self.vertices[0].dot(direction);
        let d1 = self.vertices[1].dot(direction);
        let d2 = self.vertices[2].dot(direction);

        if d0 >= d1 && d0 >= d2 {
            self.vertices[0]
        } else if d1 >= d2 {
            self.vertices[1]
        } else {
            self.vertices[2]
        }
    }

    /// Get the AABB of this triangle.
    pub fn aabb(&self) -> Aabb {
        let min = self.vertices[0].min(self.vertices[1]).min(self.vertices[2]);
        let max = self.vertices[0].max(self.vertices[1]).max(self.vertices[2]);
        Aabb::new(min, max)
    }
}

impl Face {
    /// Create a new face.
    pub fn new(indices: Vec<usize>, normal: Vec3) -> Self {
        Self { indices, normal }
    }
    
    /// Create a triangular face and compute normal from vertices.
    pub fn triangle(vertices: &[Vec3], i0: usize, i1: usize, i2: usize) -> Self {
        let a = vertices[i0];
        let b = vertices[i1];
        let c = vertices[i2];
        let normal = (b - a).cross(c - a).normalize_or_zero();
        Self {
            indices: vec![i0, i1, i2],
            normal,
        }
    }
}

/// Collision shape types.
#[derive(Debug, Clone)]
pub enum CollisionShape {
    /// Sphere with radius.
    Sphere { radius: f32 },
    /// Box with half-extents.
    Box { half_extents: Vec3 },
    /// Capsule (cylinder with hemispherical caps).
    Capsule { radius: f32, half_height: f32 },
    /// Convex hull mesh.
    Convex { vertices: Vec<Vec3>, faces: Vec<Face> },
    /// Triangle mesh collider.
    /// For collision detection, each triangle is treated as a convex shape.
    Mesh { triangles: Vec<Triangle> },
}

impl CollisionShape {
    /// Create a sphere shape.
    pub fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    /// Create a box shape from half-extents.
    pub fn cube(half_extents: Vec3) -> Self {
        Self::Box { half_extents }
    }

    /// Create a capsule shape.
    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self::Capsule { radius, half_height }
    }
    
    /// Create a convex hull from vertices.
    pub fn convex(vertices: Vec<Vec3>, faces: Vec<Face>) -> Self {
        Self::Convex { vertices, faces }
    }
    
    /// Create a convex hull from vertices, computing faces automatically.
    /// Uses a simple convex hull algorithm for small vertex sets.
    pub fn convex_hull(vertices: Vec<Vec3>) -> Self {
        if vertices.len() < 4 {
            // Degenerate - return as-is with no faces
            return Self::Convex { vertices, faces: Vec::new() };
        }

        // Simple convex hull - for small meshes, compute faces
        // For a proper implementation, use quickhull or similar
        let faces = compute_convex_faces(&vertices);
        Self::Convex { vertices, faces }
    }

    /// Create a mesh collider from a list of triangles.
    pub fn mesh(triangles: Vec<Triangle>) -> Self {
        Self::Mesh { triangles }
    }

    /// Create a mesh collider from vertices and triangle indices.
    ///
    /// Each group of 3 indices forms a triangle.
    pub fn from_vertices_and_indices(vertices: &[Vec3], indices: &[usize]) -> Self {
        assert!(indices.len() % 3 == 0, "Indices must be a multiple of 3");

        let triangles: Vec<Triangle> = indices
            .chunks(3)
            .map(|tri| {
                Triangle::new(
                    vertices[tri[0]],
                    vertices[tri[1]],
                    vertices[tri[2]],
                )
            })
            .collect();

        Self::Mesh { triangles }
    }

    /// Create a mesh collider from raw triangle data.
    pub fn from_triangles(triangle_vertices: &[[Vec3; 3]]) -> Self {
        let triangles: Vec<Triangle> = triangle_vertices
            .iter()
            .map(|&verts| Triangle::from_array(verts))
            .collect();

        Self::Mesh { triangles }
    }

    /// Get the triangles if this is a mesh shape.
    pub fn triangles(&self) -> Option<&[Triangle]> {
        match self {
            Self::Mesh { triangles } => Some(triangles),
            _ => None,
        }
    }

    /// Decompose a mesh into a collection of convex shapes using octree subdivision.
    ///
    /// This is an approximate decomposition that:
    /// 1. Computes the AABB of the mesh
    /// 2. Subdivides the AABB into octree cells up to the specified depth
    /// 3. For each cell containing triangles, creates a convex hull of the triangle vertices
    ///
    /// Returns an empty vector if this is not a mesh or the mesh is empty.
    pub fn convex_decompose(&self, max_depth: u32) -> Vec<CollisionShape> {
        let triangles = match self {
            Self::Mesh { triangles } => triangles,
            _ => return Vec::new(),
        };

        if triangles.is_empty() {
            return Vec::new();
        }

        convex_decompose_mesh(triangles, max_depth)
    }

    /// Get the support point in a given direction (for GJK).
    pub fn support(&self, direction: Vec3) -> Vec3 {
        match self {
            Self::Sphere { radius } => direction.normalize_or_zero() * *radius,
            Self::Box { half_extents } => Vec3::new(
                if direction.x >= 0.0 { half_extents.x } else { -half_extents.x },
                if direction.y >= 0.0 { half_extents.y } else { -half_extents.y },
                if direction.z >= 0.0 { half_extents.z } else { -half_extents.z },
            ),
            Self::Capsule { radius, half_height } => {
                let cap_center = if direction.y >= 0.0 {
                    Vec3::new(0.0, *half_height, 0.0)
                } else {
                    Vec3::new(0.0, -*half_height, 0.0)
                };
                cap_center + direction.normalize_or_zero() * *radius
            }
            Self::Convex { vertices, .. } => {
                // Find vertex with maximum dot product in direction
                if vertices.is_empty() {
                    return Vec3::ZERO;
                }
                let mut best = vertices[0];
                let mut best_dot = best.dot(direction);
                for &v in vertices.iter().skip(1) {
                    let d = v.dot(direction);
                    if d > best_dot {
                        best_dot = d;
                        best = v;
                    }
                }
                best
            }
            Self::Mesh { triangles } => {
                // Find vertex with maximum dot product across all triangles
                if triangles.is_empty() {
                    return Vec3::ZERO;
                }
                let mut best = triangles[0].vertices[0];
                let mut best_dot = best.dot(direction);

                for tri in triangles {
                    for &v in &tri.vertices {
                        let d = v.dot(direction);
                        if d > best_dot {
                            best_dot = d;
                            best = v;
                        }
                    }
                }
                best
            }
        }
    }

    /// Get the local-space axis-aligned bounding box.
    pub fn local_aabb(&self) -> Aabb {
        match self {
            Self::Sphere { radius } => Aabb::new(
                Vec3::splat(-*radius),
                Vec3::splat(*radius),
            ),
            Self::Box { half_extents } => Aabb::new(
                -*half_extents,
                *half_extents,
            ),
            Self::Capsule { radius, half_height } => {
                let extent = Vec3::new(*radius, *half_height + *radius, *radius);
                Aabb::new(-extent, extent)
            }
            Self::Convex { vertices, .. } => {
                if vertices.is_empty() {
                    return Aabb::new(Vec3::ZERO, Vec3::ZERO);
                }
                let mut min = vertices[0];
                let mut max = vertices[0];
                for v in vertices.iter().skip(1) {
                    min = min.min(*v);
                    max = max.max(*v);
                }
                Aabb::new(min, max)
            }
            Self::Mesh { triangles } => {
                if triangles.is_empty() {
                    return Aabb::new(Vec3::ZERO, Vec3::ZERO);
                }
                let mut min = triangles[0].vertices[0];
                let mut max = triangles[0].vertices[0];

                for tri in triangles {
                    for &v in &tri.vertices {
                        min = min.min(v);
                        max = max.max(v);
                    }
                }
                Aabb::new(min, max)
            }
        }
    }

    /// Get the center of mass (always origin for these primitives).
    pub fn center_of_mass(&self) -> Vec3 {
        match self {
            Self::Convex { vertices, .. } => {
                if vertices.is_empty() {
                    return Vec3::ZERO;
                }
                let sum: Vec3 = vertices.iter().copied().sum();
                sum / vertices.len() as f32
            }
            Self::Mesh { triangles } => {
                if triangles.is_empty() {
                    return Vec3::ZERO;
                }
                // Weighted average of triangle centroids by area
                let mut total_area = 0.0f32;
                let mut weighted_sum = Vec3::ZERO;

                for tri in triangles {
                    let edge1 = tri.vertices[1] - tri.vertices[0];
                    let edge2 = tri.vertices[2] - tri.vertices[0];
                    let area = edge1.cross(edge2).length() * 0.5;
                    total_area += area;
                    weighted_sum += tri.centroid() * area;
                }

                if total_area > 1e-10 {
                    weighted_sum / total_area
                } else {
                    Vec3::ZERO
                }
            }
            _ => Vec3::ZERO,
        }
    }

    /// Calculate the inertia tensor for a given mass.
    pub fn inertia_tensor(&self, mass: f32) -> Mat3 {
        match self {
            Self::Sphere { radius } => {
                // I = (2/5) * m * r^2 for solid sphere
                let i = (2.0 / 5.0) * mass * radius * radius;
                Mat3::from_diagonal(Vec3::splat(i))
            }
            Self::Box { half_extents } => {
                // I = (1/12) * m * (h^2 + d^2), etc.
                let h = *half_extents * 2.0; // full extents
                let xx = (1.0 / 12.0) * mass * (h.y * h.y + h.z * h.z);
                let yy = (1.0 / 12.0) * mass * (h.x * h.x + h.z * h.z);
                let zz = (1.0 / 12.0) * mass * (h.x * h.x + h.y * h.y);
                Mat3::from_diagonal(Vec3::new(xx, yy, zz))
            }
            Self::Capsule { radius, half_height } => {
                // Approximate as cylinder + hemispheres
                let r2 = radius * radius;
                let h = *half_height * 2.0;
                
                // Cylinder contribution
                let cyl_mass = mass * (h / (h + (4.0 / 3.0) * *radius));
                let cyl_xx = cyl_mass * (3.0 * r2 + h * h) / 12.0;
                let cyl_yy = cyl_mass * r2 / 2.0;
                
                // Sphere contribution (two hemispheres = one sphere)
                let sphere_mass = mass - cyl_mass;
                let sphere_i = (2.0 / 5.0) * sphere_mass * r2;
                
                let xx = cyl_xx + sphere_i + sphere_mass * (*half_height + 0.375 * *radius).powi(2);
                let yy = cyl_yy + sphere_i;
                
                Mat3::from_diagonal(Vec3::new(xx, yy, xx))
            }
            Self::Convex { .. } | Self::Mesh { .. } => {
                // Approximate inertia using bounding box
                let aabb = self.local_aabb();
                let extents = aabb.max - aabb.min;
                let xx = (1.0 / 12.0) * mass * (extents.y * extents.y + extents.z * extents.z);
                let yy = (1.0 / 12.0) * mass * (extents.x * extents.x + extents.z * extents.z);
                let zz = (1.0 / 12.0) * mass * (extents.x * extents.x + extents.y * extents.y);
                Mat3::from_diagonal(Vec3::new(xx, yy, zz))
            }
        }
    }
}

/// Decompose a mesh into convex hulls using octree subdivision.
fn convex_decompose_mesh(triangles: &[Triangle], max_depth: u32) -> Vec<CollisionShape> {
    // Compute mesh AABB
    let mut min = triangles[0].vertices[0];
    let mut max = triangles[0].vertices[0];

    for tri in triangles {
        for &v in &tri.vertices {
            min = min.min(v);
            max = max.max(v);
        }
    }

    // Add small padding to avoid edge cases
    let padding = (max - min).length() * 0.001;
    min -= Vec3::splat(padding);
    max += Vec3::splat(padding);

    // Recursively subdivide and collect convex hulls
    let mut result = Vec::new();
    decompose_octree_cell(&mut result, triangles, min, max, 0, max_depth);
    result
}

/// Recursively decompose a cell using octree subdivision.
fn decompose_octree_cell(
    result: &mut Vec<CollisionShape>,
    triangles: &[Triangle],
    cell_min: Vec3,
    cell_max: Vec3,
    depth: u32,
    max_depth: u32,
) {
    // Collect triangles that intersect this cell
    let cell_triangles: Vec<&Triangle> = triangles
        .iter()
        .filter(|tri| triangle_intersects_aabb(tri, cell_min, cell_max))
        .collect();

    if cell_triangles.is_empty() {
        return;
    }

    // If we've reached max depth or have few triangles, create a convex hull
    if depth >= max_depth || cell_triangles.len() <= 4 {
        // Collect all vertices from triangles in this cell
        let mut vertices = Vec::new();
        for tri in &cell_triangles {
            for &v in &tri.vertices {
                // Only include vertices inside or near the cell
                if point_near_aabb(v, cell_min, cell_max, 0.01) {
                    vertices.push(v);
                }
            }
        }

        // Also add cell corners that are inside the mesh (approximation)
        // For simplicity, just use the triangle vertices

        if vertices.len() >= 4 {
            let hull = CollisionShape::convex_hull(vertices);
            if let CollisionShape::Convex { faces, .. } = &hull {
                if !faces.is_empty() {
                    result.push(hull);
                }
            }
        } else if !vertices.is_empty() {
            // Not enough for a hull, but we have some vertices
            // Create a small box around the centroid as fallback
            let centroid: Vec3 = vertices.iter().copied().sum::<Vec3>() / vertices.len() as f32;
            let half_size = (cell_max - cell_min).length() * 0.1;
            result.push(CollisionShape::Box {
                half_extents: Vec3::splat(half_size.max(0.01)),
            });
            // Note: The box would need to be positioned at centroid when used
            let _ = centroid; // Silence unused warning
        }
        return;
    }

    // Subdivide into 8 octants
    let center = (cell_min + cell_max) * 0.5;

    for i in 0..8 {
        let octant_min = Vec3::new(
            if i & 1 == 0 { cell_min.x } else { center.x },
            if i & 2 == 0 { cell_min.y } else { center.y },
            if i & 4 == 0 { cell_min.z } else { center.z },
        );
        let octant_max = Vec3::new(
            if i & 1 == 0 { center.x } else { cell_max.x },
            if i & 2 == 0 { center.y } else { cell_max.y },
            if i & 4 == 0 { center.z } else { cell_max.z },
        );

        decompose_octree_cell(result, triangles, octant_min, octant_max, depth + 1, max_depth);
    }
}

/// Check if a triangle intersects an AABB.
fn triangle_intersects_aabb(tri: &Triangle, aabb_min: Vec3, aabb_max: Vec3) -> bool {
    // Quick check: if any vertex is inside, they intersect
    for &v in &tri.vertices {
        if v.x >= aabb_min.x && v.x <= aabb_max.x &&
           v.y >= aabb_min.y && v.y <= aabb_max.y &&
           v.z >= aabb_min.z && v.z <= aabb_max.z {
            return true;
        }
    }

    // Check if triangle AABB overlaps with cell AABB
    let tri_aabb = tri.aabb();
    if tri_aabb.max.x < aabb_min.x || tri_aabb.min.x > aabb_max.x {
        return false;
    }
    if tri_aabb.max.y < aabb_min.y || tri_aabb.min.y > aabb_max.y {
        return false;
    }
    if tri_aabb.max.z < aabb_min.z || tri_aabb.min.z > aabb_max.z {
        return false;
    }

    // AABBs overlap, assume intersection for simplicity
    // (Full triangle-AABB intersection test is more complex)
    true
}

/// Check if a point is inside or near an AABB.
fn point_near_aabb(point: Vec3, aabb_min: Vec3, aabb_max: Vec3, tolerance: f32) -> bool {
    point.x >= aabb_min.x - tolerance && point.x <= aabb_max.x + tolerance &&
    point.y >= aabb_min.y - tolerance && point.y <= aabb_max.y + tolerance &&
    point.z >= aabb_min.z - tolerance && point.z <= aabb_max.z + tolerance
}

/// Compute faces for a convex hull using the QuickHull algorithm.
///
/// QuickHull builds the hull by:
/// 1. Finding extreme points and building an initial tetrahedron
/// 2. Assigning remaining points to faces they're "outside" of
/// 3. For each face with outside points, finding the farthest point
/// 4. Creating new faces from horizon edges to this farthest point
/// 5. Repeating until no faces have outside points
fn compute_convex_faces(vertices: &[Vec3]) -> Vec<Face> {
    if vertices.len() < 4 {
        return Vec::new();
    }

    // Remove duplicate vertices (within tolerance)
    let unique_vertices = remove_duplicate_vertices(vertices, 1e-6);
    if unique_vertices.len() < 4 {
        return Vec::new();
    }

    // Find initial tetrahedron from extreme points
    let initial_indices = match find_initial_tetrahedron(&unique_vertices) {
        Some(indices) => indices,
        None => return Vec::new(), // Degenerate case: all points coplanar/collinear
    };

    // Build the convex hull using QuickHull
    quickhull_3d(&unique_vertices, &initial_indices, vertices)
}

/// Remove duplicate vertices within a tolerance.
fn remove_duplicate_vertices(vertices: &[Vec3], tolerance: f32) -> Vec<Vec3> {
    let mut unique = Vec::with_capacity(vertices.len());
    let tol_sq = tolerance * tolerance;

    for v in vertices {
        let is_duplicate = unique.iter().any(|u: &Vec3| (*u - *v).length_squared() < tol_sq);
        if !is_duplicate {
            unique.push(*v);
        }
    }
    unique
}

/// Find 4 non-coplanar points to form an initial tetrahedron.
/// Returns indices into the unique_vertices array.
fn find_initial_tetrahedron(vertices: &[Vec3]) -> Option<[usize; 4]> {
    let n = vertices.len();
    if n < 4 {
        return None;
    }

    // Find extreme points in each axis direction
    let (mut min_x, mut max_x) = (0, 0);
    let (mut min_y, mut max_y) = (0, 0);
    let (mut min_z, mut max_z) = (0, 0);

    for i in 1..n {
        if vertices[i].x < vertices[min_x].x { min_x = i; }
        if vertices[i].x > vertices[max_x].x { max_x = i; }
        if vertices[i].y < vertices[min_y].y { min_y = i; }
        if vertices[i].y > vertices[max_y].y { max_y = i; }
        if vertices[i].z < vertices[min_z].z { min_z = i; }
        if vertices[i].z > vertices[max_z].z { max_z = i; }
    }

    // Find the two most distant extreme points
    let extreme_pairs = [
        (min_x, max_x), (min_y, max_y), (min_z, max_z),
    ];

    let (mut p0, mut p1) = (0, 1);
    let mut max_dist_sq = 0.0f32;

    for &(a, b) in &extreme_pairs {
        let dist_sq = (vertices[a] - vertices[b]).length_squared();
        if dist_sq > max_dist_sq {
            max_dist_sq = dist_sq;
            p0 = a;
            p1 = b;
        }
    }

    if max_dist_sq < 1e-12 {
        return None; // All points are coincident
    }

    // Find third point: farthest from line p0-p1
    let line_dir = (vertices[p1] - vertices[p0]).normalize_or_zero();
    let mut p2 = 0;
    let mut max_dist = 0.0f32;

    for i in 0..n {
        if i == p0 || i == p1 {
            continue;
        }
        let to_point = vertices[i] - vertices[p0];
        let proj = to_point.dot(line_dir);
        let closest = vertices[p0] + line_dir * proj;
        let dist = (vertices[i] - closest).length();
        if dist > max_dist {
            max_dist = dist;
            p2 = i;
        }
    }

    if max_dist < 1e-6 {
        return None; // All points are collinear
    }

    // Find fourth point: farthest from plane p0-p1-p2
    let v01 = vertices[p1] - vertices[p0];
    let v02 = vertices[p2] - vertices[p0];
    let plane_normal = v01.cross(v02).normalize_or_zero();

    if plane_normal.length_squared() < 1e-12 {
        return None; // Degenerate triangle
    }

    let mut p3 = 0;
    let mut max_plane_dist = 0.0f32;

    for i in 0..n {
        if i == p0 || i == p1 || i == p2 {
            continue;
        }
        let to_point = vertices[i] - vertices[p0];
        let dist = to_point.dot(plane_normal).abs();
        if dist > max_plane_dist {
            max_plane_dist = dist;
            p3 = i;
        }
    }

    if max_plane_dist < 1e-6 {
        return None; // All points are coplanar
    }

    Some([p0, p1, p2, p3])
}

/// Internal face structure for QuickHull algorithm.
#[derive(Clone)]
struct HullFace {
    indices: [usize; 3],
    normal: Vec3,
    outside_set: Vec<usize>,
}

impl HullFace {
    fn new(vertices: &[Vec3], i0: usize, i1: usize, i2: usize) -> Self {
        let a = vertices[i0];
        let b = vertices[i1];
        let c = vertices[i2];
        let normal = (b - a).cross(c - a).normalize_or_zero();
        Self {
            indices: [i0, i1, i2],
            normal,
            outside_set: Vec::new(),
        }
    }

    fn centroid(&self, vertices: &[Vec3]) -> Vec3 {
        let a = vertices[self.indices[0]];
        let b = vertices[self.indices[1]];
        let c = vertices[self.indices[2]];
        (a + b + c) / 3.0
    }

    /// Distance from point to face plane (positive = outside).
    fn distance_to_point(&self, vertices: &[Vec3], point: Vec3) -> f32 {
        let face_point = vertices[self.indices[0]];
        (point - face_point).dot(self.normal)
    }
}

/// Main QuickHull algorithm for 3D convex hull.
fn quickhull_3d(unique_vertices: &[Vec3], initial: &[usize; 4], original_vertices: &[Vec3]) -> Vec<Face> {
    let vertices = unique_vertices;
    let [p0, p1, p2, p3] = *initial;

    // Create initial tetrahedron (4 faces)
    let mut faces = vec![
        HullFace::new(vertices, p0, p1, p2),
        HullFace::new(vertices, p0, p2, p3),
        HullFace::new(vertices, p0, p3, p1),
        HullFace::new(vertices, p1, p3, p2),
    ];

    // Compute centroid for orientation
    let centroid = (vertices[p0] + vertices[p1] + vertices[p2] + vertices[p3]) / 4.0;

    // Ensure all faces point outward
    for face in &mut faces {
        let face_center = face.centroid(vertices);
        let to_centroid = centroid - face_center;
        if face.normal.dot(to_centroid) > 0.0 {
            face.normal = -face.normal;
            face.indices.swap(1, 2);
        }
    }

    // Assign each non-tetrahedron vertex to a face's outside set
    let initial_set: std::collections::HashSet<usize> = initial.iter().copied().collect();
    for i in 0..vertices.len() {
        if initial_set.contains(&i) {
            continue;
        }
        assign_point_to_face(&mut faces, vertices, i);
    }

    // Process faces with outside points
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 10000;

    while iteration < MAX_ITERATIONS {
        iteration += 1;

        // Find a face with outside points
        let face_idx = faces.iter().position(|f| !f.outside_set.is_empty());
        let face_idx = match face_idx {
            Some(idx) => idx,
            None => break, // No more outside points - hull is complete
        };

        // Find the farthest point from this face
        let farthest_idx = {
            let face = &faces[face_idx];
            let mut best_idx = face.outside_set[0];
            let mut best_dist = face.distance_to_point(vertices, vertices[best_idx]);

            for &pt_idx in &face.outside_set[1..] {
                let dist = face.distance_to_point(vertices, vertices[pt_idx]);
                if dist > best_dist {
                    best_dist = dist;
                    best_idx = pt_idx;
                }
            }
            best_idx
        };

        // Find all faces visible from the farthest point
        let eye_point = vertices[farthest_idx];
        let mut visible_faces = Vec::new();

        for (i, face) in faces.iter().enumerate() {
            if face.distance_to_point(vertices, eye_point) > 1e-8 {
                visible_faces.push(i);
            }
        }

        if visible_faces.is_empty() {
            // Point is not truly outside any face (numerical issue)
            faces[face_idx].outside_set.retain(|&idx| idx != farthest_idx);
            continue;
        }

        // Find horizon edges (edges between visible and non-visible faces)
        let horizon = find_horizon_edges(&faces, &visible_faces);

        // Collect all outside points from visible faces
        let mut orphan_points: Vec<usize> = Vec::new();
        for &fi in &visible_faces {
            for &pt in &faces[fi].outside_set {
                if pt != farthest_idx {
                    orphan_points.push(pt);
                }
            }
        }

        // Remove visible faces (in reverse order to preserve indices)
        let mut visible_sorted = visible_faces.clone();
        visible_sorted.sort_by(|a, b| b.cmp(a));
        for fi in visible_sorted {
            faces.swap_remove(fi);
        }

        // Create new faces from horizon edges to the farthest point
        let mut new_faces = Vec::new();
        for (e0, e1) in horizon {
            let mut new_face = HullFace::new(vertices, e0, e1, farthest_idx);

            // Ensure outward orientation
            let face_center = new_face.centroid(vertices);
            let to_centroid = centroid - face_center;
            if new_face.normal.dot(to_centroid) > 0.0 {
                new_face.normal = -new_face.normal;
                new_face.indices.swap(0, 1);
            }

            new_faces.push(new_face);
        }

        // Assign orphan points to new faces
        for new_face in &mut new_faces {
            for &pt_idx in &orphan_points {
                let dist = new_face.distance_to_point(vertices, vertices[pt_idx]);
                if dist > 1e-8 {
                    new_face.outside_set.push(pt_idx);
                }
            }
        }

        faces.extend(new_faces);
    }

    // Map unique vertex indices back to original vertices
    convert_to_original_indices(&faces, unique_vertices, original_vertices)
}

/// Assign a point to the first face it's outside of.
fn assign_point_to_face(faces: &mut [HullFace], vertices: &[Vec3], point_idx: usize) {
    let point = vertices[point_idx];
    for face in faces.iter_mut() {
        if face.distance_to_point(vertices, point) > 1e-8 {
            face.outside_set.push(point_idx);
            return;
        }
    }
}

/// Find horizon edges: edges that border visible and non-visible faces.
fn find_horizon_edges(faces: &[HullFace], visible_indices: &[usize]) -> Vec<(usize, usize)> {
    use std::collections::HashMap;

    let visible_set: std::collections::HashSet<usize> = visible_indices.iter().copied().collect();

    // Count edge occurrences among visible faces
    // Edge is (min_idx, max_idx) for canonical form
    let mut edge_count: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();

    for &fi in visible_indices {
        let face = &faces[fi];
        let edges = [
            (face.indices[0], face.indices[1]),
            (face.indices[1], face.indices[2]),
            (face.indices[2], face.indices[0]),
        ];

        for (a, b) in edges {
            let key = if a < b { (a, b) } else { (b, a) };
            edge_count.entry(key).or_default().push((a, b));
        }
    }

    // Also check non-visible faces to find shared edges
    let mut horizon = Vec::new();

    for (i, face) in faces.iter().enumerate() {
        if visible_set.contains(&i) {
            continue;
        }

        let edges = [
            (face.indices[0], face.indices[1]),
            (face.indices[1], face.indices[2]),
            (face.indices[2], face.indices[0]),
        ];

        for (a, b) in edges {
            let key = if a < b { (a, b) } else { (b, a) };
            if edge_count.contains_key(&key) {
                // This edge is shared with a visible face - it's a horizon edge
                // Use the winding from the non-visible face (reversed for new face)
                horizon.push((b, a));
            }
        }
    }

    // If we didn't find horizon edges from non-visible faces,
    // find edges that appear only once in visible faces
    if horizon.is_empty() {
        for (_, occurrences) in edge_count {
            if occurrences.len() == 1 {
                let (a, b) = occurrences[0];
                horizon.push((b, a)); // Reverse for correct winding
            }
        }
    }

    horizon
}

/// Convert faces with unique vertex indices to faces with original vertex indices.
fn convert_to_original_indices(
    hull_faces: &[HullFace],
    unique_vertices: &[Vec3],
    original_vertices: &[Vec3],
) -> Vec<Face> {
    // Build mapping from unique vertices to original indices
    let tol_sq = 1e-10;
    let map_to_original = |unique_idx: usize| -> usize {
        let unique_v = unique_vertices[unique_idx];
        for (i, v) in original_vertices.iter().enumerate() {
            if (*v - unique_v).length_squared() < tol_sq {
                return i;
            }
        }
        0 // Fallback (shouldn't happen)
    };

    hull_faces
        .iter()
        .map(|hf| {
            let i0 = map_to_original(hf.indices[0]);
            let i1 = map_to_original(hf.indices[1]);
            let i2 = map_to_original(hf.indices[2]);
            Face::triangle(original_vertices, i0, i1, i2)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_support() {
        let sphere = CollisionShape::sphere(2.0);
        let support = sphere.support(Vec3::X);
        assert!((support - Vec3::new(2.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_sphere_support_diagonal() {
        let sphere = CollisionShape::sphere(1.0);
        let dir = Vec3::new(1.0, 1.0, 0.0).normalize();
        let support = sphere.support(dir);
        assert!((support.length() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_box_support() {
        let b = CollisionShape::cube(Vec3::new(1.0, 2.0, 3.0));
        let support = b.support(Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(support, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_box_support_negative() {
        let b = CollisionShape::cube(Vec3::new(1.0, 2.0, 3.0));
        let support = b.support(Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(support, Vec3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn test_capsule_support_up() {
        let c = CollisionShape::capsule(1.0, 2.0);
        let support = c.support(Vec3::Y);
        // Should be at top hemisphere center + radius in Y
        assert!((support.y - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_sphere_aabb() {
        let sphere = CollisionShape::sphere(3.0);
        let aabb = sphere.local_aabb();
        assert_eq!(aabb.min, Vec3::splat(-3.0));
        assert_eq!(aabb.max, Vec3::splat(3.0));
    }

    #[test]
    fn test_box_aabb() {
        let b = CollisionShape::cube(Vec3::new(1.0, 2.0, 3.0));
        let aabb = b.local_aabb();
        assert_eq!(aabb.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_capsule_aabb() {
        let c = CollisionShape::capsule(1.0, 2.0);
        let aabb = c.local_aabb();
        assert_eq!(aabb.min, Vec3::new(-1.0, -3.0, -1.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 3.0, 1.0));
    }

    #[test]
    fn test_sphere_inertia_symmetric() {
        let sphere = CollisionShape::sphere(1.0);
        let inertia = sphere.inertia_tensor(1.0);
        assert!((inertia.x_axis.x - inertia.y_axis.y).abs() < 0.001);
        assert!((inertia.y_axis.y - inertia.z_axis.z).abs() < 0.001);
    }

    #[test]
    fn test_box_inertia() {
        let b = CollisionShape::cube(Vec3::new(0.5, 0.5, 0.5));
        let inertia = b.inertia_tensor(1.0);
        // Cube should have equal diagonal elements
        assert!((inertia.x_axis.x - inertia.y_axis.y).abs() < 0.001);
    }

    #[test]
    fn test_center_of_mass() {
        let sphere = CollisionShape::sphere(1.0);
        assert_eq!(sphere.center_of_mass(), Vec3::ZERO);
    }

    #[test]
    fn test_capsule_inertia_not_zero() {
        let c = CollisionShape::capsule(1.0, 2.0);
        let inertia = c.inertia_tensor(1.0);
        assert!(inertia.x_axis.x > 0.0);
        assert!(inertia.y_axis.y > 0.0);
        assert!(inertia.z_axis.z > 0.0);
    }
    
    #[test]
    fn test_convex_support() {
        // Simple tetrahedron
        let vertices = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(0.0, -1.0, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        
        // Support in +Y should be top vertex
        let support = convex.support(Vec3::Y);
        assert!((support.y - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_convex_aabb() {
        let vertices = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let aabb = convex.local_aabb();
        
        assert_eq!(aabb.min, Vec3::splat(-1.0));
        assert_eq!(aabb.max, Vec3::splat(1.0));
    }
    
    #[test]
    fn test_convex_center_of_mass() {
        let vertices = vec![
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let com = convex.center_of_mass();
        
        // Should be average of all vertices
        assert!((com.x - 0.0).abs() < 0.01);
        assert!((com.y - 0.25).abs() < 0.01);
        assert!((com.z - 0.25).abs() < 0.01);
    }
    
    #[test]
    fn test_convex_inertia() {
        let vertices = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let inertia = convex.inertia_tensor(1.0);
        
        // Cube-like shape should have roughly equal diagonal
        assert!(inertia.x_axis.x > 0.0);
        assert!((inertia.x_axis.x - inertia.y_axis.y).abs() < 0.01);
    }
    
    #[test]
    fn test_face_triangle() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = Face::triangle(&vertices, 0, 1, 2);

        // Normal should point in +Z
        assert!(face.normal.z > 0.9);
        assert_eq!(face.indices.len(), 3);
    }

    // QuickHull algorithm tests

    #[test]
    fn test_quickhull_tetrahedron() {
        // Simple tetrahedron - should produce 4 faces
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 1.0),
        ];
        let faces = compute_convex_faces(&vertices);

        assert_eq!(faces.len(), 4);
        for face in &faces {
            assert_eq!(face.indices.len(), 3);
            assert!(face.normal.length() > 0.99);
        }
    }

    #[test]
    fn test_quickhull_cube() {
        // Cube vertices - should produce 12 faces (6 faces * 2 triangles each)
        // or 6 faces if algorithm produces quads (but we use triangles)
        let vertices = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let faces = compute_convex_faces(&vertices);

        // Should have 12 triangular faces for a cube
        assert_eq!(faces.len(), 12);

        // All vertices should be used
        let mut used_vertices = std::collections::HashSet::new();
        for face in &faces {
            for &idx in &face.indices {
                used_vertices.insert(idx);
            }
        }
        assert_eq!(used_vertices.len(), 8);
    }

    #[test]
    fn test_quickhull_octahedron() {
        // Octahedron vertices
        let vertices = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let faces = compute_convex_faces(&vertices);

        // Octahedron has 8 triangular faces
        assert_eq!(faces.len(), 8);
    }

    #[test]
    fn test_quickhull_random_convex() {
        // Random points - verify hull properties
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(1.0, 1.0, 2.0),
            Vec3::new(1.0, 0.5, 0.5), // Interior point
            Vec3::new(0.5, 0.5, 0.5), // Interior point
        ];
        let faces = compute_convex_faces(&vertices);

        // Should still be 4 faces (interior points ignored)
        assert_eq!(faces.len(), 4);

        // All face normals should point outward
        let centroid: Vec3 = vertices.iter().copied().sum::<Vec3>() / vertices.len() as f32;
        for face in &faces {
            let face_center: Vec3 = face.indices.iter()
                .map(|&i| vertices[i])
                .sum::<Vec3>() / face.indices.len() as f32;
            let to_centroid = centroid - face_center;
            // Normal should point away from centroid
            assert!(face.normal.dot(to_centroid) < 0.0);
        }
    }

    #[test]
    fn test_quickhull_degenerate_too_few_points() {
        // Less than 4 points - should return empty
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        let faces = compute_convex_faces(&vertices);
        assert!(faces.is_empty());
    }

    #[test]
    fn test_quickhull_degenerate_coplanar() {
        // All points coplanar - should return empty
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
        ];
        let faces = compute_convex_faces(&vertices);
        assert!(faces.is_empty());
    }

    #[test]
    fn test_quickhull_degenerate_collinear() {
        // All points collinear - should return empty
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let faces = compute_convex_faces(&vertices);
        assert!(faces.is_empty());
    }

    #[test]
    fn test_quickhull_degenerate_duplicates() {
        // Duplicate vertices - should handle gracefully
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0), // duplicate
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 1.0),
        ];
        let faces = compute_convex_faces(&vertices);

        // Should produce valid tetrahedron
        assert_eq!(faces.len(), 4);
    }

    #[test]
    fn test_quickhull_many_points() {
        // Many points forming a sphere-like shape
        let mut vertices = Vec::new();

        // Add vertices on a sphere
        for i in 0..10 {
            let theta = std::f32::consts::PI * i as f32 / 9.0;
            for j in 0..10 {
                let phi = 2.0 * std::f32::consts::PI * j as f32 / 10.0;
                let x = theta.sin() * phi.cos();
                let y = theta.sin() * phi.sin();
                let z = theta.cos();
                vertices.push(Vec3::new(x, y, z));
            }
        }

        let faces = compute_convex_faces(&vertices);

        // Should have reasonable number of faces
        assert!(!faces.is_empty());

        // All faces should have valid normals
        for face in &faces {
            assert!(face.normal.length() > 0.99);
        }
    }

    #[test]
    fn test_quickhull_faces_outward() {
        // Verify all face normals point outward
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(1.0, 1.0, 2.0),
        ];
        let faces = compute_convex_faces(&vertices);
        let centroid: Vec3 = vertices.iter().copied().sum::<Vec3>() / vertices.len() as f32;

        for face in &faces {
            let face_center: Vec3 = face.indices.iter()
                .map(|&i| vertices[i])
                .sum::<Vec3>() / face.indices.len() as f32;
            let outward = face_center - centroid;
            // Normal should have positive dot product with outward direction
            assert!(face.normal.dot(outward) > 0.0, "Face normal should point outward");
        }
    }

    // Triangle tests

    #[test]
    fn test_triangle_new() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );

        // Normal should point in +Z
        assert!((tri.normal() - Vec3::Z).length() < 0.001);
    }

    #[test]
    fn test_triangle_centroid() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        );

        let centroid = tri.centroid();
        assert!((centroid - Vec3::new(1.0, 1.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_triangle_support() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );

        // Support in +X should be (1, 0, 0)
        let support = tri.support(Vec3::X);
        assert!((support - Vec3::new(1.0, 0.0, 0.0)).length() < 0.001);

        // Support in +Y should be (0, 1, 0)
        let support = tri.support(Vec3::Y);
        assert!((support - Vec3::new(0.0, 1.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_triangle_aabb() {
        let tri = Triangle::new(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 1.0),
        );

        let aabb = tri.aabb();
        assert_eq!(aabb.min, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 2.0, 1.0));
    }

    // Mesh collider tests

    #[test]
    fn test_mesh_from_triangles() {
        let triangles = vec![
            Triangle::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
            ),
            Triangle::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.5),
            ),
        ];

        let mesh = CollisionShape::mesh(triangles);
        assert!(mesh.triangles().is_some());
        assert_eq!(mesh.triangles().unwrap().len(), 2);
    }

    #[test]
    fn test_mesh_from_vertices_and_indices() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 1.0),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];

        let mesh = CollisionShape::from_vertices_and_indices(&vertices, &indices);
        assert!(mesh.triangles().is_some());
        assert_eq!(mesh.triangles().unwrap().len(), 4);
    }

    #[test]
    fn test_mesh_support() {
        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
            Triangle::new(
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ),
        ];

        let mesh = CollisionShape::mesh(triangles);

        // Support in +X
        let support = mesh.support(Vec3::X);
        assert!((support - Vec3::new(1.0, 0.0, 0.0)).length() < 0.001);

        // Support in +Z
        let support = mesh.support(Vec3::Z);
        assert!((support - Vec3::new(0.0, 0.0, 1.0)).length() < 0.001);
    }

    #[test]
    fn test_mesh_aabb() {
        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ),
            Triangle::new(
                Vec3::new(0.0, 0.0, -1.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        ];

        let mesh = CollisionShape::mesh(triangles);
        let aabb = mesh.local_aabb();

        assert_eq!(aabb.min, Vec3::new(-1.0, 0.0, -1.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 2.0, 1.0));
    }

    #[test]
    fn test_mesh_center_of_mass() {
        // Two equal-area triangles
        let triangles = vec![
            Triangle::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(1.0, 2.0, 0.0),
            ),
            Triangle::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(1.0, -2.0, 0.0),
            ),
        ];

        let mesh = CollisionShape::mesh(triangles);
        let com = mesh.center_of_mass();

        // Center should be at x=1, y=0 (symmetric)
        assert!((com.x - 1.0).abs() < 0.01);
        assert!(com.y.abs() < 0.01);
    }

    #[test]
    fn test_mesh_inertia() {
        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        ];

        let mesh = CollisionShape::mesh(triangles);
        let inertia = mesh.inertia_tensor(1.0);

        // Should have positive diagonal elements
        assert!(inertia.x_axis.x > 0.0);
        assert!(inertia.y_axis.y > 0.0);
        assert!(inertia.z_axis.z > 0.0);
    }

    #[test]
    fn test_mesh_empty() {
        let mesh = CollisionShape::mesh(Vec::new());

        // Empty mesh should return sensible defaults
        assert_eq!(mesh.support(Vec3::X), Vec3::ZERO);
        assert_eq!(mesh.local_aabb().min, Vec3::ZERO);
        assert_eq!(mesh.center_of_mass(), Vec3::ZERO);
    }

    // Convex decomposition tests

    #[test]
    fn test_convex_decompose_simple_mesh() {
        // Create a simple tetrahedron mesh
        let triangles = vec![
            Triangle::new(
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(-1.0, -0.5, -0.5),
                Vec3::new(1.0, -0.5, -0.5),
            ),
            Triangle::new(
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, -0.5, -0.5),
                Vec3::new(0.0, -0.5, 1.0),
            ),
            Triangle::new(
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, -0.5, 1.0),
                Vec3::new(-1.0, -0.5, -0.5),
            ),
            Triangle::new(
                Vec3::new(-1.0, -0.5, -0.5),
                Vec3::new(0.0, -0.5, 1.0),
                Vec3::new(1.0, -0.5, -0.5),
            ),
        ];

        let mesh = CollisionShape::mesh(triangles);
        let decomposed = mesh.convex_decompose(1);

        // Should produce at least one convex shape
        assert!(!decomposed.is_empty());

        // Each decomposed shape should be a valid convex hull
        for shape in &decomposed {
            match shape {
                CollisionShape::Convex { vertices, faces } => {
                    assert!(!vertices.is_empty());
                    // May or may not have faces depending on vertex count
                    let _ = faces;
                }
                CollisionShape::Box { half_extents } => {
                    assert!(half_extents.x > 0.0);
                }
                _ => panic!("Unexpected shape type from decomposition"),
            }
        }
    }

    #[test]
    fn test_convex_decompose_cube_mesh() {
        // Create a cube as a mesh (12 triangles)
        let triangles = vec![
            // Front face
            Triangle::new(Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), Vec3::new(1.0, 1.0, 1.0)),
            Triangle::new(Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0, 1.0)),
            // Back face
            Triangle::new(Vec3::new(1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0, -1.0), Vec3::new(-1.0, 1.0, -1.0)),
            Triangle::new(Vec3::new(1.0, -1.0, -1.0), Vec3::new(-1.0, 1.0, -1.0), Vec3::new(1.0, 1.0, -1.0)),
            // Top face
            Triangle::new(Vec3::new(-1.0, 1.0, -1.0), Vec3::new(-1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0)),
            Triangle::new(Vec3::new(-1.0, 1.0, -1.0), Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, -1.0)),
            // Bottom face
            Triangle::new(Vec3::new(-1.0, -1.0, 1.0), Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, -1.0)),
            Triangle::new(Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0)),
            // Left face
            Triangle::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(-1.0, 1.0, 1.0)),
            Triangle::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0, -1.0)),
            // Right face
            Triangle::new(Vec3::new(1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0)),
            Triangle::new(Vec3::new(1.0, -1.0, 1.0), Vec3::new(1.0, 1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
        ];

        let mesh = CollisionShape::mesh(triangles);

        // Test with decomposition depth 0 (entire mesh as one hull)
        let decomposed_d0 = mesh.convex_decompose(0);

        // With depth 0, all triangles should form a single convex hull
        assert!(!decomposed_d0.is_empty(), "Depth 0 decomposition should produce at least one hull");

        // Test with higher depth
        let decomposed_d2 = mesh.convex_decompose(2);
        // May or may not produce more shapes depending on triangle distribution
        // Just verify it doesn't crash and produces valid output
        let _ = decomposed_d2;
    }

    #[test]
    fn test_convex_decompose_non_mesh() {
        // convex_decompose should return empty for non-mesh shapes
        let sphere = CollisionShape::sphere(1.0);
        assert!(sphere.convex_decompose(2).is_empty());

        let box_shape = CollisionShape::cube(Vec3::ONE);
        assert!(box_shape.convex_decompose(2).is_empty());
    }

    #[test]
    fn test_convex_decompose_empty_mesh() {
        let mesh = CollisionShape::mesh(Vec::new());
        let decomposed = mesh.convex_decompose(2);
        assert!(decomposed.is_empty());
    }

    #[test]
    fn test_convex_decompose_single_triangle() {
        let triangles = vec![
            Triangle::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
            ),
        ];

        let mesh = CollisionShape::mesh(triangles);
        let decomposed = mesh.convex_decompose(1);

        // Single triangle can't form a 3D hull, may produce a box fallback
        // or empty depending on implementation
        // Just verify it doesn't crash
        let _ = decomposed;
    }

    #[test]
    fn test_triangle_intersects_aabb() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
        );

        // Triangle should intersect its own AABB
        assert!(triangle_intersects_aabb(&tri, Vec3::ZERO, Vec3::new(2.0, 2.0, 0.1)));

        // Triangle should not intersect distant AABB
        assert!(!triangle_intersects_aabb(&tri, Vec3::new(10.0, 10.0, 10.0), Vec3::new(20.0, 20.0, 20.0)));
    }

    #[test]
    fn test_point_near_aabb() {
        let min = Vec3::ZERO;
        let max = Vec3::ONE;

        // Point inside should be near
        assert!(point_near_aabb(Vec3::splat(0.5), min, max, 0.0));

        // Point outside but near
        assert!(point_near_aabb(Vec3::new(1.05, 0.5, 0.5), min, max, 0.1));

        // Point far outside
        assert!(!point_near_aabb(Vec3::new(5.0, 5.0, 5.0), min, max, 0.1));
    }
}
