// Contact generation and manifold management

use glam::Vec3;

/// Single contact point between two bodies
#[derive(Debug, Clone, Copy)]
pub struct ContactPoint {
    /// Contact point in world space on body A
    pub point_a: Vec3,
    /// Contact point in world space on body B
    pub point_b: Vec3,
    /// Contact normal (from A to B)
    pub normal: Vec3,
    /// Penetration depth (positive when overlapping)
    pub depth: f32,
    /// Cached normal impulse for warm starting
    pub normal_impulse: f32,
    /// Cached tangent impulse for warm starting
    pub tangent_impulse: Vec3,
}

impl ContactPoint {
    pub fn new(point_a: Vec3, point_b: Vec3, normal: Vec3, depth: f32) -> Self {
        Self {
            point_a,
            point_b,
            normal,
            depth,
            normal_impulse: 0.0,
            tangent_impulse: Vec3::ZERO,
        }
    }

    /// Local point on body A (for manifold persistence)
    pub fn local_point_a(&self, body_a_pos: Vec3) -> Vec3 {
        self.point_a - body_a_pos
    }

    /// Local point on body B (for manifold persistence)
    pub fn local_point_b(&self, body_b_pos: Vec3) -> Vec3 {
        self.point_b - body_b_pos
    }
}

/// Contact manifold - collection of contact points between two bodies
#[derive(Debug, Clone)]
pub struct ContactManifold {
    /// Body A handle/index
    pub body_a: u32,
    /// Body B handle/index
    pub body_b: u32,
    /// Contact points (max 4 for 3D)
    pub points: arrayvec::ArrayVec<ContactPoint, 4>,
    /// Contact normal (averaged)
    pub normal: Vec3,
}

impl ContactManifold {
    pub fn new(body_a: u32, body_b: u32) -> Self {
        Self {
            body_a,
            body_b,
            points: arrayvec::ArrayVec::new(),
            normal: Vec3::ZERO,
        }
    }

    /// Add a contact point, maintaining max 4 points
    pub fn add_point(&mut self, point: ContactPoint) {
        if self.points.len() < 4 {
            self.points.push(point);
        } else {
            // Replace the point that maximizes contact area
            self.replace_worst_point(point);
        }
        self.update_normal();
    }

    /// Find and replace the point that contributes least to manifold area
    fn replace_worst_point(&mut self, new_point: ContactPoint) {
        let mut worst_idx = 0;
        let mut worst_area = f32::MAX;

        for i in 0..self.points.len() {
            // Calculate area without this point
            let area = self.area_without_point(i, new_point.point_a);
            if area < worst_area {
                worst_area = area;
                worst_idx = i;
            }
        }

        self.points[worst_idx] = new_point;
    }

    /// Calculate manifold area if we replaced point at idx with new_point
    fn area_without_point(&self, skip_idx: usize, new_point: Vec3) -> f32 {
        let mut pts: arrayvec::ArrayVec<Vec3, 4> = arrayvec::ArrayVec::new();
        for (i, p) in self.points.iter().enumerate() {
            if i != skip_idx {
                pts.push(p.point_a);
            }
        }
        pts.push(new_point);

        if pts.len() < 3 {
            return 0.0;
        }

        // Calculate area using cross products
        let mut area = 0.0;
        for i in 1..pts.len() - 1 {
            let v1 = pts[i] - pts[0];
            let v2 = pts[i + 1] - pts[0];
            area += v1.cross(v2).length() * 0.5;
        }
        area
    }

    fn update_normal(&mut self) {
        if self.points.is_empty() {
            self.normal = Vec3::ZERO;
        } else {
            let sum: Vec3 = self.points.iter().map(|p| p.normal).sum();
            self.normal = sum.normalize_or_zero();
        }
    }

    /// Get deepest penetration point
    pub fn deepest_point(&self) -> Option<&ContactPoint> {
        self.points.iter().max_by(|a, b| {
            a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Refresh contacts - remove separating points and update positions
    pub fn refresh(&mut self, threshold: f32) {
        self.points.retain(|p| p.depth > -threshold);
    }

    /// Clear cached impulses
    pub fn clear_impulses(&mut self) {
        for p in &mut self.points {
            p.normal_impulse = 0.0;
            p.tangent_impulse = Vec3::ZERO;
        }
    }
}

/// Generate contact manifold from GJK/EPA results
pub fn generate_contacts(
    body_a: u32,
    body_b: u32,
    penetration_normal: Vec3,
    penetration_depth: f32,
    point_on_a: Vec3,
    point_on_b: Vec3,
) -> ContactManifold {
    let mut manifold = ContactManifold::new(body_a, body_b);
    
    let contact = ContactPoint::new(
        point_on_a,
        point_on_b,
        penetration_normal,
        penetration_depth,
    );
    manifold.add_point(contact);
    
    manifold
}

/// Clip polygon against plane, returning clipped vertices
pub fn clip_polygon(vertices: &[Vec3], plane_normal: Vec3, plane_dist: f32) -> Vec<Vec3> {
    if vertices.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let n = vertices.len();

    for i in 0..n {
        let current = vertices[i];
        let next = vertices[(i + 1) % n];

        let d_current = current.dot(plane_normal) - plane_dist;
        let d_next = next.dot(plane_normal) - plane_dist;

        if d_current >= 0.0 {
            result.push(current);
        }

        // Edge crosses plane
        if (d_current > 0.0 && d_next < 0.0) || (d_current < 0.0 && d_next > 0.0) {
            let t = d_current / (d_current - d_next);
            let intersection = current + t * (next - current);
            result.push(intersection);
        }
    }

    result
}

/// Reduce manifold to at most 4 points while maintaining coverage
pub fn reduce_manifold(points: &[ContactPoint]) -> arrayvec::ArrayVec<ContactPoint, 4> {
    let mut result: arrayvec::ArrayVec<ContactPoint, 4> = arrayvec::ArrayVec::new();

    if points.len() <= 4 {
        for p in points {
            result.push(*p);
        }
        return result;
    }

    // Find deepest point
    let deepest_idx = points.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.depth.partial_cmp(&b.depth).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    result.push(points[deepest_idx]);

    // Find point furthest from first
    let first = points[deepest_idx].point_a;
    let second_idx = points.iter()
        .enumerate()
        .filter(|(i, _)| *i != deepest_idx)
        .max_by(|(_, a), (_, b)| {
            let da = (a.point_a - first).length_squared();
            let db = (b.point_a - first).length_squared();
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    result.push(points[second_idx]);

    // Find point furthest from line (first, second)
    let second = points[second_idx].point_a;
    let line_dir = (second - first).normalize_or_zero();
    let third_idx = points.iter()
        .enumerate()
        .filter(|(i, _)| *i != deepest_idx && *i != second_idx)
        .max_by(|(_, a), (_, b)| {
            let da = point_line_dist(a.point_a, first, line_dir);
            let db = point_line_dist(b.point_a, first, line_dir);
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i);
    
    if let Some(idx) = third_idx {
        result.push(points[idx]);
    }

    // Find point furthest from triangle plane
    if result.len() >= 3 {
        let p0 = result[0].point_a;
        let p1 = result[1].point_a;
        let p2 = result[2].point_a;
        let tri_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        
        let fourth_idx = points.iter()
            .enumerate()
            .filter(|(i, _)| *i != deepest_idx && *i != second_idx && (third_idx != Some(*i)))
            .max_by(|(_, a), (_, b)| {
                let da = (a.point_a - p0).dot(tri_normal).abs();
                let db = (b.point_a - p0).dot(tri_normal).abs();
                da.partial_cmp(&db).unwrap()
            })
            .map(|(i, _)| i);
        
        if let Some(idx) = fourth_idx {
            result.push(points[idx]);
        }
    }

    result
}

fn point_line_dist(point: Vec3, line_point: Vec3, line_dir: Vec3) -> f32 {
    let to_point = point - line_point;
    let proj = to_point.dot(line_dir);
    let closest = line_point + proj * line_dir;
    (point - closest).length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_point_new() {
        let cp = ContactPoint::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.9, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            0.1,
        );
        assert_eq!(cp.depth, 0.1);
        assert_eq!(cp.normal_impulse, 0.0);
    }

    #[test]
    fn test_manifold_add_point() {
        let mut m = ContactManifold::new(0, 1);
        let cp = ContactPoint::new(Vec3::ZERO, Vec3::X, Vec3::X, 0.1);
        m.add_point(cp);
        assert_eq!(m.points.len(), 1);
    }

    #[test]
    fn test_manifold_max_points() {
        let mut m = ContactManifold::new(0, 1);
        for i in 0..6 {
            let cp = ContactPoint::new(
                Vec3::new(i as f32, 0.0, 0.0),
                Vec3::new(i as f32, 0.1, 0.0),
                Vec3::Y,
                0.1,
            );
            m.add_point(cp);
        }
        assert_eq!(m.points.len(), 4);
    }

    #[test]
    fn test_manifold_deepest() {
        let mut m = ContactManifold::new(0, 1);
        m.add_point(ContactPoint::new(Vec3::ZERO, Vec3::X, Vec3::X, 0.1));
        m.add_point(ContactPoint::new(Vec3::Y, Vec3::new(1.0, 1.0, 0.0), Vec3::X, 0.5));
        let deepest = m.deepest_point().unwrap();
        assert_eq!(deepest.depth, 0.5);
    }

    #[test]
    fn test_manifold_refresh() {
        let mut m = ContactManifold::new(0, 1);
        m.add_point(ContactPoint::new(Vec3::ZERO, Vec3::X, Vec3::X, 0.1));
        m.add_point(ContactPoint::new(Vec3::Y, Vec3::X, Vec3::X, -0.2));
        m.refresh(0.1);
        assert_eq!(m.points.len(), 1);
    }

    #[test]
    fn test_generate_contacts() {
        let m = generate_contacts(0, 1, Vec3::X, 0.1, Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0));
        assert_eq!(m.body_a, 0);
        assert_eq!(m.body_b, 1);
        assert_eq!(m.points.len(), 1);
    }

    #[test]
    fn test_clip_polygon_inside() {
        // Plane at y=-1 with normal Y - all vertices have y >= 0, so all are "inside"
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y];
        let result = clip_polygon(&verts, Vec3::Y, -1.0);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_clip_polygon_partial() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y];
        let result = clip_polygon(&verts, Vec3::Y, 0.5);
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_reduce_manifold_small() {
        let points = vec![
            ContactPoint::new(Vec3::ZERO, Vec3::X, Vec3::X, 0.1),
            ContactPoint::new(Vec3::Y, Vec3::X, Vec3::X, 0.2),
        ];
        let reduced = reduce_manifold(&points);
        assert_eq!(reduced.len(), 2);
    }
}
