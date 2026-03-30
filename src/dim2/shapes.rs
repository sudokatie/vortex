//! 2D collision shapes

use glam::Vec2;
use super::Aabb2D;

/// 2D collision shape types
#[derive(Debug, Clone)]
pub enum Shape2D {
    /// Circle with radius
    Circle { radius: f32 },
    /// Axis-aligned box with half-extents
    Box { half_extents: Vec2 },
    /// Capsule (stadium) with radius and half-height
    Capsule { radius: f32, half_height: f32 },
    /// Convex polygon (vertices in CCW order)
    Polygon { vertices: Vec<Vec2> },
}

impl Shape2D {
    /// Create a circle
    pub fn circle(radius: f32) -> Self {
        Shape2D::Circle { radius }
    }
    
    /// Create an axis-aligned box
    pub fn rect(half_extents: Vec2) -> Self {
        Shape2D::Box { half_extents }
    }
    
    /// Create a square
    pub fn square(half_size: f32) -> Self {
        Shape2D::Box { half_extents: Vec2::splat(half_size) }
    }
    
    /// Create a capsule
    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Shape2D::Capsule { radius, half_height }
    }
    
    /// Create a convex polygon from vertices (CCW order)
    pub fn polygon(vertices: Vec<Vec2>) -> Self {
        Shape2D::Polygon { vertices }
    }
    
    /// Create a regular polygon
    pub fn regular_polygon(sides: usize, radius: f32) -> Self {
        let mut vertices = Vec::with_capacity(sides);
        let angle_step = std::f32::consts::TAU / sides as f32;
        
        for i in 0..sides {
            let angle = i as f32 * angle_step - std::f32::consts::FRAC_PI_2;
            vertices.push(Vec2::new(
                angle.cos() * radius,
                angle.sin() * radius,
            ));
        }
        
        Shape2D::Polygon { vertices }
    }
    
    /// Support function for GJK/EPA (furthest point in direction)
    pub fn support(&self, dir: Vec2) -> Vec2 {
        match self {
            Shape2D::Circle { radius } => {
                dir.normalize_or_zero() * *radius
            }
            Shape2D::Box { half_extents } => {
                Vec2::new(
                    if dir.x >= 0.0 { half_extents.x } else { -half_extents.x },
                    if dir.y >= 0.0 { half_extents.y } else { -half_extents.y },
                )
            }
            Shape2D::Capsule { radius, half_height } => {
                let cap_center = Vec2::new(
                    0.0,
                    if dir.y >= 0.0 { *half_height } else { -*half_height },
                );
                cap_center + dir.normalize_or_zero() * *radius
            }
            Shape2D::Polygon { vertices } => {
                let mut best = vertices[0];
                let mut best_dot = best.dot(dir);
                
                for v in vertices.iter().skip(1) {
                    let d = v.dot(dir);
                    if d > best_dot {
                        best_dot = d;
                        best = *v;
                    }
                }
                best
            }
        }
    }
    
    /// Local-space AABB
    pub fn local_aabb(&self) -> Aabb2D {
        match self {
            Shape2D::Circle { radius } => {
                Aabb2D::new(Vec2::splat(-*radius), Vec2::splat(*radius))
            }
            Shape2D::Box { half_extents } => {
                Aabb2D::new(-*half_extents, *half_extents)
            }
            Shape2D::Capsule { radius, half_height } => {
                Aabb2D::new(
                    Vec2::new(-*radius, -half_height - radius),
                    Vec2::new(*radius, *half_height + *radius),
                )
            }
            Shape2D::Polygon { vertices } => {
                let mut aabb = Aabb2D::new(vertices[0], vertices[0]);
                for v in vertices.iter().skip(1) {
                    aabb.expand(*v);
                }
                aabb
            }
        }
    }
    
    /// Compute moment of inertia about center for given mass
    pub fn moment_of_inertia(&self, mass: f32) -> f32 {
        match self {
            Shape2D::Circle { radius } => {
                // I = 0.5 * m * r^2
                0.5 * mass * radius * radius
            }
            Shape2D::Box { half_extents } => {
                // I = (1/12) * m * (w^2 + h^2)
                let w = half_extents.x * 2.0;
                let h = half_extents.y * 2.0;
                mass * (w * w + h * h) / 12.0
            }
            Shape2D::Capsule { radius, half_height } => {
                // Approximate: treat as rectangle + two semicircles
                let rect_mass = mass * (half_height * 2.0) / ((half_height * 2.0) + std::f32::consts::PI * radius);
                let circle_mass = mass - rect_mass;
                
                let w = radius * 2.0;
                let h = half_height * 2.0;
                let rect_i = rect_mass * (w * w + h * h) / 12.0;
                let circle_i = 0.5 * circle_mass * radius * radius;
                
                rect_i + circle_i
            }
            Shape2D::Polygon { vertices } => {
                // Use shoelace formula for area and moment
                if vertices.len() < 3 {
                    return 0.0;
                }
                
                let mut sum_i = 0.0;
                let mut sum_a = 0.0;
                
                for i in 0..vertices.len() {
                    let p0 = vertices[i];
                    let p1 = vertices[(i + 1) % vertices.len()];
                    
                    let cross = p0.x * p1.y - p1.x * p0.y;
                    sum_a += cross;
                    sum_i += cross * (p0.dot(p0) + p0.dot(p1) + p1.dot(p1));
                }
                
                let area = sum_a.abs() * 0.5;
                let density = mass / area;
                
                (sum_i.abs() / 12.0) * density
            }
        }
    }
    
    /// World-space AABB given position and rotation.
    pub fn aabb(&self, position: Vec2, rotation: f32) -> Aabb2D {
        // For simplicity, compute rotated AABB by transforming local AABB corners
        let local = self.local_aabb();
        let cos = rotation.cos();
        let sin = rotation.sin();
        
        // Transform all 4 corners and find bounding box
        let corners = [
            local.min,
            Vec2::new(local.max.x, local.min.y),
            local.max,
            Vec2::new(local.min.x, local.max.y),
        ];
        
        let mut min = Vec2::splat(f32::MAX);
        let mut max = Vec2::splat(f32::MIN);
        
        for c in &corners {
            let rotated = Vec2::new(
                c.x * cos - c.y * sin,
                c.x * sin + c.y * cos,
            );
            let world = position + rotated;
            min = min.min(world);
            max = max.max(world);
        }
        
        Aabb2D::new(min, max)
    }
    
    /// Compute area
    pub fn area(&self) -> f32 {
        match self {
            Shape2D::Circle { radius } => {
                std::f32::consts::PI * radius * radius
            }
            Shape2D::Box { half_extents } => {
                4.0 * half_extents.x * half_extents.y
            }
            Shape2D::Capsule { radius, half_height } => {
                std::f32::consts::PI * radius * radius + 4.0 * radius * half_height
            }
            Shape2D::Polygon { vertices } => {
                if vertices.len() < 3 {
                    return 0.0;
                }
                
                let mut sum = 0.0;
                for i in 0..vertices.len() {
                    let p0 = vertices[i];
                    let p1 = vertices[(i + 1) % vertices.len()];
                    sum += p0.x * p1.y - p1.x * p0.y;
                }
                sum.abs() * 0.5
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_support() {
        let c = Shape2D::circle(2.0);
        let s = c.support(Vec2::X);
        assert!((s - Vec2::new(2.0, 0.0)).length() < 0.001);
    }
    
    #[test]
    fn test_box_support() {
        let b = Shape2D::rect(Vec2::new(1.0, 2.0));
        let s = b.support(Vec2::ONE);
        assert_eq!(s, Vec2::new(1.0, 2.0));
    }
    
    #[test]
    fn test_circle_aabb() {
        let c = Shape2D::circle(1.0);
        let aabb = c.local_aabb();
        assert_eq!(aabb.min, Vec2::splat(-1.0));
        assert_eq!(aabb.max, Vec2::splat(1.0));
    }
    
    #[test]
    fn test_circle_area() {
        let c = Shape2D::circle(1.0);
        let area = c.area();
        assert!((area - std::f32::consts::PI).abs() < 0.001);
    }
    
    #[test]
    fn test_box_area() {
        let b = Shape2D::rect(Vec2::new(1.0, 2.0));
        let area = b.area();
        assert!((area - 8.0).abs() < 0.001);
    }
    
    #[test]
    fn test_regular_polygon() {
        let hex = Shape2D::regular_polygon(6, 1.0);
        if let Shape2D::Polygon { vertices } = hex {
            assert_eq!(vertices.len(), 6);
            for v in &vertices {
                assert!((v.length() - 1.0).abs() < 0.001);
            }
        } else {
            panic!("Expected polygon");
        }
    }
}
