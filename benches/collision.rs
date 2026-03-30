//! Collision detection benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use glam::Vec3;
use vortex::collision::*;
use vortex::math::Transform;

fn bench_gjk_spheres(c: &mut Criterion) {
    let sphere = CollisionShape::sphere(1.0);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(1.5, 0.0, 0.0));
    
    c.bench_function("gjk_spheres_intersecting", |b| {
        b.iter(|| {
            gjk_intersection(
                black_box(&sphere), 
                black_box(&t1), 
                black_box(&sphere), 
                black_box(&t2)
            )
        })
    });
    
    let t3 = Transform::from_position(Vec3::new(10.0, 0.0, 0.0));
    c.bench_function("gjk_spheres_separated", |b| {
        b.iter(|| {
            gjk_intersection(
                black_box(&sphere), 
                black_box(&t1), 
                black_box(&sphere), 
                black_box(&t3)
            )
        })
    });
}

fn bench_gjk_boxes(c: &mut Criterion) {
    let box_shape = CollisionShape::cube(Vec3::ONE);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(1.5, 0.0, 0.0));
    
    c.bench_function("gjk_boxes_intersecting", |b| {
        b.iter(|| {
            gjk_intersection(
                black_box(&box_shape), 
                black_box(&t1), 
                black_box(&box_shape), 
                black_box(&t2)
            )
        })
    });
}

fn bench_gjk_convex(c: &mut Criterion) {
    // Create a tetrahedron
    let vertices = vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(0.0, -1.0, 1.0),
    ];
    let convex = CollisionShape::convex_hull(vertices);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(1.5, 0.0, 0.0));
    
    c.bench_function("gjk_convex_hull", |b| {
        b.iter(|| {
            gjk_intersection(
                black_box(&convex), 
                black_box(&t1), 
                black_box(&convex), 
                black_box(&t2)
            )
        })
    });
}

fn bench_epa(c: &mut Criterion) {
    let sphere = CollisionShape::sphere(1.0);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));
    
    c.bench_function("epa_spheres", |b| {
        b.iter(|| {
            gjk_intersection(&sphere, &t1, &sphere, &t2)
                .and_then(|simplex| epa(simplex, &sphere, &t1, &sphere, &t2))
                .map(|p| black_box(p))
        })
    });
    
    let box_shape = CollisionShape::cube(Vec3::ONE);
    c.bench_function("epa_boxes", |b| {
        b.iter(|| {
            gjk_intersection(&box_shape, &t1, &box_shape, &t2)
                .and_then(|simplex| epa(simplex, &box_shape, &t1, &box_shape, &t2))
                .map(|p| black_box(p))
        })
    });
}

fn bench_broadphase(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadphase");
    
    for count in [100, 500, 1000].iter() {
        // Setup bodies
        let mut sap = SweepAndPrune::new();
        let mut bvh = Bvh::new();
        let mut hash = SpatialHash::new(2.0);
        
        use slotmap::SlotMap;
        use vortex::world::BodyHandle;
        let mut map: SlotMap<BodyHandle, ()> = SlotMap::with_key();
        
        for i in 0..*count {
            let handle = map.insert(());
            let x = (i % 10) as f32 * 3.0;
            let y = ((i / 10) % 10) as f32 * 3.0;
            let z = (i / 100) as f32 * 3.0;
            let aabb = Aabb::from_center_extents(Vec3::new(x, y, z), Vec3::ONE);
            
            sap.insert(handle, aabb);
            bvh.insert(handle, aabb);
            hash.insert(handle, aabb);
        }
        
        group.bench_with_input(BenchmarkId::new("sap", count), count, |b, _| {
            b.iter(|| black_box(sap.query_pairs()))
        });
        
        group.bench_with_input(BenchmarkId::new("bvh", count), count, |b, _| {
            b.iter(|| black_box(bvh.query_pairs()))
        });
        
        group.bench_with_input(BenchmarkId::new("spatial_hash", count), count, |b, _| {
            b.iter(|| black_box(hash.query_pairs()))
        });
    }
    
    group.finish();
}

fn bench_aabb(c: &mut Criterion) {
    let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
    let b = Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5));
    
    c.bench_function("aabb_intersects", |b_iter| {
        b_iter.iter(|| black_box(a.intersects(&b)))
    });
    
    c.bench_function("aabb_merged", |b_iter| {
        b_iter.iter(|| black_box(a.merged(&b)))
    });
    
    c.bench_function("aabb_contains_point", |b_iter| {
        b_iter.iter(|| black_box(a.contains_point(Vec3::splat(0.5))))
    });
}

fn bench_support(c: &mut Criterion) {
    let sphere = CollisionShape::sphere(1.0);
    let box_shape = CollisionShape::cube(Vec3::ONE);
    let capsule = CollisionShape::capsule(0.5, 1.0);
    
    let dir = Vec3::new(1.0, 1.0, 1.0).normalize();
    
    c.bench_function("support_sphere", |b| {
        b.iter(|| black_box(sphere.support(dir)))
    });
    
    c.bench_function("support_box", |b| {
        b.iter(|| black_box(box_shape.support(dir)))
    });
    
    c.bench_function("support_capsule", |b| {
        b.iter(|| black_box(capsule.support(dir)))
    });
    
    let vertices = vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(0.0, -1.0, 1.0),
    ];
    let convex = CollisionShape::convex_hull(vertices);
    
    c.bench_function("support_convex_4v", |b| {
        b.iter(|| black_box(convex.support(dir)))
    });
}

criterion_group!(
    benches,
    bench_gjk_spheres,
    bench_gjk_boxes,
    bench_gjk_convex,
    bench_epa,
    bench_broadphase,
    bench_aabb,
    bench_support,
);
criterion_main!(benches);
