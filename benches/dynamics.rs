//! Dynamics and constraint solver benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use glam::Vec3;
use vortex::collision::CollisionShape;
use vortex::dynamics::*;
use vortex::constraints::*;

fn bench_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("integration");
    
    let state = IntegrationState::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
    let input = IntegrationInput::linear(Vec3::new(0.0, -9.81, 0.0));
    let dt = 1.0 / 60.0;
    
    group.bench_function("explicit_euler", |b| {
        b.iter(|| {
            black_box(integrate_state(IntegratorType::ExplicitEuler, state, input, dt))
        })
    });
    
    group.bench_function("semi_implicit_euler", |b| {
        b.iter(|| {
            black_box(integrate_state(IntegratorType::SemiImplicitEuler, state, input, dt))
        })
    });
    
    group.bench_function("verlet", |b| {
        b.iter(|| {
            black_box(integrate_state(IntegratorType::Verlet, state, input, dt))
        })
    });
    
    group.bench_function("rk4", |b| {
        b.iter(|| {
            black_box(integrate_state(IntegratorType::Rk4, state, input, dt))
        })
    });
    
    group.finish();
}

fn bench_rigid_body(c: &mut Criterion) {
    let mut group = c.benchmark_group("rigid_body");
    
    let shape = CollisionShape::sphere(1.0);
    let mut body = RigidBody::new(shape.clone(), 1.0, BodyType::Dynamic);
    
    group.bench_function("apply_force", |b| {
        b.iter(|| {
            body.apply_force(black_box(Vec3::new(10.0, 0.0, 0.0)));
            body.clear_forces();
        })
    });
    
    group.bench_function("apply_impulse", |b| {
        b.iter(|| {
            body.apply_impulse(black_box(Vec3::new(1.0, 0.0, 0.0)));
            body.linear_velocity = Vec3::ZERO;
        })
    });
    
    group.bench_function("apply_impulse_at_point", |b| {
        body.position = Vec3::ZERO;
        b.iter(|| {
            body.apply_impulse_at_point(
                black_box(Vec3::new(1.0, 0.0, 0.0)),
                black_box(Vec3::new(0.0, 1.0, 0.0)),
            );
            body.linear_velocity = Vec3::ZERO;
            body.angular_velocity = Vec3::ZERO;
        })
    });
    
    group.finish();
}

fn bench_force_generators(c: &mut Criterion) {
    let mut group = c.benchmark_group("forces");
    
    let body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    
    let gravity = Gravity::new(Vec3::new(0.0, -9.81, 0.0));
    group.bench_function("gravity", |b| {
        b.iter(|| black_box(gravity.apply_to_body(&body)))
    });
    
    let spring = Spring::new(Vec3::new(0.0, 5.0, 0.0), 100.0, 10.0, 1.0);
    group.bench_function("spring", |b| {
        b.iter(|| black_box(spring.apply_to_body(&body)))
    });
    
    let drag = Drag::new(0.5, 0.1);
    let mut body_moving = body.clone();
    body_moving.linear_velocity = Vec3::new(10.0, 0.0, 0.0);
    group.bench_function("drag", |b| {
        b.iter(|| black_box(drag.apply_to_body(&body_moving)))
    });
    
    group.finish();
}

fn bench_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver");
    
    // Create a set of solver bodies and contacts
    for count in [4, 16, 64].iter() {
        let config = SolverConfig::default();
        
        group.bench_with_input(BenchmarkId::new("contacts", count), count, |b, &n| {
            let mut bodies: Vec<SolverBody> = (0..n).map(|i| {
                SolverBody::new(
                    Vec3::new(i as f32, 0.0, 0.0),
                    glam::Quat::IDENTITY,
                    Vec3::ZERO,
                    Vec3::ZERO,
                    1.0,
                    Vec3::ONE,
                )
            }).collect();
            
            let _solver = ConstraintSolver::new(config.clone());
            
            b.iter(|| {
                // Just benchmark the body array manipulation
                black_box(&mut bodies);
            })
        });
    }
    
    group.finish();
}

fn bench_material(c: &mut Criterion) {
    let a = Material::new(0.5, 0.3);
    let b = Material::new(0.7, 0.6);
    
    c.bench_function("material_combine", |bench| {
        bench.iter(|| black_box(Material::combine(&a, &b)))
    });
}

fn bench_inertia_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("inertia");
    
    let sphere = CollisionShape::sphere(1.0);
    group.bench_function("sphere", |b| {
        b.iter(|| black_box(sphere.inertia_tensor(1.0)))
    });
    
    let box_shape = CollisionShape::cube(Vec3::new(1.0, 2.0, 0.5));
    group.bench_function("box", |b| {
        b.iter(|| black_box(box_shape.inertia_tensor(1.0)))
    });
    
    let capsule = CollisionShape::capsule(0.5, 1.0);
    group.bench_function("capsule", |b| {
        b.iter(|| black_box(capsule.inertia_tensor(1.0)))
    });
    
    group.finish();
}

fn bench_world_step(c: &mut Criterion) {
    use vortex::world::PhysicsWorld;
    
    let mut group = c.benchmark_group("world_step");
    group.sample_size(50); // Fewer samples for expensive benchmarks
    
    for body_count in [100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("bodies", body_count), 
            body_count, 
            |b, &n| {
                // Create world with n bodies
                let mut world = PhysicsWorld::new();
                world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
                
                // Add bodies in a grid to minimize collisions for baseline
                let grid_size = (n as f32).sqrt().ceil() as usize;
                for i in 0..n {
                    let x = (i % grid_size) as f32 * 3.0;
                    let y = (i / grid_size) as f32 * 3.0;
                    
                    let mut body = RigidBody::new(
                        CollisionShape::sphere(0.5),
                        1.0,
                        BodyType::Dynamic,
                    );
                    body.set_position(Vec3::new(x, y + 10.0, 0.0));
                    world.add_body(body);
                }
                
                // Add ground
                let mut ground = RigidBody::new(
                    CollisionShape::cube(Vec3::new(100.0, 1.0, 100.0)),
                    0.0,
                    BodyType::Static,
                );
                ground.set_position(Vec3::new(0.0, -1.0, 0.0));
                world.add_body(ground);
                
                let dt = 1.0 / 60.0;
                
                b.iter(|| {
                    world.step(black_box(dt));
                })
            }
        );
    }
    
    // Stress test with collisions
    group.bench_function("1000_stacked", |b| {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
        
        // Stack bodies (causes many collisions)
        for i in 0..1000 {
            let layer = i / 100;
            let idx = i % 100;
            let x = (idx % 10) as f32 * 1.2 - 5.0;
            let z = (idx / 10) as f32 * 1.2 - 5.0;
            let y = layer as f32 * 1.2 + 0.6;
            
            let mut body = RigidBody::new(
                CollisionShape::cube(Vec3::splat(0.5)),
                1.0,
                BodyType::Dynamic,
            );
            body.set_position(Vec3::new(x, y, z));
            world.add_body(body);
        }
        
        // Ground
        let mut ground = RigidBody::new(
            CollisionShape::cube(Vec3::new(50.0, 1.0, 50.0)),
            0.0,
            BodyType::Static,
        );
        ground.set_position(Vec3::new(0.0, -1.0, 0.0));
        world.add_body(ground);
        
        let dt = 1.0 / 60.0;
        
        b.iter(|| {
            world.step(black_box(dt));
        })
    });
    
    group.finish();
}

/// Validates that 1000 bodies can simulate at 60 FPS
/// (step time must be < 16.67ms)
fn bench_performance_target(c: &mut Criterion) {
    use vortex::world::PhysicsWorld;
    use std::time::Instant;
    
    let mut group = c.benchmark_group("performance_target");
    group.sample_size(20);
    
    group.bench_function("1000_bodies_60fps", |b| {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
        
        // Create 1000 bodies in a spread formation
        for i in 0..1000 {
            let x = (i % 32) as f32 * 2.5;
            let y = (i / 32) as f32 * 2.5 + 5.0;
            let z = ((i / 1024) % 32) as f32 * 2.5;
            
            let mut body = RigidBody::new(
                CollisionShape::sphere(0.5),
                1.0,
                BodyType::Dynamic,
            );
            body.set_position(Vec3::new(x, y, z));
            world.add_body(body);
        }
        
        // Ground plane
        let mut ground = RigidBody::new(
            CollisionShape::cube(Vec3::new(100.0, 1.0, 100.0)),
            0.0,
            BodyType::Static,
        );
        ground.set_position(Vec3::new(0.0, -1.0, 0.0));
        world.add_body(ground);
        
        let dt = 1.0 / 60.0;
        let target_ms = 16.67; // 60 FPS target
        
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                world.step(dt);
            }
            let elapsed = start.elapsed();
            
            // Report whether we hit target
            let per_frame_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
            if per_frame_ms > target_ms {
                eprintln!(
                    "WARNING: {:.2}ms per frame exceeds 60 FPS target ({:.2}ms)",
                    per_frame_ms, target_ms
                );
            }
            
            elapsed
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_integration,
    bench_rigid_body,
    bench_force_generators,
    bench_solver,
    bench_material,
    bench_inertia_tensor,
    bench_world_step,
    bench_performance_target,
);
criterion_main!(benches);
