//! Constraints and joints.

// Placeholder types for now
pub trait Joint {}

pub struct DistanceJoint;
impl Joint for DistanceJoint {}

pub struct BallJoint;
impl Joint for BallJoint {}

pub struct HingeJoint;
impl Joint for HingeJoint {}
