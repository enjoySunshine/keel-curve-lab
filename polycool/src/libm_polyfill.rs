// Copyright 2025 the Kurbo Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[allow(dead_code, reason = "unused if std and libm are both around")]
pub(crate) trait FloatFuncs: Sized {
    fn abs(self) -> Self;
    fn acos(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn cbrt(self) -> Self;
    fn ceil(self) -> Self;
    fn cos(self) -> Self;
    fn log2(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn powi(self, pow: i32) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn sqrt(self) -> Self;
}

impl FloatFuncs for f64 {
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    fn acos(self) -> Self {
        libm::acos(self)
    }
    fn atan2(self, other: Self) -> Self {
        libm::atan2(self, other)
    }
    fn cbrt(self) -> Self {
        libm::cbrt(self)
    }
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    fn cos(self) -> Self {
        libm::cos(self)
    }
    fn log2(self) -> Self {
        libm::log2(self)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fma(self, a, b)
    }
    fn powi(self, pow: i32) -> Self {
        libm::pow(self, pow as _)
    }
    fn sin_cos(self) -> (Self, Self) {
        libm::sincos(self)
    }
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
}
