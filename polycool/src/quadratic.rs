// Copyright 2025 the Kurbo Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use arrayvec::ArrayVec;

use crate::Quadratic;
use crate::solve;

#[cfg(feature = "libm")]
#[allow(unused_imports, reason = "unused if libm and std are both around")]
use crate::libm_polyfill::FloatFuncs as _;

impl Quadratic {
    /// This is like [`Quadratic::eval`] but faster.
    ///
    /// It would be nice if we could just make `eval` like this, but I couldn't
    /// figure out how, given the lack of specialization.
    #[doc(hidden)]
    pub fn eval_opt(&self, x: f64) -> f64 {
        let [c0, c1, c2] = self.coeffs;
        c0 + c1 * x + c2 * x * x
    }

    /// Returns the roots of this quadratic, in increasing order.
    ///
    /// Double-roots are only counted once.
    pub fn roots(&self) -> ArrayVec<f64, 2> {
        let &[c0, c1, c2] = self.coeffs();
        solve::solve_quadratic(c0, c1, c2)
    }

    /// Returns the two distinct roots of this quadratic, but only if the
    /// discriminant is positive.
    pub fn positive_discriminant_roots(&self) -> Option<(f64, f64)> {
        let &[c, b, a] = self.coeffs();
        let disc = b * b - 4.0 * a * c;
        if disc.is_finite() {
            if disc > 0.0 {
                let q = -0.5 * (b + disc.sqrt().copysign(b));
                let r0 = q / a;
                let r1 = c / q;
                Some((r0.min(r1), r0.max(r1)))
            } else {
                None
            }
        } else {
            self.positive_discriminant_roots_scaled()
        }
    }

    #[cold]
    fn positive_discriminant_roots_scaled(&self) -> Option<(f64, f64)> {
        if self.is_finite() {
            let scale = 2.0_f64.powi(-515);
            (*self * scale).positive_discriminant_roots()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn root_evaluation() {
        arbtest::arbtest(|u| {
            let q = crate::arbitrary::quadratic(u)?;
            // Arbitrary quadratics can have coefficients with wild magnitudes,
            // so we need to adjust our error expectations accordingly.
            let magnitude = q.max_abs_coefficient().max(1.0);

            for r in q.roots() {
                let y = q.eval(r);
                // To evaluate the polynomial, we need to square r, so our error
                // should be relative to the magnitude of r squared.
                let r_magnitude = r.abs().max(1.0);
                let threshold = r_magnitude * 1e-14 * r_magnitude * magnitude;
                if y.is_finite() && threshold.is_finite() {
                    assert!(
                        y.abs() <= threshold,
                        "quadratic {q:?} had root {r} evaluate to {y:?}, but expected |y| <= {threshold:?}"
                    );
                }
            }
            Ok(())
        })
        .budget_ms(5_000);
    }
}
