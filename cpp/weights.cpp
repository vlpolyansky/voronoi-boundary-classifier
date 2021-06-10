#include <cmath>

#include "weights.h"

Weight::Weight(int d) : d(d) {}

PolynomialWeight::PolynomialWeight(int d, int p) : Weight(d), p(p) {}

float PolynomialWeight::estimate(float z) {
    return static_cast<float>(1.f / std::pow(z, p));
}

ThresholdWeight::ThresholdWeight(int d, float threshold) : Weight(d), threshold(threshold) {}

float ThresholdWeight::estimate(float z) {
    return z < threshold ? 1.f : 0.f;
}

GaussianPolynomialWeight::GaussianPolynomialWeight(int d, float p, float sigma, float scale) : Weight(d), p(p), sigma(sigma), scale(scale) {}

float GaussianPolynomialWeight::estimate(float z) {
//    return std::exp(-0.5f * z * z / (sigma * sigma)) / std::log(z + 1);
    return std::exp(-0.5f * z * z / (sigma * sigma)) / std::pow(z * scale, p);
//    return std::exp(-0.5f * z * z / (sigma * sigma)) * (1.f / (10 * z) + 1);
}

GaussianConicalWeight::GaussianConicalWeight(int d, float sigma) : Weight(d),
                                                                   multiplier(1.f / (sigma * std::sqrt(2.0f))) {}

float GaussianConicalWeight::estimate(float z) {
    return 1 - std::erf(z * multiplier) * (float)std::pow(z, d);
}
