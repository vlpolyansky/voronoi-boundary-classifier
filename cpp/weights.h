#pragma once


class Weight {
public:
    explicit Weight(int d);
    virtual float estimate(float z) = 0;
protected:
    int d;
};

/*
 * Important!
 * Weight: 1 / z^(d + p), returned value: 1 / z^p
 */
class PolynomialWeight : public Weight {
public:
    explicit PolynomialWeight(int d, int p);

    float estimate(float z) override;
private:
    int p;
};

/*
 * Important!
 * Weight: exp(-0.5 * z^2 / sigma^2) / z^(d + p), but due to an optimization
 * z^d is not included in the return value.
 */
class GaussianPolynomialWeight : public Weight {
public:
    GaussianPolynomialWeight(int d, float p, float sigma, float scale);

    float estimate(float z) override;
private:
    float p;
    float sigma;
    float scale;
};

class ThresholdWeight : public Weight {
public:
    explicit ThresholdWeight(int d, float threshold);

    float estimate(float z) override;

private:
    float threshold;
};

class GaussianConicalWeight : public Weight {
public:
    GaussianConicalWeight(int d, float sigma);

    float estimate(float z) override;
private:
    float multiplier;
};