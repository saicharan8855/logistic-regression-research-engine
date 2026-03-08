import nbformat as nbf

# =============================================================================
# THEORY FILE 1: Bernoulli MLE (DEEP - 3-4 pages)
# =============================================================================

bernoulli_content = """# Bernoulli MLE and the Log-Loss Function

## Introduction

This notebook derives how the **Bernoulli probability mass function (PMF)** leads to the **Negative Log-Likelihood (Log-Loss)** cost function used in binary classification. We show that minimizing log-loss is equivalent to maximizing the likelihood of observed data under the assumption that labels follow a Bernoulli distribution.

---

## The Bernoulli Distribution

For a binary random variable $Y \\in \\{0, 1\\}$ with parameter $p \\in [0, 1]$, the Bernoulli PMF is:

$$P(Y = y \\mid p) = p^y (1-p)^{1-y}$$

**Interpretation:**
- When $y = 1$: $P(Y=1 \\mid p) = p$
- When $y = 0$: $P(Y=0 \\mid p) = 1-p$

This compact notation elegantly handles both cases.

---

## Logistic Regression Setup

In binary classification with features $x_i \\in \\mathbb{R}^n$ and labels $y_i \\in \\{0, 1\\}$, we model:

$$P(y_i = 1 \\mid x_i; \\theta) = \\sigma(\\theta^T x_i) = \\frac{1}{1 + e^{-\\theta^T x_i}}$$

where $\\sigma(z)$ is the **sigmoid function**.

**Key Property:** The sigmoid maps any real number to $(0, 1)$, making it a valid probability.

Let $p_i = \\sigma(\\theta^T x_i)$ be the predicted probability for sample $i$.

---

## Likelihood Function

For a dataset of $m$ independent samples $(x_1, y_1), \\ldots, (x_m, y_m)$, the **likelihood** of the parameters $\\theta$ is:

$$\\mathcal{L}(\\theta) = \\prod_{i=1}^{m} P(y_i \\mid x_i; \\theta)$$

Substituting the Bernoulli PMF:

$$\\mathcal{L}(\\theta) = \\prod_{i=1}^{m} p_i^{y_i} (1-p_i)^{1-y_i}$$

**Interpretation:** This is the probability of observing the entire dataset given parameters $\\theta$.

**Goal of MLE:** Find $\\theta$ that maximizes $\\mathcal{L}(\\theta)$.

---

## Log-Likelihood Derivation

Products are difficult to optimize. We take the **natural logarithm**, which is monotonically increasing (preserves the maximizer):

$$\\ell(\\theta) = \\log \\mathcal{L}(\\theta) = \\log \\prod_{i=1}^{m} p_i^{y_i} (1-p_i)^{1-y_i}$$

**Step 1:** Apply log to product (becomes sum):

$$\\ell(\\theta) = \\sum_{i=1}^{m} \\log \\left[ p_i^{y_i} (1-p_i)^{1-y_i} \\right]$$

**Step 2:** Use log property $\\log(ab) = \\log a + \\log b$:

$$\\ell(\\theta) = \\sum_{i=1}^{m} \\left[ y_i \\log p_i + (1-y_i) \\log(1-p_i) \\right]$$

**Step 3:** Divide by $m$ (average log-likelihood):

$$\\ell(\\theta) = \\frac{1}{m} \\sum_{i=1}^{m} \\left[ y_i \\log p_i + (1-y_i) \\log(1-p_i) \\right]$$

---

## From Maximization to Minimization

Machine learning conventionally frames optimization as **minimization**. We define the **Negative Log-Likelihood (NLL)** as the **cost function**:

$$J(\\theta) = -\\ell(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y_i \\log p_i + (1-y_i) \\log(1-p_i) \\right]$$

**Key Insight:**

$$\\boxed{\\text{Maximizing } \\ell(\\theta) \\iff \\text{Minimizing } J(\\theta)}$$

This $J(\\theta)$ is the **Binary Cross-Entropy Loss** (also called **Log-Loss**).

---

## Expanded Form

Substituting $p_i = \\sigma(\\theta^T x_i)$:

$$J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y_i \\log \\sigma(\\theta^T x_i) + (1-y_i) \\log(1 - \\sigma(\\theta^T x_i)) \\right]$$

**This is the cost function we minimize in logistic regression.**

---

## Why This Loss Function?

The log-loss is **not arbitrary**. It emerges naturally from:

1. **Probabilistic Modeling:** Assuming labels follow a Bernoulli distribution
2. **Maximum Likelihood Estimation:** Finding parameters that best explain observed data
3. **Convexity:** The resulting cost function is convex, guaranteeing a global minimum

**Alternative Perspective:** Log-loss measures the **Kullback-Leibler divergence** between the true distribution and predicted distribution.

---

## Gradient of the Cost Function

To optimize via gradient descent, we need $\\nabla J(\\theta)$.

**Derivation:**

Starting from:
$$J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y_i \\log p_i + (1-y_i) \\log(1-p_i) \\right]$$

**Step 1:** Derivative with respect to $p_i$:

$$\\frac{\\partial J}{\\partial p_i} = -\\frac{1}{m} \\left[ \\frac{y_i}{p_i} - \\frac{1-y_i}{1-p_i} \\right]$$

**Step 2:** Simplify:

$$\\frac{\\partial J}{\\partial p_i} = -\\frac{1}{m} \\cdot \\frac{y_i(1-p_i) - (1-y_i)p_i}{p_i(1-p_i)} = -\\frac{1}{m} \\cdot \\frac{y_i - p_i}{p_i(1-p_i)}$$

**Step 3:** Chain rule with $p_i = \\sigma(z_i)$ where $z_i = \\theta^T x_i$:

Recall: $\\frac{d\\sigma(z)}{dz} = \\sigma(z)(1 - \\sigma(z))$

$$\\frac{\\partial p_i}{\\partial \\theta} = \\sigma(z_i)(1-\\sigma(z_i)) \\cdot x_i = p_i(1-p_i) \\cdot x_i$$

**Step 4:** Combine via chain rule:

$$\\frac{\\partial J}{\\partial \\theta} = \\frac{\\partial J}{\\partial p_i} \\cdot \\frac{\\partial p_i}{\\partial \\theta} = -\\frac{1}{m} \\cdot \\frac{y_i - p_i}{p_i(1-p_i)} \\cdot p_i(1-p_i) \\cdot x_i$$

**Step 5:** Cancel terms:

$$\\frac{\\partial J}{\\partial \\theta} = -\\frac{1}{m}(y_i - p_i) x_i$$

**Step 6:** Sum over all samples:

$$\\nabla J(\\theta) = \\frac{1}{m} \\sum_{i=1}^{m} (p_i - y_i) x_i$$

**Vectorized form:**

$$\\boxed{\\nabla J(\\theta) = \\frac{1}{m} X^T (\\sigma(X\\theta) - y)}$$

where $X \\in \\mathbb{R}^{m \\times n}$ is the design matrix.

**Remarkable simplicity:** The gradient has the same form as linear regression!

---

## Numerical Stability

Direct computation of $\\log(\\sigma(z))$ can cause numerical issues when $z$ is very large or small.

**Stable formulation:**

$$\\log(\\sigma(z)) = \\log\\left(\\frac{1}{1+e^{-z}}\\right) = -\\log(1 + e^{-z})$$

**For positive $z$:**
$$-\\log(1 + e^{-z})$$

**For negative $z$:**
$$z - \\log(1 + e^{z})$$

This avoids overflow/underflow in the exponential.

---

## Connection to Information Theory

The log-loss measures the **cross-entropy** between:
- True distribution: $q(y) = y$ (labels are deterministic)
- Predicted distribution: $p(y) = \\sigma(\\theta^T x)$

**Cross-entropy:** $H(q, p) = -\\mathbb{E}_q[\\log p]$

For binary classification:
$$H(y, p) = -[y \\log p + (1-y) \\log(1-p)]$$

Averaged over dataset → Log-Loss.

---

## Summary

| Concept | Formula |
|---------|---------|
| **Bernoulli PMF** | $P(y \\mid p) = p^y (1-p)^{1-y}$ |
| **Likelihood** | $\\mathcal{L}(\\theta) = \\prod_{i=1}^{m} p_i^{y_i} (1-p_i)^{1-y_i}$ |
| **Log-Likelihood** | $\\ell(\\theta) = \\frac{1}{m} \\sum_{i=1}^{m} [y_i \\log p_i + (1-y_i) \\log(1-p_i)]$ |
| **Cost Function (Log-Loss)** | $J(\\theta) = -\\ell(\\theta)$ |
| **Gradient** | $\\nabla J(\\theta) = \\frac{1}{m} X^T (p - y)$ |

**Key Takeaway:** Log-loss is not an arbitrary choice—it's the statistically optimal cost function when labels follow a Bernoulli distribution.
"""

# =============================================================================
# THEORY FILE 2: Logit Link (CONCEPTUAL - 1-2 pages)
# =============================================================================

logit_content = """# The Logit Link Function

## Motivation

In linear regression, we model:
$$y = \\theta^T x + \\epsilon$$

where $y \\in \\mathbb{R}$ (unbounded).

In **binary classification**, labels are $y \\in \\{0, 1\\}$, and we want to predict **probabilities** $p \\in (0, 1)$.

**Problem:** A linear model $\\theta^T x$ can produce any real number, but probabilities must be in $(0, 1)$.

**Solution:** Use a **link function** to map $(-\\infty, \\infty) \\to (0, 1)$.

---

## The Logit Function

The **logit** (log-odds) function is:

$$\\text{logit}(p) = \\log\\left(\\frac{p}{1-p}\\right)$$

**Domain:** $p \\in (0, 1)$  
**Range:** $(-\\infty, \\infty)$

### Interpretation: Odds and Log-Odds

**Odds:**
$$\\text{odds}(p) = \\frac{p}{1-p}$$

- If $p = 0.5$ (50% chance), odds = 1 (even odds)
- If $p = 0.75$ (75% chance), odds = 3 (3-to-1 in favor)
- If $p = 0.25$ (25% chance), odds = 1/3 (3-to-1 against)

**Log-Odds (Logit):**
$$\\text{logit}(p) = \\log(\\text{odds}(p))$$

- Converts odds from $(0, \\infty)$ to $(-\\infty, \\infty)$
- Symmetric around 0: $\\text{logit}(0.5) = 0$

---

## Why the Logit Link?

In logistic regression, we model:

$$\\text{logit}(p) = \\theta^T x$$

**Equivalently:**

$$\\log\\left(\\frac{p}{1-p}\\right) = \\theta^T x$$

**Why this choice?**

1. **Maps probabilities to the entire real line:** $(0, 1) \\to (-\\infty, \\infty)$
2. **Symmetric:** $\\text{logit}(p) = -\\text{logit}(1-p)$
3. **Interpretable:** Coefficients $\\theta_j$ represent **log-odds ratios**
4. **Member of GLM family:** Logit is the canonical link for Bernoulli distribution

---

## The Inverse: Sigmoid Function

To get probabilities from the linear model, we invert the logit:

$$p = \\text{logit}^{-1}(\\theta^T x) = \\sigma(\\theta^T x)$$

where the **sigmoid (logistic) function** is:

$$\\sigma(z) = \\frac{1}{1 + e^{-z}} = \\frac{e^z}{1 + e^z}$$

**Proof that sigmoid inverts logit:**

Starting from:
$$\\text{logit}(p) = z \\quad \\implies \\quad \\log\\left(\\frac{p}{1-p}\\right) = z$$

Exponentiate both sides:
$$\\frac{p}{1-p} = e^z$$

Solve for $p$:
$$p = e^z(1-p) \\quad \\implies \\quad p = e^z - pe^z \\quad \\implies \\quad p(1 + e^z) = e^z$$

$$p = \\frac{e^z}{1 + e^z} = \\frac{1}{1 + e^{-z}} = \\sigma(z)$$

---

## Properties of the Sigmoid

### 1. Range: (0, 1)

$$\\lim_{z \\to -\\infty} \\sigma(z) = 0, \\quad \\lim_{z \\to \\infty} \\sigma(z) = 1$$

### 2. Symmetry

$$\\sigma(-z) = 1 - \\sigma(z)$$

**Proof:**
$$\\sigma(-z) = \\frac{1}{1 + e^{z}} = \\frac{1}{1 + e^{z}} \\cdot \\frac{e^{-z}}{e^{-z}} = \\frac{e^{-z}}{e^{-z} + 1}$$

$$1 - \\sigma(z) = 1 - \\frac{1}{1 + e^{-z}} = \\frac{e^{-z}}{1 + e^{-z}}$$

### 3. Derivative (Chain Rule Friendly)

$$\\frac{d\\sigma(z)}{dz} = \\sigma(z)(1 - \\sigma(z))$$

**Proof:**
$$\\frac{d}{dz} \\left(\\frac{1}{1 + e^{-z}}\\right) = \\frac{e^{-z}}{(1 + e^{-z})^2}$$

$$= \\frac{1}{1 + e^{-z}} \\cdot \\frac{e^{-z}}{1 + e^{-z}} = \\sigma(z) \\cdot (1 - \\sigma(z))$$

**Implication:** Gradient computations simplify dramatically.

---

## Interpretation of Coefficients

From:
$$\\log\\left(\\frac{p}{1-p}\\right) = \\theta_0 + \\theta_1 x_1 + \\cdots + \\theta_n x_n$$

**For a one-unit increase in $x_j$:**

$$\\Delta \\log\\left(\\frac{p}{1-p}\\right) = \\theta_j$$

**Exponentiate:**

$$\\frac{\\text{odds after increase}}{\\text{odds before increase}} = e^{\\theta_j}$$

**Interpretation:** $e^{\\theta_j}$ is the **multiplicative change in odds** per unit increase in $x_j$.

**Example:**
- If $\\theta_j = 0.5$, then $e^{0.5} \\approx 1.65$ → Odds increase by 65%
- If $\\theta_j = -0.5$, then $e^{-0.5} \\approx 0.61$ → Odds decrease by 39%

---

## Why Not Other Functions?

### Alternative: Probit Link

$$\\Phi^{-1}(p) = \\theta^T x$$

where $\\Phi$ is the standard normal CDF.

**Probit vs Logit:**
- Very similar in practice
- Logit has closed-form inverse (sigmoid)
- Probit requires numerical integration
- **Logit is preferred for computational efficiency**

### Alternative: Complementary Log-Log

$$\\log(-\\log(1-p)) = \\theta^T x$$

Used when events are rare or asymmetric.

---

## Summary

| Concept | Formula | Range |
|---------|---------|-------|
| **Logit** | $\\log\\left(\\frac{p}{1-p}\\right)$ | $(-\\infty, \\infty)$ |
| **Sigmoid** | $\\sigma(z) = \\frac{1}{1+e^{-z}}$ | $(0, 1)$ |
| **Logistic Model** | $p = \\sigma(\\theta^T x)$ | Probability |
| **Odds Ratio** | $e^{\\theta_j}$ | Multiplicative effect |

**Key Insight:** The logit link elegantly transforms probabilities into an unbounded space where linear modeling is valid, while the sigmoid maps predictions back to valid probabilities.
"""

# =============================================================================
# THEORY FILE 3: GLM Proof (CONCEPTUAL - 1-2 pages)
# =============================================================================

glm_content = """# Logistic Regression as a Generalized Linear Model

## What is a Generalized Linear Model (GLM)?

A **Generalized Linear Model** extends linear regression to non-Gaussian response distributions. It consists of three components:

1. **Random Component:** Response $Y$ follows a distribution from the **exponential family**
2. **Systematic Component:** Linear predictor $\\eta = \\theta^T x$
3. **Link Function:** $g(\\mu) = \\eta$ connects the mean $\\mu = \\mathbb{E}[Y]$ to the linear predictor

---

## The Exponential Family

A distribution belongs to the exponential family if its PDF/PMF can be written as:

$$f(y; \\phi) = \\exp\\left(\\frac{y\\phi - b(\\phi)}{a(\\psi)} + c(y, \\psi)\\right)$$

where:
- $\\phi$ is the **natural parameter**
- $b(\\phi)$ is the **cumulant function**
- $a(\\psi)$ is the **dispersion parameter** (often 1)
- $c(y, \\psi)$ is a normalization term

**Examples:**
- Gaussian: $Y \\sim \\mathcal{N}(\\mu, \\sigma^2)$
- Bernoulli: $Y \\sim \\text{Bernoulli}(p)$
- Poisson: $Y \\sim \\text{Poisson}(\\lambda)$

---

## Bernoulli Distribution in Exponential Family Form

For $Y \\in \\{0, 1\\}$ with $P(Y=1) = p$:

$$P(Y = y) = p^y (1-p)^{1-y}$$

**Step 1:** Take logarithm:

$$\\log P(Y=y) = y \\log p + (1-y) \\log(1-p)$$

**Step 2:** Algebraic manipulation:

$$= y \\log p + \\log(1-p) - y\\log(1-p)$$

$$= y \\left[\\log p - \\log(1-p)\\right] + \\log(1-p)$$

$$= y \\log\\left(\\frac{p}{1-p}\\right) + \\log(1-p)$$

**Step 3:** Define natural parameter $\\phi$:

$$\\phi = \\log\\left(\\frac{p}{1-p}\\right) = \\text{logit}(p)$$

Then:
$$p = \\sigma(\\phi) = \\frac{1}{1 + e^{-\\phi}}$$

$$1 - p = 1 - \\sigma(\\phi) = \\frac{e^{-\\phi}}{1 + e^{-\\phi}}$$

**Step 4:** Substitute:

$$\\log P(Y=y) = y\\phi + \\log\\left(\\frac{e^{-\\phi}}{1 + e^{-\\phi}}\\right)$$

$$= y\\phi - \\log(1 + e^{\\phi})$$

**Step 5:** Exponentiate:

$$P(Y=y) = \\exp\\left(y\\phi - \\log(1 + e^{\\phi})\\right)$$

**This matches the exponential family form with:**
- Natural parameter: $\\phi = \\log\\left(\\frac{p}{1-p}\\right)$
- Cumulant function: $b(\\phi) = \\log(1 + e^{\\phi})$
- Dispersion: $a(\\psi) = 1$

---

## GLM Components for Logistic Regression

### 1. Random Component

$$Y \\sim \\text{Bernoulli}(p)$$

with exponential family representation proven above.

### 2. Systematic Component

$$\\eta = \\theta^T x$$

Linear predictor combines features.

### 3. Link Function

The **canonical link** for the Bernoulli distribution is the **logit**:

$$g(p) = \\log\\left(\\frac{p}{1-p}\\right) = \\eta$$

**Why canonical?** Because $g(\\mu) = \\phi$, the natural parameter.

**Inverse link (response function):**

$$p = g^{-1}(\\eta) = \\sigma(\\eta) = \\frac{1}{1 + e^{-\\eta}}$$

---

## Mean and Variance Relationships

For exponential family distributions:

$$\\mathbb{E}[Y] = \\mu = b'(\\phi)$$

$$\\text{Var}(Y) = b''(\\phi) \\cdot a(\\psi)$$

**For Bernoulli:**

$$b(\\phi) = \\log(1 + e^{\\phi})$$

**First derivative:**

$$b'(\\phi) = \\frac{e^{\\phi}}{1 + e^{\\phi}} = \\frac{1}{1 + e^{-\\phi}} = \\sigma(\\phi) = p$$

$$\\mathbb{E}[Y] = p \\quad \\checkmark$$

**Second derivative:**

$$b''(\\phi) = \\frac{d}{d\\phi}\\left(\\frac{e^{\\phi}}{1 + e^{\\phi}}\\right) = \\frac{e^{\\phi}}{(1 + e^{\\phi})^2} = p(1-p)$$

$$\\text{Var}(Y) = p(1-p) \\quad \\checkmark$$

---

## Maximum Likelihood in GLM Framework

For GLMs, the log-likelihood is:

$$\\ell(\\theta) = \\sum_{i=1}^{m} \\frac{y_i \\phi_i - b(\\phi_i)}{a(\\psi)} + c(y_i, \\psi)$$

**For logistic regression** ($a(\\psi) = 1$):

$$\\ell(\\theta) = \\sum_{i=1}^{m} \\left[y_i \\phi_i - \\log(1 + e^{\\phi_i})\\right]$$

where $\\phi_i = \\theta^T x_i$.

**Substituting $e^{\\phi_i} = \\frac{p_i}{1-p_i}$:**

$$\\log(1 + e^{\\phi_i}) = \\log\\left(\\frac{1}{1-p_i}\\right) = -\\log(1-p_i)$$

$$\\ell(\\theta) = \\sum_{i=1}^{m} \\left[y_i \\phi_i + \\log(1-p_i)\\right]$$

$$= \\sum_{i=1}^{m} \\left[y_i \\log\\left(\\frac{p_i}{1-p_i}\\right) + \\log(1-p_i)\\right]$$

$$= \\sum_{i=1}^{m} \\left[y_i \\log p_i - y_i \\log(1-p_i) + \\log(1-p_i)\\right]$$

$$= \\sum_{i=1}^{m} \\left[y_i \\log p_i + (1-y_i) \\log(1-p_i)\\right]$$

**This is the log-likelihood we derived from the Bernoulli distribution!**

---

## Summary: Logistic Regression in the GLM Framework

| GLM Component | Logistic Regression |
|---------------|---------------------|
| **Random Component** | $Y \\sim \\text{Bernoulli}(p)$ |
| **Systematic Component** | $\\eta = \\theta^T x$ |
| **Link Function** | $g(p) = \\log\\left(\\frac{p}{1-p}\\right)$ (logit) |
| **Inverse Link** | $p = \\sigma(\\eta)$ (sigmoid) |
| **Natural Parameter** | $\\phi = \\text{logit}(p)$ |
| **Mean** | $\\mathbb{E}[Y] = p = \\sigma(\\theta^T x)$ |
| **Variance** | $\\text{Var}(Y) = p(1-p)$ |

---

## Why This Matters

Understanding logistic regression as a GLM provides:

1. **Theoretical foundation:** Not an ad-hoc model, but derived from probability theory
2. **Connection to other models:** Poisson regression, gamma regression all share the same framework
3. **Optimal properties:** MLE estimators are consistent, asymptotically normal, and efficient
4. **Generalization:** Easy to extend to multinomial logistic regression (softmax)

**Key Insight:** The logit link is not arbitrary—it's the canonical link for Bernoulli-distributed outcomes in the GLM framework.
"""

# =============================================================================
# THEORY FILE 4: Hessian PSD (DEEP - 3-4 pages)
# =============================================================================

hessian_content = """# The Hessian Matrix and Positive Semi-Definiteness

## Introduction

The **Hessian matrix** is the matrix of second-order partial derivatives of the cost function. For logistic regression, proving that the Hessian is **positive semi-definite (PSD)** establishes that:

1. The cost function is **convex**
2. Any local minimum is a **global minimum**
3. **Newton's method** converges to the optimal solution

This notebook provides a rigorous proof of the PSD property.

---

## The Cost Function (Recap)

For logistic regression, the cost function is:

$$J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[y_i \\log(p_i) + (1-y_i) \\log(1-p_i)\\right]$$

where $p_i = \\sigma(\\theta^T x_i) = \\frac{1}{1 + e^{-\\theta^T x_i}}$.

---

## The Gradient (First Derivative)

We previously derived:

$$\\nabla J(\\theta) = \\frac{1}{m} X^T (p - y)$$

where:
- $X \\in \\mathbb{R}^{m \\times n}$ is the design matrix
- $p \\in \\mathbb{R}^m$ is the vector of predicted probabilities
- $y \\in \\mathbb{R}^m$ is the vector of true labels

**Component form:**

$$\\frac{\\partial J}{\\partial \\theta_j} = \\frac{1}{m} \\sum_{i=1}^{m} (p_i - y_i) x_{ij}$$

---

## The Hessian (Second Derivative)

The **Hessian matrix** $H \\in \\mathbb{R}^{n \\times n}$ is defined as:

$$H_{jk} = \\frac{\\partial^2 J}{\\partial \\theta_j \\partial \\theta_k}$$

**Goal:** Compute $H$ and prove it is PSD.

### Step 1: Differentiate the Gradient

From:
$$\\frac{\\partial J}{\\partial \\theta_j} = \\frac{1}{m} \\sum_{i=1}^{m} (p_i - y_i) x_{ij}$$

Take derivative with respect to $\\theta_k$:

$$\\frac{\\partial^2 J}{\\partial \\theta_k \\partial \\theta_j} = \\frac{1}{m} \\sum_{i=1}^{m} x_{ij} \\frac{\\partial p_i}{\\partial \\theta_k}$$

(Note: $y_i$ is constant, so $\\frac{\\partial y_i}{\\partial \\theta_k} = 0$)

### Step 2: Compute $\\frac{\\partial p_i}{\\partial \\theta_k}$

Recall $p_i = \\sigma(z_i)$ where $z_i = \\theta^T x_i = \\sum_{\\ell=1}^{n} \\theta_\\ell x_{i\\ell}$.

**Chain rule:**

$$\\frac{\\partial p_i}{\\partial \\theta_k} = \\frac{\\partial \\sigma(z_i)}{\\partial z_i} \\cdot \\frac{\\partial z_i}{\\partial \\theta_k}$$

**Derivative of sigmoid:**

$$\\frac{\\partial \\sigma(z_i)}{\\partial z_i} = \\sigma(z_i)(1 - \\sigma(z_i)) = p_i(1 - p_i)$$

**Derivative of linear predictor:**

$$\\frac{\\partial z_i}{\\partial \\theta_k} = \\frac{\\partial}{\\partial \\theta_k} \\sum_{\\ell=1}^{n} \\theta_\\ell x_{i\\ell} = x_{ik}$$

**Combine:**

$$\\frac{\\partial p_i}{\\partial \\theta_k} = p_i(1 - p_i) \\cdot x_{ik}$$

### Step 3: Substitute Back

$$H_{jk} = \\frac{1}{m} \\sum_{i=1}^{m} x_{ij} \\cdot p_i(1-p_i) \\cdot x_{ik}$$

$$= \\frac{1}{m} \\sum_{i=1}^{m} p_i(1-p_i) \\cdot x_{ij} x_{ik}$$

---

## Matrix Form of the Hessian

Define the **diagonal weight matrix** $R \\in \\mathbb{R}^{m \\times m}$:

$$R = \\text{diag}(r_1, r_2, \\ldots, r_m)$$

where:

$$r_i = p_i(1 - p_i)$$

**Then the Hessian in matrix notation is:**

$$\\boxed{H = \\frac{1}{m} X^T R X}$$

**Proof:**

$$(X^T R X)_{jk} = \\sum_{i=1}^{m} x_{ij} \\cdot r_i \\cdot x_{ik} = \\sum_{i=1}^{m} p_i(1-p_i) \\cdot x_{ij} x_{ik}$$

Dividing by $m$ gives $H_{jk}$. ✓

---

## Positive Semi-Definiteness: Definition

A matrix $H$ is **positive semi-definite (PSD)** if for all vectors $v \\in \\mathbb{R}^n$:

$$v^T H v \\geq 0$$

**Strictly PSD:** $v^T H v > 0$ for all $v \\neq 0$.

---

## Proof that $H$ is PSD

### Approach 1: Quadratic Form

We need to show:

$$v^T H v \\geq 0 \\quad \\forall v \\in \\mathbb{R}^n$$

**Substitute $H = \\frac{1}{m} X^T R X$:**

$$v^T H v = v^T \\left(\\frac{1}{m} X^T R X\\right) v = \\frac{1}{m} v^T X^T R X v$$

**Rearrange (associativity):**

$$= \\frac{1}{m} (Xv)^T R (Xv)$$

Let $w = Xv \\in \\mathbb{R}^m$. Then:

$$v^T H v = \\frac{1}{m} w^T R w$$

**Expand $w^T R w$:**

Since $R$ is diagonal with entries $r_i = p_i(1-p_i)$:

$$w^T R w = \\sum_{i=1}^{m} r_i w_i^2 = \\sum_{i=1}^{m} p_i(1-p_i) w_i^2$$

**Key observation:** For $p_i \\in (0, 1)$:

$$p_i(1-p_i) > 0$$

**Proof:**
- $p_i > 0$ and $1 - p_i > 0$ (since $p_i \\in (0,1)$)
- Product of positive numbers is positive

**Therefore:**

$$v^T H v = \\frac{1}{m} \\sum_{i=1}^{m} p_i(1-p_i) w_i^2 \\geq 0$$

**Conclusion:** $H$ is PSD. ✓

---

### Approach 2: Eigenvalue Criterion

A symmetric matrix is PSD if and only if all its eigenvalues are non-negative.

**Claim:** All eigenvalues of $H$ are $\\geq 0$.

**Proof:**

Let $\\lambda$ be an eigenvalue of $H$ with eigenvector $v$:

$$Hv = \\lambda v$$

Multiply both sides by $v^T$:

$$v^T H v = \\lambda v^T v$$

From Approach 1, we know $v^T H v \\geq 0$. Also, $v^T v = \\|v\\|^2 > 0$ (since $v \\neq 0$).

**Therefore:**

$$\\lambda = \\frac{v^T H v}{v^T v} \\geq 0$$

**All eigenvalues are non-negative → $H$ is PSD.** ✓

---

## When is $H$ Strictly Positive Definite?

$H$ is **strictly positive definite** if $v^T H v > 0$ for all $v \\neq 0$.

From our derivation:

$$v^T H v = \\frac{1}{m} \\sum_{i=1}^{m} p_i(1-p_i) w_i^2$$

where $w = Xv$.

**$H$ is strictly PD if and only if:**

$$\\sum_{i=1}^{m} p_i(1-p_i) w_i^2 > 0 \\quad \\forall v \\neq 0$$

**This holds when:**

1. $w \\neq 0$ (i.e., $Xv \\neq 0$)
2. At least one $w_i \\neq 0$
3. $p_i \\in (0, 1)$ for all $i$ (not exactly 0 or 1)

**Condition for strict PD:**

$$\\text{rank}(X) = n$$

i.e., the columns of $X$ are linearly independent.

**Implication:**
- If features are linearly independent → $H$ is strictly PD → Unique global minimum
- If features are collinear → $H$ is PSD but not strictly PD → Multiple optimal solutions (though predictions are unique)

---

## Implications for Optimization

### 1. Convexity

Since $H$ is PSD everywhere:

$$J(\\theta)$$ **is a convex function**

**Convexity guarantees:**
- Any local minimum is a global minimum
- Gradient descent will converge to the global optimum
- No need to worry about initialization (unlike neural networks)

### 2. Newton's Method

**Newton's update rule:**

$$\\theta^{(t+1)} = \\theta^{(t)} - H^{-1} \\nabla J(\\theta^{(t)})$$

**Requirements:**
- $H$ must be invertible (strictly PD)
- If $H$ is only PSD (due to multicollinearity), add regularization:

$$H_{\\text{reg}} = H + \\lambda I$$

where $\\lambda > 0$ is small.

**Advantages of Newton's method:**
- Quadratic convergence (much faster than gradient descent)
- Naturally accounts for curvature of the cost function
- Often converges in 5-10 iterations

**Disadvantages:**
- Requires computing and inverting $H$ (O($n^3$) complexity)
- Impractical for high-dimensional problems

---

## Summary

| Property | Result |
|----------|--------|
| **Hessian Formula** | $H = \\frac{1}{m} X^T R X$ |
| **Weight Matrix** | $R = \\text{diag}(p_1(1-p_1), \\ldots, p_m(1-p_m))$ |
| **Quadratic Form** | $v^T H v = \\frac{1}{m} \\sum_{i=1}^{m} p_i(1-p_i) (x_i^T v)^2 \\geq 0$ |
| **PSD Property** | ✓ Proven (all $p_i(1-p_i) > 0$) |
| **Strictly PD Condition** | $\\text{rank}(X) = n$ (no multicollinearity) |
| **Convexity** | $J(\\theta)$ is convex |
| **Global Minimum** | Guaranteed to exist |

**Key Takeaway:** The Hessian being PSD is what makes logistic regression computationally tractable—we can use gradient-based optimization with confidence that we'll find the global optimum.
"""

# =============================================================================
# THEORY FILE 5: Fisher Information (CONCEPTUAL - 1-2 pages)
# =============================================================================

fisher_content = """# Fisher Information Matrix

## Introduction

The **Fisher Information Matrix** quantifies the amount of information that observable data carries about unknown parameters. In logistic regression, it provides the theoretical foundation for:

1. **Asymptotic variance** of parameter estimates
2. **Standard errors** and confidence intervals
3. **Wald tests** for hypothesis testing
4. **Cramér-Rao lower bound** (efficiency of estimators)

---

## Definition

For a parameter vector $\\theta \\in \\mathbb{R}^n$, the **Fisher Information Matrix** $I(\\theta) \\in \\mathbb{R}^{n \\times n}$ is defined as:

$$I(\\theta) = \\mathbb{E}\\left[\\left(\\frac{\\partial \\log p(Y \\mid X; \\theta)}{\\partial \\theta}\\right) \\left(\\frac{\\partial \\log p(Y \\mid X; \\theta)}{\\partial \\theta}\\right)^T\\right]$$

**Equivalently (under regularity conditions):**

$$I(\\theta) = -\\mathbb{E}\\left[\\frac{\\partial^2 \\log p(Y \\mid X; \\theta)}{\\partial \\theta \\partial \\theta^T}\\right]$$

**Interpretation:** Expected curvature of the log-likelihood surface.

---

## Fisher Information for Logistic Regression

### Single Observation

For a single sample $(x, y)$, the log-likelihood contribution is:

$$\\ell(\\theta; x, y) = y \\log p + (1-y) \\log(1-p)$$

where $p = \\sigma(\\theta^T x)$.

**Gradient (score):**

$$\\frac{\\partial \\ell}{\\partial \\theta} = (y - p) x$$

**Hessian:**

$$\\frac{\\partial^2 \\ell}{\\partial \\theta \\partial \\theta^T} = -p(1-p) x x^T$$

**Fisher Information (single sample):**

$$I_1(\\theta) = -\\mathbb{E}\\left[\\frac{\\partial^2 \\ell}{\\partial \\theta \\partial \\theta^T}\\right]$$

$$= \\mathbb{E}[p(1-p) x x^T]$$

Since $p = \\sigma(\\theta^T x)$ is deterministic given $x$:

$$I_1(\\theta) = p(1-p) x x^T$$

### Full Dataset

For $m$ independent samples $(x_1, y_1), \\ldots, (x_m, y_m)$:

$$I(\\theta) = \\sum_{i=1}^{m} I_i(\\theta) = \\sum_{i=1}^{m} p_i(1-p_i) x_i x_i^T$$

**Matrix form:**

$$I(\\theta) = X^T R X$$

where $R = \\text{diag}(r_1, \\ldots, r_m)$ with $r_i = p_i(1-p_i)$.

**Comparison with Hessian:**

From the Hessian derivation:

$$H = \\frac{1}{m} X^T R X$$

**Therefore:**

$$\\boxed{I(\\theta) = m \\cdot H}$$

**Key Insight:** The observed Hessian is proportional to the Fisher Information.

---

## Asymptotic Properties of MLE

Under regularity conditions, the **Maximum Likelihood Estimator (MLE)** $\\hat{\\theta}$ is:

1. **Consistent:** $\\hat{\\theta} \\xrightarrow{p} \\theta^*$ as $m \\to \\infty$
2. **Asymptotically normal:**

$$\\sqrt{m}(\\hat{\\theta} - \\theta^*) \\xrightarrow{d} \\mathcal{N}(0, I(\\theta^*)^{-1})$$

3. **Asymptotically efficient:** Achieves the Cramér-Rao lower bound

**Practical implication:**

$$\\hat{\\theta} \\sim \\mathcal{N}\\left(\\theta^*, \\frac{1}{m} I(\\theta^*)^{-1}\\right)$$

for large $m$.

---

## Standard Errors and Confidence Intervals

The **covariance matrix** of $\\hat{\\theta}$ is approximated by:

$$\\text{Cov}(\\hat{\\theta}) \\approx I(\\hat{\\theta})^{-1} = H^{-1} / m$$

**Standard error of $\\hat{\\theta}_j$:**

$$\\text{SE}(\\hat{\\theta}_j) = \\sqrt{[I(\\hat{\\theta})^{-1}]_{jj}}$$

**95% Confidence Interval:**

$$\\hat{\\theta}_j \\pm 1.96 \\cdot \\text{SE}(\\hat{\\theta}_j)$$

**Interpretation:** With 95% confidence, the true parameter $\\theta_j^*$ lies in this interval.

---

## Wald Test for Coefficient Significance

**Null hypothesis:** $H_0: \\theta_j = 0$ (feature $j$ has no effect)

**Test statistic:**

$$z_j = \\frac{\\hat{\\theta}_j}{\\text{SE}(\\hat{\\theta}_j)}$$

**Under $H_0$:**

$$z_j \\sim \\mathcal{N}(0, 1)$$

**P-value (two-tailed):**

$$p = 2 \\cdot P(|Z| > |z_j|) = 2 \\cdot \\Phi(-|z_j|)$$

where $\\Phi$ is the standard normal CDF.

**Decision rule:**
- If $p < 0.05$: Reject $H_0$ → Feature is significant
- If $p \\geq 0.05$: Fail to reject $H_0$ → Feature may not be informative

---

## Likelihood Ratio Test

**Alternative to Wald test:** Compare nested models.

**Full model:** $\\mathcal{L}(\\hat{\\theta})$ with all features

**Restricted model:** $\\mathcal{L}(\\hat{\\theta}_0)$ with feature $j$ removed

**Test statistic:**

$$\\Lambda = 2[\\ell(\\hat{\\theta}) - \\ell(\\hat{\\theta}_0)]$$

**Under $H_0$:**

$$\\Lambda \\sim \\chi^2(1)$$

**P-value:**

$$p = P(\\chi^2(1) > \\Lambda)$$

---

## Cramér-Rao Lower Bound

**Theorem:** For any unbiased estimator $\\tilde{\\theta}$:

$$\\text{Cov}(\\tilde{\\theta}) \\geq I(\\theta)^{-1}$$

(in the sense of positive semi-definite ordering)

**Implication:** The MLE achieves this bound asymptotically → **Most efficient estimator**.

---

## Practical Computation

**Step 1:** Compute predicted probabilities $p_i = \\sigma(\\theta^T x_i)$

**Step 2:** Construct weight matrix $R = \\text{diag}(p_1(1-p_1), \\ldots, p_m(1-p_m))$

**Step 3:** Compute Fisher Information: $I(\\hat{\\theta}) = X^T R X$

**Step 4:** Invert to get covariance: $\\text{Cov}(\\hat{\\theta}) = I(\\hat{\\theta})^{-1}$

**Step 5:** Extract standard errors: $\\text{SE}(\\hat{\\theta}_j) = \\sqrt{[\\text{Cov}(\\hat{\\theta})]_{jj}}$

**Step 6:** Compute Wald statistics: $z_j = \\hat{\\theta}_j / \\text{SE}(\\hat{\\theta}_j)$

**Step 7:** Compute p-values: $p_j = 2 \\Phi(-|z_j|)$

---

## Numerical Stability

**Warning:** $I(\\hat{\\theta})$ may be near-singular if:
- Features are highly correlated (multicollinearity)
- Some $p_i \\approx 0$ or $p_i \\approx 1$ (perfect separation)

**Solutions:**
1. **Ridge regularization:** $I_{\\text{reg}} = I(\\hat{\\theta}) + \\lambda I$
2. **Remove redundant features**
3. **Use pseudo-inverse** if inversion fails

---

## Summary

| Concept | Formula |
|---------|---------|
| **Fisher Information** | $I(\\theta) = X^T R X$ |
| **Relation to Hessian** | $I(\\theta) = m \\cdot H$ |
| **Covariance Matrix** | $\\text{Cov}(\\hat{\\theta}) = I(\\hat{\\theta})^{-1}$ |
| **Standard Error** | $\\text{SE}(\\hat{\\theta}_j) = \\sqrt{[I(\\hat{\\theta})^{-1}]_{jj}}$ |
| **Wald Statistic** | $z_j = \\hat{\\theta}_j / \\text{SE}(\\hat{\\theta}_j)$ |
| **Asymptotic Distribution** | $\\hat{\\theta} \\sim \\mathcal{N}(\\theta^*, I(\\theta^*)^{-1})$ |

**Key Takeaway:** The Fisher Information Matrix is the bridge between optimization (Hessian) and statistical inference (standard errors, p-values, confidence intervals).
"""

# =============================================================================
# THEORY FILE 6: Inference Theory (DEEP - 3-4 pages)
# =============================================================================

inference_content = """# Statistical Inference for Logistic Regression

## Introduction

After fitting a logistic regression model, we need to answer:

1. **Which features are statistically significant?**
2. **How confident are we in our parameter estimates?**
3. **Is the model better than a null model?**
4. **How do we interpret the effect sizes?**

This notebook covers the complete statistical inference framework.

---

## Maximum Likelihood Estimation (Recap)

The **MLE** $\\hat{\\theta}$ maximizes the log-likelihood:

$$\\hat{\\theta} = \\arg\\max_{\\theta} \\ell(\\theta) = \\arg\\max_{\\theta} \\sum_{i=1}^{m} [y_i \\log p_i + (1-y_i) \\log(1-p_i)]$$

**Properties:**
- **Consistent:** $\\hat{\\theta} \\to \\theta^*$ as $m \\to \\infty$
- **Asymptotically normal:** Distribution approaches Gaussian
- **Asymptotically efficient:** Minimum variance among unbiased estimators

---

## Asymptotic Distribution of $\\hat{\\theta}$

For large $m$, the MLE is approximately:

$$\\hat{\\theta} \\sim \\mathcal{N}\\left(\\theta^*, I(\\theta^*)^{-1}\\right)$$

where $I(\\theta^*)$ is the **Fisher Information Matrix** at the true parameters.

**Estimated covariance matrix:**

$$\\widehat{\\text{Cov}}(\\hat{\\theta}) = I(\\hat{\\theta})^{-1} = (X^T R X)^{-1}$$

where:
- $R = \\text{diag}(p_1(1-p_1), \\ldots, p_m(1-p_m))$
- $p_i = \\sigma(\\hat{\\theta}^T x_i)$ are fitted probabilities

---

## Standard Errors

The **standard error** of $\\hat{\\theta}_j$ measures the uncertainty in the estimate:

$$\\text{SE}(\\hat{\\theta}_j) = \\sqrt{[I(\\hat{\\theta})^{-1}]_{jj}}$$

**Interpretation:** If we repeated the experiment many times, the estimates $\\hat{\\theta}_j$ would vary with standard deviation $\\approx \\text{SE}(\\hat{\\theta}_j)$.

**Smaller SE → More precise estimate**

---

## Confidence Intervals

A **95% confidence interval** for $\\theta_j$ is:

$$\\hat{\\theta}_j \\pm 1.96 \\cdot \\text{SE}(\\hat{\\theta}_j)$$

**Interpretation:** We are 95% confident that the true parameter $\\theta_j^*$ lies in this interval.

**General formula for $(1-\\alpha)$ confidence:**

$$\\hat{\\theta}_j \\pm z_{\\alpha/2} \\cdot \\text{SE}(\\hat{\\theta}_j)$$

where $z_{\\alpha/2}$ is the critical value from the standard normal distribution.

**Examples:**
- 90% CI: $z_{0.05} = 1.645$
- 95% CI: $z_{0.025} = 1.96$
- 99% CI: $z_{0.005} = 2.576$

---

## Wald Test for Individual Coefficients

**Null hypothesis:** $H_0: \\theta_j = 0$ (feature $x_j$ has no effect on the outcome)

**Alternative:** $H_1: \\theta_j \\neq 0$ (two-tailed test)

**Test statistic:**

$$z_j = \\frac{\\hat{\\theta}_j}{\\text{SE}(\\hat{\\theta}_j)}$$

**Under $H_0$:**

$$z_j \\sim \\mathcal{N}(0, 1)$$

**P-value (two-tailed):**

$$p_j = 2 \\cdot P(|Z| > |z_j|) = 2 \\cdot [1 - \\Phi(|z_j|)]$$

where $\\Phi$ is the standard normal CDF.

**Decision rule (at significance level $\\alpha = 0.05$):**
- If $p_j < 0.05$: **Reject $H_0$** → Feature is statistically significant
- If $p_j \\geq 0.05$: **Fail to reject $H_0$** → Insufficient evidence of effect

---

## Interpreting P-values

**Common misconceptions:**
- ❌ "$p = 0.03$" does **NOT** mean "3% chance the null hypothesis is true"
- ❌ "Not significant" ($p > 0.05$) does **NOT** mean "no effect"

**Correct interpretation:**
- ✅ "If $H_0$ were true, we'd see data this extreme only 3% of the time"
- ✅ "Strong evidence against $H_0$, assuming our model assumptions hold"

**P-values do NOT measure:**
- Effect size (use coefficients for that)
- Importance (statistical ≠ practical significance)
- Probability that a hypothesis is true (that's Bayesian)

---

## Odds Ratios

Recall the logistic model:

$$\\log\\left(\\frac{p}{1-p}\\right) = \\theta_0 + \\theta_1 x_1 + \\cdots + \\theta_n x_n$$

**Interpretation of $\\theta_j$:**

For a **one-unit increase** in $x_j$ (holding others constant):

$$\\Delta \\log(\\text{odds}) = \\theta_j$$

**Exponentiate to get the multiplicative effect:**

$$\\text{OR}_j = e^{\\theta_j}$$

**Interpretation:**
- If $\\text{OR}_j > 1$: Odds increase by a factor of $\\text{OR}_j$
- If $\\text{OR}_j < 1$: Odds decrease
- If $\\text{OR}_j = 1$: No effect ($\\theta_j = 0$)

**Example:**
- $\\theta_j = 0.5 \\Rightarrow \\text{OR}_j = e^{0.5} \\approx 1.65$ → **65% increase in odds**
- $\\theta_j = -0.5 \\Rightarrow \\text{OR}_j = e^{-0.5} \\approx 0.61$ → **39% decrease in odds**

**Confidence interval for odds ratio:**

$$\\left[e^{\\hat{\\theta}_j - 1.96 \\cdot \\text{SE}(\\hat{\\theta}_j)}, \\; e^{\\hat{\\theta}_j + 1.96 \\cdot \\text{SE}(\\hat{\\theta}_j)}\\right]$$

---

## Likelihood Ratio Test

**Purpose:** Test whether a subset of coefficients are jointly zero.

**Example:** Is the full model significantly better than the null model (intercept only)?

**Full model:** $\\ell(\\hat{\\theta})$ with all $n$ features

**Null model:** $\\ell(\\hat{\\theta}_0)$ with only intercept

**Test statistic:**

$$\\Lambda = 2[\\ell(\\hat{\\theta}) - \\ell(\\hat{\\theta}_0)] = -2 \\log\\left(\\frac{\\mathcal{L}(\\hat{\\theta}_0)}{\\mathcal{L}(\\hat{\\theta})}\\right)$$

**Under $H_0$ (null model is correct):**

$$\\Lambda \\sim \\chi^2(k)$$

where $k$ is the number of parameters tested (usually $k = n$ for full vs null).

**P-value:**

$$p = P(\\chi^2(k) > \\Lambda)$$

**Decision:**
- If $p < 0.05$: **Reject null model** → Features improve fit
- If $p \\geq 0.05$: Null model is adequate

---

## Deviance

**Deviance** measures goodness of fit:

$$D = -2 \\ell(\\hat{\\theta})$$

**Null deviance:** $D_0 = -2 \\ell(\\hat{\\theta}_0)$ (intercept-only model)

**Residual deviance:** $D = -2 \\ell(\\hat{\\theta})$ (full model)

**Deviance reduction:**

$$\\Delta D = D_0 - D = \\Lambda$$

(This is the likelihood ratio test statistic!)

**Interpretation:**
- Large $\\Delta D$ → Model explains data much better than null
- Small $\\Delta D$ → Features don't add much predictive power

---

## Pseudo R²

Unlike linear regression, logistic regression doesn't have a natural $R^2$. Several **pseudo-$R^2$** measures exist:

### McFadden's R²

$$R^2_{\\text{McFadden}} = 1 - \\frac{\\ell(\\hat{\\theta})}{\\ell(\\hat{\\theta}_0)}$$

**Range:** $[0, 1]$

**Interpretation:**
- 0: Model is no better than null
- 1: Perfect fit (unrealistic in practice)
- **Values 0.2-0.4 are considered good**

### Cox-Snell R²

$$R^2_{\\text{Cox-Snell}} = 1 - \\left(\\frac{\\mathcal{L}(\\hat{\\theta}_0)}{\\mathcal{L}(\\hat{\\theta})}\\right)^{2/m}$$

**Issue:** Maximum value < 1 even for perfect models.

### Nagelkerke R²

$$R^2_{\\text{Nagelkerke}} = \\frac{R^2_{\\text{Cox-Snell}}}{1 - \\mathcal{L}(\\hat{\\theta}_0)^{2/m}}$$

**Corrects Cox-Snell to have range [0, 1].**

---

## Inference Table (Summary Output)

A typical inference table looks like:

| Feature | Coefficient ($\\hat{\\theta}_j$) | Std Error | z-value | p-value | Odds Ratio | 95% CI |
|---------|--------------------------------|-----------|---------|---------|------------|--------|
| Age | 0.0234 | 0.0056 | 4.18 | <0.001 | 1.024 | [1.013, 1.035] |
| Income | 0.0012 | 0.0004 | 3.00 | 0.003 | 1.001 | [1.000, 1.002] |
| Gender | -0.3456 | 0.1234 | -2.80 | 0.005 | 0.708 | [0.556, 0.902] |

**Reading this table:**
- **Age:** Significant ($p < 0.001$). Each year increases odds by 2.4%.
- **Income:** Significant ($p = 0.003$). Small effect (OR ≈ 1).
- **Gender:** Significant ($p = 0.005$). Being in reference category decreases odds by 29%.

---

## Multicollinearity and Variance Inflation

**Problem:** If features are highly correlated, standard errors inflate.

**Variance Inflation Factor (VIF):**

$$\\text{VIF}_j = \\frac{1}{1 - R_j^2}$$

where $R_j^2$ is the $R^2$ from regressing $x_j$ on all other features.

**Rule of thumb:**
- $\\text{VIF}_j < 5$: Acceptable
- $5 \\leq \\text{VIF}_j < 10$: Moderate multicollinearity
- $\\text{VIF}_j \\geq 10$: Severe multicollinearity → Consider removing feature

**Effect of multicollinearity:**
- ❌ Large standard errors (imprecise estimates)
- ❌ Unstable coefficients (change drastically with small data changes)
- ❌ Inflated p-values (may miss significant effects)
- ✅ **Predictions are still valid** (multicollinearity affects inference, not prediction)

---

## Separation Issues

**Complete separation:** A feature perfectly predicts the outcome.

**Example:** If all $y_i = 1$ when $x_{ij} > 10$ and all $y_i = 0$ when $x_{ij} \\leq 10$, then $\\hat{\\theta}_j \\to \\infty$.

**Symptoms:**
- Coefficients blow up ($|\\hat{\\theta}_j| > 10$)
- Standard errors become huge
- Optimization fails to converge

**Solutions:**
1. **Penalized MLE (Ridge/Lasso):** Add $\\frac{\\lambda}{2} \\|\\theta\\|^2$ to cost
2. **Firth's correction:** Bias-reduced estimator
3. **Remove separating feature** (if not of interest)

---

## Summary: Complete Inference Workflow

**Step 1:** Fit model to get $\\hat{\\theta}$

**Step 2:** Compute Fisher Information: $I(\\hat{\\theta}) = X^T R X$

**Step 3:** Invert to get covariance: $\\text{Cov}(\\hat{\\theta}) = I(\\hat{\\theta})^{-1}$

**Step 4:** Extract standard errors: $\\text{SE}(\\hat{\\theta}_j) = \\sqrt{[\\text{Cov}(\\hat{\\theta})]_{jj}}$

**Step 5:** Compute Wald statistics: $z_j = \\hat{\\theta}_j / \\text{SE}(\\hat{\\theta}_j)$

**Step 6:** Compute p-values: $p_j = 2[1 - \\Phi(|z_j|)]$

**Step 7:** Compute confidence intervals: $\\hat{\\theta}_j \\pm 1.96 \\cdot \\text{SE}(\\hat{\\theta}_j)$

**Step 8:** Convert to odds ratios: $\\text{OR}_j = e^{\\hat{\\theta}_j}$

**Step 9:** Likelihood ratio test: $\\Lambda = 2[\\ell(\\hat{\\theta}) - \\ell(\\hat{\\theta}_0)]$

**Step 10:** Compute pseudo-$R^2$ for goodness of fit

---

## Key Takeaways

1. **Standard errors quantify uncertainty** in parameter estimates
2. **Wald tests** determine statistical significance of individual features
3. **Confidence intervals** provide a range of plausible values
4. **Odds ratios** give interpretable effect sizes
5. **Likelihood ratio test** assesses overall model fit
6. **Multicollinearity** inflates standard errors but doesn't affect predictions
7. **Separation** causes numerical instability → Use regularization

**Inference is essential for understanding which features matter and how confident we are in our conclusions.**
"""

# =============================================================================
# CREATE ALL NOTEBOOKS
# =============================================================================

print("Generating theory notebooks...")

# Create each notebook
notebooks = [
    ("Theory/01_Bernoulli_MLE.ipynb", bernoulli_content, "Bernoulli MLE"),
    ("Theory/02_Logit_Link.ipynb", logit_content, "Logit Link"),
    ("Theory/03_GLM_Proof.ipynb", glm_content, "GLM Proof"),
    ("Theory/04_Hessian_PSD.ipynb", hessian_content, "Hessian PSD"),
    ("Theory/05_Fisher_Information.ipynb", fisher_content, "Fisher Information"),
    ("Theory/06_Inference_Theory.ipynb", inference_content, "Inference Theory")
]

for filepath, content, name in notebooks:
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(content))
    
    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"✅ Created {name}")

print("\n🎉 All 6 theory notebooks created successfully!")
print("\nFiles created:")
for filepath, _, name in notebooks:
    print(f"  - {filepath}")
