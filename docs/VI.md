<h1 style="font-size:46px;color:blue">Variational inference</h1>

# Motivation

Given some data $\mathcal{D}$ it is usually the case that we want to model the behaviour of our data given some hidden or latent variables. That is to say, there are some latent variables driving our data, but we don't have observations about them.

Assuming that's the case, future data can be predicted knowing the *behaviour* of these latent variables. For doing so, one might want to compute the *posterior* probability density of the latent variables $Z$ (where $Z$ denotes here the set of all the latent variables) *given* the observed data $X = \mathcal{D}$. 

!!! note "Posterior"
    $$
    p(Z | X = \mathcal{D}) = \dfrac{p(X | Z)p(Z)}{p(X)}
    $$

Here, $X = \mathcal{D}$ is used to emphasize that the data $\mathcal{D}$ taking specific values (i.e. $\mathcal{D} = \{ 1.2, 1.3, 1.8\}$), is withdrawn from some probability distribution whose random variable (or random vector) is denoted by $X$.

!!! tip "About X"
    The data $\mathcal{D}$ is driven either by a random variable $X$ with *independent and identically distributed* (i.i.d) samples $\mathcal{D} = \{ x_{i}\}_{i=1}^{N}$, or by a random vector $\mathbf{X}$ with entries $[\mathbf{X}]_{i} = x_{i}, \; i=1,..., N$, having a multivariate probability distribution. The difference between both cases is that in the latter case the observed data is correlated, while in the former case we assume independence of each sample.

!!! warning "Notation for X"
    For simplicity in either case we will use the notation $X$ to refer either to a random variable or random vector.


The problem with the *posterior* distribution is that for it to be a valid probability distribution function it requires the computation of the marginal probability $p(X)$.

A look at the marginal probability $p(X)$ shows that it becomes very easily an intractable computation:

!!! note "Marginal"
    $$
    p(X) = \int p(X,Z) dZ = \int p(X | Z) p(Z)dZ
    $$

For many *prior* $p(Z)$ and *likelihood* $p(X | Z)$, the product of them will make the integral intractable analytically, and computationally expensive through numerical methods. Thus, there is a need to look for candidate approximations (commonly referred as *surrogates*) for $p(Z | X)$.

# Surrogates formulation

In order to know if a surrogate $q(Z)$ is a good one, we need to measure the mismatch between $q(Z)$ and $p(Z | X)$.

Since both $q(Z)$ and $p(Z|X)$ are probability distributions, a natural measure for such mismatch is the KL-divergence:

!!! note "KL-divergence"
    $$
    D_{KL}(q(Z), p(Z)) = \int q(Z) \log \left( \dfrac{q(Z)}{p(Z)} \right) dZ = \mathbb{E}_{Z \sim q(Z)}\left[ \log \left( \dfrac{q(Z)}{p(Z)} \right) \right]
    $$

Note that $D_{KL}$ is not a *distance* measure and so, it is not symmetrical (meaning that $D_{KL}(q,p) \neq D_{KL}(p,q)$).

Next we plug in the posterior into the KL-divergence formula so that we have:

$$
D_{KL}(q(Z), p(Z | X)) = \mathbb{E}_{Z \sim q(Z)}\left[ \log \left( \dfrac{q(Z)}{p(Z | X)} \right) \right] \: \: \: \: \: (1)
$$

Note that this equation is not very helpful, since precisely what we are trying to get rid of is the posterior $p(Z | X)$. However, we can achieve that by rewriting the posterior:

$$
D_{KL}(q(Z), p(Z | X)) = \mathbb{E}_{Z \sim q(Z)}\left[ \log \left( \dfrac{q(Z) p(X)}{p(Z, X)} \right) \right] = \mathbb{E}_{Z \sim q(Z)}\left[ \log q(Z) \right] + \mathbb{E}_{Z \sim q(Z)}\left[ \log p(X) \right] - \mathbb{E}_{Z \sim q(Z)}\left[ \log p(Z,X) \right] \: \: \: \: \: (2)
$$

where we are using the definition of the joint probability distribution in terms of the conditional:

!!! note "Joint distribution from conditional"
    $$
    p(Z| X) = \dfrac{p(Z, X)}{p(X)}
    $$

We can now state the variational inference problem or task:

!!! note "Variational inference optimization task"
    To find $q^{*}(Z)$ such that:

    $$
    q^{*}(Z) = arg\min_{q} D_{KL}(q(Z), p(Z | X))
    $$

## Unfold the optimization task

Note that we still have no access to the *evidence* or *marginal* $p(X)$ in Equation (2), and therefore we can't compute the KL-divergence. We can, however, overcome this difficulty by also noting that, from its definition, we have that $D_{KL}(q,p) \geq 0$. Let's use this inequality into Equation (2):

$$
\mathbb{E}_{Z \sim q(Z)}\left[ \log p(X) \right] \geq \mathbb{E}_{Z \sim q(Z)}\left[ \log p(Z,X) \right] - \mathbb{E}_{Z \sim q(Z)}\left[ \log q(Z) \right] \: \: \: \: \: (3)
$$

and since the prior $p(X)$ does not depend on $Z$ we can simplify the first expected value:

$$
\log p(X) \geq \mathbb{E}_{Z \sim q(Z)}\left[ \log p(Z,X) \right] - \mathbb{E}_{Z \sim q(Z)}\left[ \log q(Z) \right] \: \: \: \: \: (4)
$$

We will use this inequality in a minute, but for now, let's use some notation to make equations shorter. Let's call $L(q(Z)) =: \mathbb{E}_{Z \sim q(Z)}\left[ \log p(Z,X) \right] - \mathbb{E}_{Z \sim q(Z)}\left[ \log q(Z) \right]$.

Note that when solving the variational optimization task, $p(X)$ remains constant. Therefore, we can rewrite Equation (2) as:

$$
D_{KL}(q(Z), p(Z | X)) - p(X) = \mathbb{E}_{Z \sim q(Z)}\left[ \log q(Z) \right] - \mathbb{E}_{Z \sim q(Z)}\left[ \log p(Z,X) \right] = -L(q(Z))\: \: \: \: \: (5)
$$

Since, as we mentioned, $p(X)$ is constant in the variational optimization task, we can therefore equivalently minimize $-L(q(Z))$. In order to use our previously found inequality, let's multiply Equation (6) by -1:

$$
-D_{KL}(q(Z), p(Z | X)) + p(X) = L(q(Z))\: \: \: \: \: (6)
$$

Equation (6) means that we can address the variational optimization task equivalently by *maximizing* $L(q(Z))$:

!!! note "Equivalent variational inference optimization task"
    To find $q^{*}(Z)$ such that:

    $$
    q^{*}(Z) = arg\max_{q} L(q(Z)) = \mathbb{E}_{Z \sim q(Z)}\left[ \log p(Z,X) \right] - \mathbb{E}_{Z \sim q(Z)}\left[ \log q(Z) \right]
    $$

Note in inequality (4) that when $p(X) = L(q(Z))$ we have $D_{KL}(q(Z), p(Z|X)) = 0$, and so the surrogate $q(Z)$ is no longer an approximation but instead is *exactly* equal to the posterior. 

Although this case is not achieved, inequality (4) provides a lower bound for the prior and also for the goodness of fit of our surrogate. For this reason, the quantity $L(q(Z))$ is called the ELBO (Evidence Lower BOund).

# Mean field variational family

The mean field variational family approach decomposes the surrogate $q(Z)$ into a product of independent distributions $q_{i}(Z_{i})$:

!!! note "Mean field variational family"
    $$
    q(Z), Z \in \mathbb{R}^{D} = \prod_{i=1}^{D} q_{i}(Z_{i})
    $$

Writing down the ELBO for this $q(Z)$ we have:

$$
\int_{Z} \prod_{i=1}^{D} q_{i}(Z_{i}) \log p(Z,D) dZ - \int_{Z} \prod_{i=1}^{D} q_{i}(Z_{i}) \log q(Z) dZ := I_{1} - I_{2} \: \: \: \: \: (7)
$$

where $\int_{Z} f(Z) dZ = \int_{Z_{1}} \int_{Z_{2}} \int_{...} \int_{Z_{D}} f(Z_{1}, ..., Z_{D}) dZ_{D}dZ_{D-1}...dZ_{1}$ is a D-integral and we introduce 

$$
I_{1} := \int_{Z} \prod_{i=1}^{D} q_{i}(Z_{i}) \log p(Z,D) dZ
$$

and 

$$
I_{2} := \int_{Z} \prod_{i=1}^{D} q_{i}(Z_{i}) \log q(Z) dZ
$$ 

to simplify each expression separately.

Let's start simplyfing $I_{1}$.

Let's assume we choose the index of one of $D$ latent variables. Let's call the chosen index $i$, while all the other indexes are 'captured' by the iterator $j$.

Therefore, we have:

$$
I_{1} = \int_{Z_{i}} \int_{Z_{j: \forall j \neq i}} q_{i} (Z_{i}) \prod_{\forall j \neq i}^{D} q_{j}(Z_{j}) \log p(Z, D) dZ_{j: \forall j \neq i} dZ_{i} \: \: \: \: \: (8)
$$

$$
I_{1} = \int_{Z_{i}} q_{i}(Z_{i}) \int_{Z_{j: \forall j \neq i}} \prod_{\forall j \neq i}^{D} q_{j}(Z_{j}) \log p(Z, D) dZ_{j: \forall j \neq i} dZ_{i} \: \: \: \: \: (9)
$$

$$
I_{1} = \int_{Z_{i}} q_{i}(Z_{i}) \: \mathbb{E}_{Z_{j: \forall j \neq i} \sim \prod_{\forall j \neq i} q_{j}(Z_{j})} \left[ \log p(Z, D) \right] dZ_{i} \: \: \: \: \: (10)
$$

where the notation $\int_{Z_{j: \forall j \neq i}} f(Z_{j: \forall j \neq i}) dZ_{j: \forall j \neq i}$ represents the $D-1$ integral over all the latent variables not indexed by $i$.

On the other hand, for $I_{2}$ we have that:

$$
I_{2} = \int_{Z_{i}} \int_{Z_{j: \forall j \neq i}} \left[ q_{i}(Z_{i}) \prod_{\forall j \neq i} q_{j}(Z_{j}) \right] \left[ \log q_{i}(Z_{i}) + \sum_{j: \forall j \neq i} \log q_{j}(Z_{j}) \right] dZ_{j: \forall j \neq i} dZ_{i} \: \: \: \: \: (11)
$$

$$
I_{2} = \int_{Z_{i}} \int_{Z_{j: \forall j \neq i}} q_{i}(Z_{i}) \prod_{\forall j \neq i} q_{j}(Z_{j}) \log q_{i}(Z_{i}) \: dZ_{j: \forall j \neq i} dZ_{i} + \int_{Z_{i}} \int_{Z_{j: \forall j \neq i}} q_{i}(Z_{i}) \prod_{\forall j \neq i} q_{j}(Z_{j}) \sum_{j: \forall j \neq i} \log q_{j}(Z_{j}) \: dZ_{j: \forall j \neq i} dZ_{i} \: \: \: \: \: (12)
$$



Mean field variational family disadvantage is that it doesn't approximate well Multimodal posterior distributions.

<div style="margin-bottom:66px"><div>