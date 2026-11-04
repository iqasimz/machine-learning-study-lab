# DL10. Generative models: VAEs, GANs and diffusion

A generative model learns a way to produce new samples with patterns resembling its training data. The main families use different mechanisms: a variational autoencoder, or VAE, encodes a distribution over latent codes and decodes samples; a generative adversarial network (GAN) trains a generator against a real/fake discriminator; diffusion learns to remove controlled noise. Their objectives are not interchangeable quality scores.

The selected example shows one mechanism with authored numbers, not generated images or measured model quality. Choose a family to see its specific inputs, calculations and limitations; no generative model is trained by the selector.

Autoencoders introduced a code and reconstruction. This lesson distinguishes probabilistic codes, adversarial learning and denoising, before comparing what each objective actually encourages.

[course map](../COURSE_MAP.md). Required background: DL09, P3.

## 1. How generative families differ

**Key terms.** A **generative model** models or samples data; a **latent variable** is unobserved explanatory state. A **VAE** optimizes a likelihood bound; a **GAN** trains generator/discriminator objectives; **diffusion** learns to reverse a specified corruption process.

**Process.** Choose the model family and its training target, transform training examples into latent/noisy/adversarial learning cases, optimize the relevant networks and evaluate generation separately from training loss.

A VAE uses an approximate posterior encoder and decoder with a prior regularizer. Reparameterization expresses a latent sample through a differentiable transformation plus external noise. A GAN's discriminator classifies real versus generated examples while the generator learns against that signal; two coupled objectives can become unstable or collapse to limited outputs.

A diffusion noise-prediction recipe corrupts data at a sampled noise level and learns the noise under a declared parameterization. Sampling uses a reverse procedure and schedule, not one call to the training loss. These are different statistical mechanisms, not three interchangeable architecture names. Full modern image-generation engineering is beyond the first local lab.

## 2. A worked calculation

**Symbols and assumptions.** A VAE balances explaining an observed input with keeping its encoded distribution close to a chosen prior. The negative reconstruction-likelihood term and KL penalty are separate pieces. ELBO means evidence lower bound: with the standard construction, maximizing it corresponds to minimizing the associated negative bound, not to minimizing the KL term alone.

For a VAE with approximate q(z|x), prior p(z) and decoder p(x|z), the negative evidence lower bound is reconstruction negative log likelihood plus KL(q||p). beta-VAE scales the KL term by beta. epsilon denotes sampled noise in the diffusion illustration.

If VAE reconstruction cost is 10 nats and KL is 2 nats, beta=1 gives objective 12; beta=4 gives 18 with different gradients, not worse reconstruction measured directly. For a scalar diffusion corruption x_t=sqrt(a_bar)x_0+sqrt(1−a_bar)epsilon, x_0=1,a_bar=0.64,epsilon=0.5 gives x_t=0.8+0.6×0.5=1.1. The corrupted value need not stay in the original range.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Choosing and evaluating a generative objective

Compare PCA/autoencoder reconstructions before adding probabilistic latent training. GAN discriminator success is not a reliable standalone measure of generator quality. Diffusion steps, sampler and noise schedule affect different cost/quality aspects. Held-out likelihood, perceptual metrics, diversity and memorization tests answer different questions.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: compare only compatible evidence

Keep initial experiments to small authored distributions or bounded image reconstructions. No diffusion model, GAN or pretrained image generator is promised as a working dashboard control in stage 1.

**Check.** Why does a lower VAE objective not automatically mean every displayed image is sharper?

**Answer.** The objective trades probabilistic reconstruction and prior compatibility; visual sharpness is a different property. Changing beta changes the trade-off, so compare stated outcomes rather than only the scalar objective.

## 5. VAE training: learn a latent distribution and a decoder

A variational autoencoder uses a probabilistic decoder to explain an observation x from a hidden code z. Its encoder predicts an approximate posterior q(z|x), because the true latent posterior may be hard to compute. A prior p(z), often standard normal, makes it possible to sample codes without first supplying an observed x. The deterministic autoencoder in DL09 does not have this full probabilistic objective.

With a diagonal Gaussian encoder, predict mean mu and positive standard deviation sigma, then sample external standard-normal noise epsilon. Reparameterization writes z=mu+sigma×epsilon. It keeps a differentiable path through the encoder parameters while randomness comes from epsilon. An implementation commonly predicts log variance and computes sigma=exp(0.5 log variance) rather than allowing negative standard deviations.

\[
L_{VAE}=-\mathbb E_{q_\phi(z\mid x)}\log p_\theta(x\mid z)+\operatorname{KL}(q_\phi(z\mid x)\|p(z)),
\quad
\operatorname{KL}=\tfrac12\sum_j(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2).
\]

Minimize this negative ELBO; equivalently maximize its negative. The decoder likelihood determines the reconstruction term: a squared-error expression requires a stated Gaussian/noise convention, not just an arbitrary visual similarity. A finite sampled estimate is not the exact data log likelihood. Scaling KL by beta changes the objective; beta≠1 need not retain the original bound's interpretation.

For one dimension, mu=1 and sigma=0.5 give KL=0.5(1+0.25−1−log 0.25)=0.8181 nats. With sampled epsilon=−1, z=0.5. An estimated reconstruction log probability −2 gives estimated ELBO −2.8181 and minimized negative ELBO 2.8181. These are assumed values, not a generated image's quality rating.

Input → encoder distribution → sampled code → probabilistic decoder → reconstruction/prior loss is training. For unconditional generation, sample z from the prior and decode; no input image or posterior encoder is required. Posterior collapse, a poorly matched likelihood or an overly restrictive decoder/prior can limit usefulness.

**Practice.** Could a lower KL alone indicate success? **Answer.** No. An encoder that ignores x can match the prior while losing useful information. Inspect reconstruction, sampling and the complete objective. [VAE paper](https://arxiv.org/abs/1312.6114).

## 6. GAN training: two networks, two update objectives

A generative adversarial network trains a generator G to transform noise z into a sample and a discriminator D to distinguish real data from generated examples. D's output is a training discriminator score, not a universal image-quality or factuality measure. There is no VAE posterior encoder required by this basic formulation.

One common implementation minimizes these two losses:

\[
L_D=-\mathbb E_x\log D(x)-\mathbb E_z\log(1-D(G(z))),\qquad
L_G=-\mathbb E_z\log D(G(z)).
\]

The second is the non-saturating generator loss, not the original minimax generator expression. During a discriminator step, detached generated samples prevent an unintended generator update. During a generator step, D's parameters are held fixed but gradients must still pass through its operations to G. Freezing parameters is not equivalent to disabling the entire derivative graph.

If D(real)=0.8 and D(fake)=0.2, one real/fake pair contributes discriminator loss −log 0.8−log 0.8=0.4463. The generator's non-saturating loss is −log 0.2=1.6094 for that fake sample. They are different objectives and need not move together. Their numbers are not comparable to VAE negative ELBO or diffusion noise loss.

Noise → generated sample → discriminator feedback → selected network's update is the local computation. Alternate under a declared update ratio and evaluation protocol. A generator may collapse to a few convincing outputs instead of covering the data; a discriminator can overfit or overwhelm the other network. Wasserstein-style losses and gradient penalties change the formulation and require their own constraints, not a checkbox labelled “stable”.

**Practice.** D labels every fake correctly. Does that prove G is improving? **Answer.** No. It may describe a strong discriminator or a weak/collapsed generator. Evaluate held-out sample realism, coverage and memorization separately. [GAN paper](https://arxiv.org/abs/1406.2661).

## 7. Diffusion training: predict noise at a sampled noise level

A diffusion model specifies a fixed process that progressively corrupts data, then learns information needed to reverse that corruption. In a common noise-prediction formulation, the network receives noisy data x_t and a timestep/noise-level representation t; it predicts the sampled noise. It is not trained to classify the timestep or directly minimize an image-quality score.

Let beta_t be a forward noise variance, alpha_t=1−beta_t and alpha-bar_t the product of alphas through t. For original x_0 and fresh standard-normal noise epsilon:

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad
L_{simple}=\mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|^2.
\]

Training samples a permitted example, a timestep and noise, constructs x_t directly and calculates the declared loss reduction. It need not run the entire sequence of forward corruptions for each update. Noise-level sampling, parameterization (noise, clean data or velocity), weighting and network architecture are separate choices. A U-Net or another suitable denoising network predicts a same-compatible-shaped output; attention can incorporate conditioning.

For x_0=1, alpha-bar=0.64 and epsilon=−0.5, x_t=0.8−0.3=0.5. If predicted noise is −0.4, squared noise error is 0.01. A corresponding clean-value estimate is (0.5−0.6×(−0.4))/0.8=0.925. Noise loss and clean-value error therefore need not be numerically identical. Values can leave the original data range during corruption.

**Practice.** If training sampled timestep 50, must it execute fifty denoiser calls first? **Answer.** Not for this closed-form training construction. One noisy example can be constructed directly; generation has a separate iterative procedure. Do not interpret the tiny scalar calculation as a trained denoiser or a sufficient image-generation recipe.

The next section follows the reverse sampling path and adds conditioning. [DDPM training and sampling algorithms](https://arxiv.org/abs/2006.11239).

## 8. Diffusion sampling, conditioning and guidance

Sampling starts from noise and repeatedly calls a trained model according to a reverse-time sampler. The original training noise is not known at use time, so “subtract the original noise” is not an executable generation rule. A DDPM noise-prediction reverse step uses the following model-dependent mean and a declared variance:

\[
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)+\sigma_t z.
\]

Here z is fresh sampling noise when the chosen step uses it; the final step convention commonly removes it. The denoiser weights remain fixed. A sampler, schedule, number of network evaluations and random seed affect the produced result. DDIM supplies another trajectory family; fewer steps can reduce computation but need measured quality comparisons rather than a universal speed/quality claim.

**Conditioning** supplies requested information, such as a class, text representation or another image. A conditional denoiser can read that information through cross-attention or another declared mechanism. **Classifier guidance** uses an external classifier's gradients. **Classifier-free guidance (CFG)** instead combines conditional and unconditional predictions from a suitably trained system. Dropping conditioning on some training cases supports the latter; it does not mean training never uses conditions.

Using the convention below, scale s=0 is unconditional and s=1 is the conditional prediction:

\[
\hat\epsilon=\epsilon_u+s(\epsilon_c-\epsilon_u).
\]

If epsilon_u=0.2, epsilon_c=0.5 and s=2, the guided value is 0.8. This extrapolation is not automatically a better sample: guidance can trade diversity against adherence and introduce artifacts. Other papers use shifted guidance-scale conventions, so define yours before comparing settings.

**Practice.** Must changing the sampler retrain the denoiser? **Answer.** Not necessarily; compatible samplers can use the same model. Compatibility, parameterization and noise-level conventions must match. [DDIM](https://arxiv.org/abs/2010.02502) · [Classifier-free guidance](https://arxiv.org/abs/2207.12598).

## 9. Latent diffusion and other generation paths

Latent diffusion runs the noisy denoising process in a compressed representation, then decodes the resulting latent into data. This separates two learned components: an encoder/decoder that defines the representation, and a diffusion model operating in that space. Compression changes cost and what detail can be preserved; a lower-dimensional tensor is not automatically a semantically faithful one.

For a spatial example, an RGB image 256×256 contains 196,608 values. A hypothetical encoder output with four channels at 32×32 contains 4,096, or 48 times fewer values. This ratio counts tensor entries only. It is not a measured 48× speedup: denoiser widths, attention, decoding, precision and implementation all affect runtime. At generation, noise → latent denoising → image decoder is the path; encoding a real image instead is useful for image-to-image variants with a separately specified corruption/conditioning procedure.

The same high-level generative questions extend to audio or video, but their axes, timing, temporal consistency, rights and evaluation differ. A video denoiser must account for frames as well as spatial content; image realism alone does not establish coherent motion. A text condition does not turn an image-only decoder into an audio model.

Other families solve different formulations. An autoregressive generator samples the next discrete unit given a prefix. A normalizing flow uses an invertible change of variables and a tractable Jacobian determinant under its construction. Flow matching learns a vector field along a specified probability path and integrates it during sampling; it is not identical to an invertible likelihood flow or the noise-prediction formula above. These are comparison pointers, not extra implemented model families or prerequisites for the current diffusion sequence.

**Practice.** If the compression decoder removes small printed characters, can a perfect latent denoiser necessarily restore them? **Answer.** No. The representation/decoder can be a bottleneck. Evaluate both reconstruction and generation, including the details required by the intended use.

[Latent diffusion](https://arxiv.org/abs/2112.10752) · [Flow matching](https://arxiv.org/abs/2210.02747).

## 10. Evaluate generation: realism, coverage, conditions and memorization

Generation needs several questions, not a single leaderboard number. **Realism** concerns resemblance to plausible data; **coverage/diversity** concerns how much of the intended distribution is represented; **condition adherence** asks whether the requested content is present; **memorization** asks whether outputs reproduce training material. Safety, bias and rights add further requirements, not deductions from visual quality.

Fréchet Inception Distance (FID) compares Gaussian summaries of pretrained image-feature distributions. It does not directly judge prompt adherence, factual correctness or safety. For Gaussian feature means mu and covariances Sigma, it combines squared mean distance with a covariance term. Its value depends on the feature pipeline, sample count and reference data. A one-dimensional illustration is:

\[
FID=(\mu_r-\mu_g)^2+\sigma_r^2+\sigma_g^2-2\sigma_r\sigma_g.
\]

If the real summary has mean 0 and standard deviation 1, while the generated summary has mean 1 and standard deviation 2, this expression is 1+1+4−4=2. Two different distributions can have identical first two moments, so even this idealized summary can miss differences. Small-sample estimates introduce additional uncertainty and bias.

Freeze reference split, sample count, resolutions, preprocessing, condition set and sampler configuration before comparing. Inspect nearest training neighbors carefully, exact duplicates and sensitive-content failures under a lawful access policy. Similarity is a warning to investigate, not automatic proof of copying. Human studies need clear questions, appropriate consent and uncertainty; an LLM judge is not independent ground truth.

**Practice.** Model A has lower FID but repeatedly ignores requested object counts. Which model is better? **Answer.** The evidence is incomplete until the intended use and adherence measure are specified. Report the trade-off rather than declaring universal superiority. Retain a small fixed condition suite and several seeds for regression checks.

The local deterministic autoencoder does not implement VAE/GAN/diffusion training or image-generation evaluation merely because these lessons are present. [FID paper](https://arxiv.org/abs/1706.08500).

## Sources

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661).
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
