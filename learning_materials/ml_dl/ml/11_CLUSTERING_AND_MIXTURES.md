# ML11. Clustering and K-means

Clustering groups examples without using target class labels. K-means describes each group by a center, assigns training points to their nearest center, then moves each center to the mean of its assigned points. Repeating these two steps seeks compact groups; it does not establish that the groups are meaningful real-world categories.

Training points [0, 2, 8, 10] start with two centers at 0 and 10. Their assignments move the centers to 1 and 9. A separate query control lets you inspect where another point would be assigned before that update.

Classification uses known labels; clustering does not. The additional notes compare K-means with Gaussian mixtures, which allow soft membership, and DBSCAN, which groups by density.

[course map](../COURSE_MAP.md). Required background: ML02, P3.

## 1. Clustering assumptions and group shapes

**Key terms.** A **cluster** is an algorithm-defined group; a **centroid** is its representative mean; a **mixture model** assigns probabilities to components. **GMM** means Gaussian mixture model. **EM (expectation-maximization)** alternates probabilistic membership estimates, called responsibilities, and parameter fitting. A **density cluster** follows connected dense regions.

**Process.** Choose features and geometry without using evaluation labels, fit the cluster model and inspect assignments or responsibilities. Evaluate stability and usefulness for a stated downstream purpose before attaching human meaning to groups.

K-means alternates nearest-centroid assignment and centroid means, reducing within-cluster squared-distance objective at its local steps. It favours particular geometric shapes and scales. Multiple initializations reduce sensitivity to a poor local solution but do not turn clusters into ground-truth categories.

Gaussian mixtures allow covariance and soft membership under a density model. DBSCAN connects sufficiently dense neighbourhoods and can mark noise; hierarchical clustering merges or splits groups under a linkage rule. Each encodes different assumptions. A visually attractive projection is not independent evidence that the original data contains meaningful classes.

## 2. A worked calculation

**Symbols and assumptions.** k is cluster count, mu_j is centroid j and c_i is sample i's assignment. The k-means objective sums ||x_i−mu_(c_i)||² over cases.

For one-dimensional points [0,2,8,10] and initial centroids [0,10], assignments are [0,0,1,1]. Updated means are [1,9]; objective becomes (0−1)²+(2−1)²+(8−9)²+(10−9)²=4. Before and after this update, a point at 5 is tied and needs a deterministic rule. These are arithmetic clusters, not discovered social categories.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Choosing K-means, mixtures or DBSCAN

K-means needs k and a restart policy. DBSCAN uses radius eps and minimum neighbourhood population; changing scaling changes the meaning of eps. Mixture covariance type and regularization trade flexibility against unstable estimates. Silhouette scores compare distance separation but do not prove external usefulness.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: inspect clusters without using labels to fit

Use Wine labels only for a post-hoc diagnostic after unsupervised fitting, clearly labelled as such. Compare scaled/unscaled clustering and restart stability. Do not select preprocessing using label agreement and still call the whole procedure label-free.

**Check.** Why can asking for three clusters produce three groups even when no three natural groups exist?

**Answer.** The optimization is constrained to that requested k. A returned solution satisfies the algorithm's setup, not the existence of a true three-category world.

## 5. How a Gaussian mixture learns soft assignments

**Key terms.** A mixture weight π_k is a component's prior proportion. A responsibility r_ik is the posterior probability that component k generated point i. EM alternates expectation and maximization.

**Process.** Initialize component parameters, calculate responsibilities under the current mixture, update weights/means/covariances using those responsibilities and repeat until the likelihood stopping rule or iteration limit.

A Gaussian mixture describes several bell-shaped groups at once. For each point, calculate how plausible it is under each group and normalize these values into soft memberships. A membership of 0.8 means the point contributes weight 0.8 to that component's update, not that we have established a true real-world category. In the formula, the normal-density term uses that group's mean and covariance, and N_k adds its fractional memberships.

\[
r_{ik}=\frac{\pi_k\,\mathcal N(x_i;\mu_k,\Sigma_k)}
{\sum_j\pi_j\,\mathcal N(x_i;\mu_j,\Sigma_j)},\quad
N_k=\sum_i r_{ik},\quad
\mu_k^{new}=\frac{\sum_i r_{ik}x_i}{N_k}.
\]

For scalar points [0,2] and responsibilities [0.8,0.2] for one component, N_k=1 and its updated mean is 0.4. The other component with [0.2,0.8] has mean 1.6. Hard K-means would assign each point to one center rather than splitting its contribution.

### What can go wrong?

A Gaussian component can collapse around very few points and drive covariance toward singularity; covariance regularization prevents numerical collapse but does not prove a meaningful grouping. Different initializations can reach different local optima. A fitted group need not correspond to a human category.

The lab never supplies reference class labels to the mixture fit. Adjusted Rand compares its groups with those withheld labels afterward; selecting settings by that score introduces supervised guidance into the overall procedure.

## Sources

- [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html).
- [Deep Learning, author site](https://www.deeplearningbook.org/).
