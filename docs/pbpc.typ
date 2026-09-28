// Self-contained source and figures. From barcodecnv/: typst compile docs/pbpc.typ docs/pbpc.pdf
#set document(title: "BarcodeCNV: barcode pseudobulk CN inference", author: "Jamie D Matthews", date: datetime(year: 2026, month: 9, day: 28))
#set page(paper: "a4", margin: (x: 21mm, top: 19mm, bottom: 19mm),
  header: context if counter(page).get().first() > 1 {
    text(8pt, fill: rgb("586575"))[BarcodeCNV / Python methods draft · 28 September 2026]
  }, footer: context align(right, text(8pt, fill: rgb("586575"))[#counter(page).display()]))
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: .58em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 16pt, fill: rgb("163b47"))
#show heading.where(level: 2): set text(size: 11.5pt, fill: rgb("163b47"))
#set math.equation(numbering: "(1)")
#show raw: set text(size: 8.5pt)
#let note(body) = block(width: 100%, inset: 10pt, radius: 3pt, fill: rgb("eef4f5"), body)
#let doi(id, body) = link("https://doi.org/" + id, body)
#let tbl(..cells) = table(columns: (1fr, 1fr), inset: 5pt, stroke: .35pt + rgb("d5dfe3"), ..cells)

#text(10pt, tracking: 1pt, fill: rgb("586575"))[METHODS / PROTOTYPE DRAFT]
#v(3mm)
#text(27pt, weight: "bold", fill: rgb("163b47"))[BarcodeCNV]
#v(1mm)
#text(17pt)[Barcode pseudobulk copy-number inference]
#v(2mm)
#text(10pt, fill: rgb("586575"))[Rationale, algorithm, uncertainty and evaluation · Python implementation]
#v(3mm)

#text(12pt, weight: "bold", fill: rgb("163b47"))[Abstract]

BarcodeCNV uses static lineage barcodes to pool sparse single-cell RNA counts, identify supported groups of genetically similar barcodes, and estimate their copy-number alterations. The aim is to support genotype-phenotype analysis while leaving weakly supported memberships unresolved. Grouping starts from self-centred, InferCNV-style smoothed expression and uses recursive binary splitting, with whole-cell bootstraps to assess split support and membership stability. Count-based CN contrasts and local smoothed haplotype-fraction (HF) differences can further subdivide these groups. An external normal expression reference, supplied directly or fitted as a mixture of available normal profiles, supports CN calling. A default 5% CN-independent depth-outlier component limits the influence of individual genes inconsistent with that reference. A dual-signal hidden Markov model (HMM) combines unsmoothed expression counts and phased allele counts to report uncertain CN states for each barcode and for directly pooled cells within each final group. A permutation diagnostic separately tests for barcode-associated regional signal.

#pagebreak()
= Rationale and scope

Static lineage barcodes provide a natural unit for pooling sparse single-cell RNA measurements. The resulting pseudobulks differ in their information about copy number: ten well-covered cells can be more informative than hundreds of shallow cells, especially at regions distinguishing candidate genotypes. BarcodeCNV uses these pseudobulks to identify expression-supported groups, refine them with copy-number and haplotype evidence, and report uncertain copy-number profiles.

The motivating application is genotype-phenotype analysis. Merging genetically different populations can obscure or create associations. Leaving weakly supported barcode memberships unresolved is therefore useful. The tool reports groups, unresolved barcodes, bootstrap membership stability and count-based CN probabilities; it does not require a predetermined number of reporting groups.

The method combines an adaptation of #link("https://github.com/broadinstitute/infercnv")[Broad Institute InferCNV]'s expression preprocessing with depth and phased-allele modelling derived from #doi("10.1038/s41587-022-01468-y")[Gao et al.'s Numbat]. The Python implementation uses NumPy, SciPy and Numba for numerical work. Its expression-based grouping rules, CN-path contrasts and local haplotype refinement are pragmatic reporting procedures, rather than one joint probability model of clone identity.

#note[*Current scope.* The standalone Python project, package and executable are named `barcodecnv`. Each barcode is assumed to have one CN profile. There is no inferred ancestral tree or latent clone-partition posterior. Grouping uses bootstrap stability, whereas CN calling uses conditional HMM probabilities. The distinction is intentional and matters when interpreting confidence.]

== How it works, in plain language

+ *Add together cells carrying the same barcode.* Sum their gene counts and phased allele counts, keeping the total amount of RNA measured. These are the barcode pseudobulks.
+ *Find repeated patterns along chromosomes.* Smooth expression and compare each barcode with the cohort's median expression, so initial grouping does not depend on an external normal reference. Propose a division into two groups, check its support, and repeat inside supported groups. Resample whole cells within barcodes to see whether divisions and memberships persist. Leave unstable memberships unresolved; the number of final groups is not chosen in advance.
+ *Choose a normal-expression baseline for calling.* Use a supplied reference or fit a mixture of available normal profiles. Allow each gene's count a 5% prior chance of coming from a broad outlier distribution, so an isolated reference discrepancy need not force a CN change. This softens gene-level mismatch; it does not integrate uncertainty in the fitted reference or correct every kind of mismatch.
+ *Explain the actual counts.* Use expression and phased allele counts together to compare gains, losses, ordinary copy number and copy-neutral loss of heterozygosity. The HMM favours continuous chromosome segments and retains alternative explanations where evidence is weak. It uses unsmoothed counts rather than treating smoothed expression values as independent measurements.
+ *Look for further supported differences.* Draw complete possible CN profiles from each barcode HMM and test contrasts inside expression groups. Then look for local differences in smoothed HF, which can distinguish retained haplotypes even when broad CN classes agree. Resampling assesses support for these HF subdivisions. These refinements split existing groups; they do not merge groups or recover already unresolved members.
+ *Report both barcode and group profiles.* Keep individual barcode calls, including unresolved barcodes. Also pool cells directly within each final group and call that group again, fitting its noise at the new depth. Display uncertain members and confident disagreements alongside pooled calls. Group calls assume the selected membership is fixed; they do not remove uncertainty about whether grouping was correct.

A separate permutation test asks whether the supplied barcodes explain regional count structure beyond chance. It shuffles barcode labels among comparable cells; it does not fit HMMs during shuffling. A small p-value supports using barcode-associated information, but does not establish that every group or CN call is correct.

#pagebreak()
= Inputs and expression features

== Cell counts, reference and genomic coordinates

The input is a cell-count bundle containing a sparse gene-by-cell expression matrix, per-cell phased H1/H2 counts, cell-to-barcode labels, exchangeability blocks, gene/SNP positions, a genetic map and an external diploid expression reference. A reference is required for the full workflow but not the separate association diagnostic. The Python workflow can fit a normal expression panel and perform BAM-to-allele counting/phasing before constructing this bundle.

Let $b$ index barcodes, $g$ genes, $l$ SNPs and $t$ positions in the sorted union of gene midpoints and SNP sites. The barcode expression count is $Y_(b,g)$, and its diploid expectation is
$ E_(b,g) = r_g sum_(i in b) L_i, $
where $r_g$ is the supplied reference fraction and $L_i$ is the whole-assay RNA library size of cell $i$. Library sizes are calculated before restricting to annotated/reference-matched genes. A positive expected count with an observed zero is informative; a gene without a usable reference contributes no depth evidence. Distinct genes remain separate likelihood terms even at a shared genomic position.

The HMM treats the supplied reference as fixed. A well-matched assay and cell type are desirable; exact donor matching is not assumed. Within-barcode genomic heterogeneity, absolute ploidy and uncertainty in fitted reference-panel weights are not explicitly inferred.

== InferCNV-style smoothing

Grouping uses a cohort-relative reference: the median across barcodes of each gene's count divided by whole-library exposure. This reference is rescaled to the median selected-gene pseudobulk library. The `infercnv_smoothing` function applies the following transform to the pseudobulks and reference together:

+ Remove rows that are zero in every pseudobulk and the reference.
+ Normalize each column to the median selected-gene library size, then apply $log_2(1+x)$.
+ Subtract the mean reference log expression for each gene and clip to $[-3,3]$.
+ Smooth within chromosomes using a 101-gene triangular window, renormalizing its weights at chromosome ends.
+ Subtract each column's genome-wide median, then subtract the smoothed reference mean for each gene again.

Missing rows return as missing values. This preprocessing follows InferCNV 1.28.0 through step 14, with numerical tests against R-generated fixtures. It does not invoke InferCNV's HMM. The external-reference expression panel uses the same transform with the supplied reference instead of the cohort median. Both expression panels are relative, recentered signals; their signs need not correspond exactly to absolute CN calls.

For recursive grouping, adjacent sets of 25 genes within each chromosome are reduced to their mean smoothed signal multiplied by the square root of the number of genes. The HMM itself retains unsmoothed gene and SNP count records on the fine grid.

== Whole-cell resampling

The default is 64 bootstrap replicates. Within each barcode, sample its observed cells with replacement, retaining its cell count. Recompute pseudobulk counts, library exposures, cohort reference, smoothing and reduced features for each replicate. Whole-cell sampling retains covariance among genes measured in a cell. Smoothed genes are not treated as independently resampled observations.

Write $x_b$ for the observed feature vector and $x_b^((r))$ for replicate $r$. A barcode's precision weight is inversely proportional to its mean bootstrap feature variance, with a numerical floor. Thus grouping precision reflects observed variation, rather than using cell count alone. With very few cells, empirical resampling cannot reveal variation absent from those cells; bootstrap stability is not automatically calibrated in this limit.

#pagebreak()
= Recursive expression grouping

== Candidate divisions and support

Within a candidate node, center the feature vectors using precision weights and compute the leading component of the weighted feature matrix. Its largest squared singular value is the node's variation statistic. Propose two children from the signs of the leading projection and update assignments to their closest weighted centroid until stable or 30 iterations have elapsed.

A second pass uses a precision core: the highest-weight half of the node, retaining at least four barcodes. Average-linkage correlation clustering proposes a binary division of this core; other barcodes are attached to the closest core centroid. This allows consistent structure among informative barcodes to propose a division when noisier members obscure it.

Each pass uses the same checks:

+ There must be at least four barcodes in the node and at least two in each proposed child.
+ Centered bootstrap residuals define a no-difference reference distribution. Recompute the leading variation statistic in each residual replicate. Its empirical tail fraction, including the observed sample, must be at most 0.05.
+ Within both children, precision-weighted mean pairwise correlation of centered profiles must be at least 0.1. In the core pass, only core members enter this coherence calculation.
+ For each barcode, compare its replicate feature vector with replicate child centroids, excluding that barcode from its own centroid. It must return to its proposed child in at least 90% of replicates. Each child must retain at least two such stable members.

If these checks pass, recurse on the stable members of each child. Unstable members become unresolved, rather than being forced down a branch. If a division is unsupported, retain the current node as a group when it has at least two barcodes.

The final expression partition is the common refinement of the two passes: barcodes must have positive labels in both, and a joint label combination must contain at least two barcodes. Group zero denotes unresolved membership. These reporting groups are not limited to five; the separate phase-pooling partition described below is.

== Interpreting bootstrap stability

Stability is the minimum accepted-branch membership stability along the relevant recursive route. A value of one can also mean that no accepted split challenged a barcode's membership; it is not proof that the barcode belongs to a biologically homogeneous clone. Node tables record candidate members, support statistics, coherence and reasons for accepting or rejecting each division.

The empirical tail fractions account for selecting a leading direction within the tested node. They do not provide a family-wise error guarantee for adaptive recursion, and membership stability is not a posterior probability of clone identity. Precision weights are estimated from the same bootstrap sample. These features make the procedure a data-dependent heuristic whose operating characteristics require empirical evaluation.

#note[*Small does not necessarily mean uncertain.* Two barcodes with ten cells can have different stability if only one covers the regions distinguishing its candidate group. Conversely, many cells cannot repair an uninformative contrast or a systematic reference mismatch. Neither a barcode-size cutoff nor a global confidence threshold establishes identifiability.]

== A separate partition for phase alignment

Average-linkage correlation clustering of the full cohort-relative smoothed profiles chooses two to five phase-pooling groups by maximum mean silhouette. Fewer than three barcodes or identical profiles use one group. These groups pool counts for estimating a common haplotype alignment. They are distinct from the final reporting groups, and later refinements do not change them or restart this fit.

#pagebreak()
= Count likelihoods and spatial priors

== Depth with an explicit outlier component

For CN state $c$, let $f(c)$ be its expression fold relative to diploid. The default inlier model is Poisson-lognormal (PLN):
$ Y | U,c ~ "Poisson"(E f(c) exp(U)), quad U ~ "Normal"(mu,sigma^2). $
The supplied reference times dosage is a *latent median rate*: the inlier mean is $E f(c) exp(mu + sigma^2/2)$. The automatic fit keeps $mu=0$ and estimates $sigma$ through the coordinate $alpha=exp(sigma^2)-1$. The reference parameterization therefore matters when comparing expression panels with calls.

The default likelihood permits a gene's count to arise from a broad, CN-independent component:
$ p(Y|c) = (1-epsilon) p_("PLN")(Y; E f(c) exp(mu),sigma)
  + epsilon p_("PLN")(Y; E exp(mu),tau). $
Here PLN's second argument is its latent median rate, $epsilon=0.05$, and $tau=3$ in natural-log units. The second term is identical across all CN states. An extreme expression discrepancy can consequently contribute little CN discrimination, while informative genes retain their signal. This is a prior mixture probability per gene observation, not a fixed fraction of genes discarded or a false-call rate. It does not describe correlated regional reference errors.

The same setting enters pooled phase/noise fitting, barcode calls and group pseudobulk calls. Use `--depth-outlier-probability` to change it. Consider a lower value, such as 0.01, when there is evidence of a well-matched reference; zero disables it. Inlier dispersion is refitted. The optional NB inlier family uses mean $E f(c) exp(mu)$ and variance $m+alpha m^2$; its outlier component remains PLN.

== Phased allele counts

For a SNP with total count $N$ and oriented H1 count $A$, the emission is
$ A | N,c,F ~ "BetaBinomial"(N,kappa q(c),kappa(1-q(c))). $
The binary phase state $F$ selects H1 or H2 as $A$. Before a fixed logit-bias adjustment, $q(c)=(c_1+a_0)/(c_1+c_2+2a_0)$; the zero-copy state uses $1/2$. Defaults are $a_0=0.1$ and $kappa=20$. The bias is estimated once from the aggregated allele counts, clipping the H1 fraction to $[0.45,0.55]$, and is held fixed during dispersion fitting.

A SNP log likelihood is multiplied by its supplied heterozygosity probability. This is a weighted likelihood, not an explicit mixture over germline genotypes. Gene weights, where supplied to the lower-level count API, similarly multiply gene log likelihoods. The outlier component modifies depth only; it does not reduce allele evidence.

== Copy states and transitions

There are ten oriented states: diploid $(1,1)$; deletion $(1,0)$; CN-LOH $(2,0)$; gain $(2,1)$; amplification $(5,1)$; and zero-copy deletion $(0,0)$, with reversed orientation for each asymmetric state. Folds are 1, 0.5, 1, 1.5, 3 and 0.05 respectively. Reports sum these into diploid, gain, loss and CN-LOH.

The chromosome-start prior $pi$ assigns 0.8 to diploid and shares the remainder equally among the five other configurations, then among their orientations. With genomic separation $Delta_t$ and $ell=30$ Mb,
$ T_t(i,j)=rho_t bb(1)[i=j]+(1-rho_t)pi_j, quad rho_t=exp(-Delta_t/ell). $
Chromosome boundaries reset to $pi$. This is a persistence prior, not a minimum event length. For genetic distance $d$ in centimorgans, the phase-switch probability is $(1-exp(-2 r d))/2$, with $r=1$. Phase starts uniformly at each chromosome, advances on entry to SNPs and persists through gene-only positions. Population-phase unary weights default to zero.

#pagebreak()
= Fitting and HMM computation

== Pooled alignment and fitted noise

Pool cells within the phase-pooling groups. Initialize dispersion from the median absolute deviation of positive log count/reference ratios. At fixed dispersion, alternate CN forward-backward inference and a shared two-state phase HMM. Each update uses the other factor's expected log likelihood. Two initializations, supplied population orientation and pooled allele imbalance, are tried; choose the fit with the larger variational bound. This is a structured variational approximation for shared phase.

Then maximize the summed group-HMM log evidence over a single common dispersion, holding the fitted phase probabilities fixed, and refit phase at that dispersion. The default phase cap is 60 iterations with maximum probability change below $10^(-4)$. The scalar SciPy bounded search uses log-$sigma$ coordinates, a search interval $[0.02,2]$ and a 35-iteration cap; the heuristic initial estimate is retained if the optimizer does worse. Boundary estimates and convergence status are recorded. Failed fits do not report a completed workflow.

== Joint CN and phase within each barcode

Default barcode calling uses a separate 20-state HMM for each barcode, with joint state $S_(b,t)=(C_(b,t),F_(b,t))$. Its transition is the product of the CN and phase transitions. The emission sums unsmoothed depth and allele log likelihoods at their respective fine-grid positions. A new common barcode-level dispersion maximizes the sum of barcode joint-HMM log evidences; it is not copied from the phase-pooling fit.

Conditioning on fitted nuisance values $hat(theta)$ and fixed reference $E$, the target for each barcode is
$ p(S_b | D_b,hat(theta),E) prop p(S_(b,1))
  product_(t=1)^T e_(b,t)(S_(b,t))
  product_(t=1)^(T-1) T_t^S(S_(b,t),S_(b,t+1)). $
The shared pooled phase fit supplies alignment for cross-barcode HF features; the default barcode HMM does not condition on that phase path. A global orientation can remain unidentified even when CN is well resolved. The optional `--barcode-phase conditional` route instead uses expected log allele likelihoods under the pooled phase probabilities and a ten-state CN HMM.

== Forward-backward and complete path draws

The forward recursion combines local emissions with the initial and transition priors:
$ a_1(s)=pi_s^S e_1(s), quad
  a_(t+1)(j)=e_(t+1)(j) sum_i a_t(i)T_t^S(i,j). $
Its terminal sum is the evidence for that chain. A backward recursion yields position marginals proportional to $a_t(s)b_t(s)$. Summing over phase and detailed CN states produces the barcode-by-marker-by-four-class probability tensor. Priors are already included in these calculations.

For distance calculations, sample the terminal state from its posterior and draw backward using
$ p(S_t=i | S_(t+1)=j,D,hat(theta)) prop a_t(i)T_t^S(i,j). $
Each draw is one coherent posterior path with spatial dependence retained. The default is 256 independent paths per barcode, conditional on fitted parameters. Drawing each position independently would discard that dependence. This is direct HMM sampling, not an iterative chain over fitted parameters: there is no warmup or MCMC convergence diagnostic. More draws reduce Monte Carlo variation in the downstream contrast summaries, not uncertainty in fitted noise or reference.

#note[*What is integrated and what is fixed?* Default barcode CN and local phase paths are integrated jointly. Dispersion, allele bias, reference, CN/phase hyperparameters and grouping choices are fixed estimates or settings. The method does not integrate their uncertainty into the reported CN probabilities.]

#pagebreak()
= Refinement with CN paths and haplotype fractions

== Cell-weighted CN contrasts

Refinement only subdivides existing positive expression groups. At each draw $r$, compute the fraction of retained genes at which two barcode paths have different *broad* CN classes:
$ d_(a,b)^((r)) = frac(1,G) sum_g bb(1)[C_(a,g)^((r)) != C_(b,g)^((r))]. $
Here $C$ denotes the collapsed class for this calculation; opposite retained haplotypes within CN-LOH count as the same class. Each gene contributes equally, irrespective of its expression level. This is gene-fraction distance, not physical genome fraction or inferred evolutionary distance.

Build an average-linkage hierarchy from mean distances. For candidate children $A,B$, average between-barcode distances with weights $n_a n_b$, where $n_b$ is barcode cell count. Let $D_(A,B)^((r))$ be the between-child average and $W_A^((r)),W_B^((r))$ the within-child averages over distinct barcode pairs. Define
$ delta^((r)) = D_(A,B)^((r)) - (W_A^((r))+W_B^((r)))/2. $
Accept a split only when both children have at least two barcodes, the fifth percentile of $delta$ is positive, and $bb(E)[delta]/bb(E)[D_(A,B)] >= 0.25$. Recursively inspect children of accepted splits; otherwise retain the node. These conditional contrast summaries are not posterior probabilities of a clone partition. Already unresolved barcodes remain unresolved.

== Local smoothed haplotype refinement

The next stage can distinguish allele patterns that broad CN distance misses. Use the MAP pooled phase orientation to align per-cell H1/H2 counts; weight counts by supplied SNP heterozygosity probabilities. Pool them into 2 Mb base bins and combine five adjacent bins into 10 Mb windows at 2 Mb spacing, without crossing chromosomes or bridging gaps outside the physical window.

For each base bin, use a weak Beta$(1/2,1/2)$ contribution. The observed window fraction pools H1 plus half a count per represented bin over total counts plus one per represented bin. Transform to signed fraction $2"HF"-1$ and subtract the median across barcodes at each window.

For resampling, draw whole cells within barcodes, draw one beta fraction per base bin, and use the same base-bin draw in every overlapping window. Weight the window average by total counts plus prior mass and repeat cohort centering. This preserves overlap-induced covariance. Whole-cell and beta resampling overlap as uncertainty sources, and MAP phase omits orientation uncertainty; the resulting stability remains heuristic.

Apply the two recursive splitting passes and their common refinement independently within every current group. This stage can produce unresolved members, but cannot merge separated groups or recover members already unresolved by expression grouping. It is enabled by default and skipped automatically when no usable allele counts exist.

== Ordering for display

Rows are ordered by final reporting group, then by the observed expression hierarchy within each group; unresolved barcodes appear last. The same order and group boundaries are used in all modalities. No simulation truth enters feature construction, inference, grouping or production plot ordering.

#pagebreak()
= Group profiles and uncertainty reporting

== Literal group pseudobulks

After final grouping, sum member cells' expression counts per gene and H1/H2 counts per SNP. Sum whole-library exposures to obtain group reference expectations. Fit a new common dispersion across these group pseudobulks, then infer a joint CN/phase path distribution for each group. Group zero is excluded from this pooling.

Each reported group therefore has three complementary summaries:

#tbl(
  [*Summary*], [*Interpretation*],
  [Barcode CN probabilities], [Individual barcode evidence; these remain available for unresolved barcodes too.],
  [Cell-weighted consensus], [Average of barcode class probabilities, describing the inferred composition of the group.],
  [Group pseudobulk probabilities], [Conditional state probabilities from summed cell counts, assuming a homogeneous CN/phase profile for the selected group.])

The consensus is not a probability distribution over one shared group state. The group pseudobulk is conditional on fixed membership and does not feed back into grouping or barcode calls. Its increased coverage can resolve an event that was uncertain in individual members, but it can also conceal heterogeneity if the group is inappropriate.

For each group and position, report the cell-weighted fraction of barcodes whose maximum class probability is below 0.95. Also report the fraction belonging to confident barcodes whose MAP class disagrees with the consensus, and separately with the pooled call. These are descriptive disagreement measures, not tests of homogeneity. Cell weights apply to barcode calls; the tool does not claim independent CN measurements for each member cell.

== What the figures show

`summary.png` stacks cohort-relative expression, external-reference expression, rolling HF when available, and barcode CN calls. CN hue identifies the most probable broad state; saturation represents its marginal probability. The separate stability strip summarizes the minimum expression/HF membership stability. White expression/HF gaps indicate missing display support, not a diploid call. Group boundaries align across modalities.

`group_cn.png` separates cell-weighted consensus, literal-pseudobulk calls, uncertain member fractions and confident disagreement with the pooled call. A smooth-looking profile is not, by itself, evidence of biological accuracy.

#note[*Keep three kinds of information separate.* Production output consists of calls, conditional probabilities, bootstrap stability, disagreement and diagnostic statistics. Ground truth consists of simulated genotypes, clone labels and ancestry. Purity, completeness and call error are evaluations that combine the two and are unavailable on ordinary real data.]

== Segment tables

Segments are contiguous runs of the marginal MAP broad class on the fine gene/SNP grid, split at chromosome boundaries. Their bounds are the first and last observed markers, not precise breakpoints. Reported mean/minimum probabilities summarize marker marginals, not the probability of the whole segment. Short runs can persist when local data outweigh the spatial prior; the default does not impose a minimum segment length.

#pagebreak()
= Barcode association and interpretation

== A fast, label-blind regional statistic

The association diagnostic uses regional cell counts in 10 Mb bins. Before using barcode labels, estimate each region's expression rate per library count and its cohort allele fraction. Alleles are oriented with the supplied population-phase MAP convention. For barcode-region totals $D_(b,k)$ and exposure $L_b$, let $M_(b,k)=L_b hat(r)_k$. Let $A_(b,k),N_(b,k)$ be aligned H1 and total allele counts. The score is
$ S = sum_(b,k) frac((D_(b,k)-M_(b,k))^2,M_(b,k))
  + sum_(b,k) frac((A_(b,k)-N_(b,k)hat(q)_k)^2,
    N_(b,k)hat(q)_k(1-hat(q)_k)). $
Zero-denominator terms contribute zero. These are standardized residual summaries; no theoretical chi-squared null is assumed.

Shuffle whole-cell barcode labels within the supplied exchangeability blocks. This preserves barcode sizes within each block while allowing their RNA exposures to change with the shuffled cells. The statistic's cohort centering and feature definitions remain fixed. With $Q=199$ shuffles, the joint depth-plus-allele score receives the Monte Carlo p-value
$ p_("perm") = frac(1+sum_(q=1)^Q bb(1)[S_q >= S_("obs")],Q+1), $
including numerical ties. This convention follows #doi("10.2202/1544-6115.1585")[Phipson and Smyth (2010)]. Depth and allele contributions are also reported, but only the joint score receives the formal p-value. No HMM is fitted during shuffling. If blocks permit no exchanges between barcode labels, the result is unassessable.

Blocks must encode appropriate sample/batch restrictions; the program does not infer them automatically. A small p-value indicates barcode-associated regional structure under these restrictions. Expression programmes can contribute, and coarse regions can conceal opposing focal changes or phase patterns. A nonsignificant result can reflect limited power or barcode timing even when CN variation is present. It is a lineage-information diagnostic, not a general data-quality score or evidence that clone probabilities are calibrated.

== Three distinct uncertainty questions

+ *Are the count models numerically fitted?* Inspect phase convergence, scalar optimization convergence and dispersion boundary flags. The finite HMM sums paths conditional on those fits; there is no retained MCMC chain to diagnose.
+ *Are memberships stable under resampling?* Inspect the expression/HF stability and recursive node audits. Stability is conditional on the observed cells, transform, phase alignment and candidate splits.
+ *Do inferred probabilities and groups have the claimed biological reliability?* This requires known-truth experiments and external evidence. Neither numerical convergence nor bootstrap stability proves calibration.

Reference mismatch, coordinated expression variation, barcode collisions, phase errors and post-barcoding evolution can violate the model. More counts reduce sampling noise but may amplify systematic errors. The depth outlier mixture addresses isolated depth discrepancies; it is not a model of all these failure modes. Whole-cell bootstraps also have limited ability to characterize variation in tiny pseudobulks.

For genotype-phenotype analyses, examine purity together with completeness in simulations. Purity evaluates within-group genetic mixing, while completeness evaluates recovery of true same-clone pairs across barcodes. Cell-pair weighting reflects the number of cells exposed to a merge, but real phenotypic confounding need not scale linearly with contamination. Neither score alone establishes suitability for a particular downstream association.

#pagebreak()
= Current calling results at three depths

== Evaluation design and provenance

The saved calling benchmark includes five gene-based histories at 500, 5,000 and 20,000 nominal UMIs/cell, plus focal BAM-derived assays with 1, 2 and 4 Mb events at three depths. The histories have five broad clones and 50 barcodes, ranging from 10 to 1,000 cells. The focal controls have 40-cell barcode pseudobulks. Their high-depth assay is approximately 18,500 UMIs/cell.

The tables and figure below use *PLN depth, joint CN/phase inference, a 30 Mb correlation length and the 5% outlier mixture*. Dispersion is refitted at each aggregation level in the benchmark. These are saved *fixed-group calling evaluations*: histories reuse inferred group pseudobulks, while focal cases use individual observed barcodes. They do not constitute a rerun of end-to-end grouping after promoting the mixture throughout the workflow. Simulation truth is used only after inference to evaluate calls.

False alterations are the cell-weighted fraction of truth-diploid genes called altered among assigned barcodes. Event recovery is the correctly called gene fraction in an event, averaged over affected assigned barcodes with cell weights. An event is detected when recovery is at least 50%. There are 74 observable history events per depth and 18 focal events; a history event with no retained genes is excluded.

#figure(image("figures/calling-evaluation.png", width: 100%),
  caption: [Evaluation of the saved 5% mixture calls. All plotted quantities require ground truth. Histories average errors equally over five histories. “Generating reference” is an exact-reference diagnostic; “held-out fitted reference” uses a profile fitted without the generating donor. Focal high depth is approximately 18,500 UMIs/cell. Memberships are fixed in this comparison.])

#table(columns: (1.65fr, 1fr, 1fr, 1fr), inset: 5pt, stroke: .35pt + rgb("d5dfe3"),
  [*Evaluation / 5% mixture*], [*Low*], [*Medium*], [*High*],
  [History false alterations], [0.064%], [0.021%], [0.025%],
  [History events detected], [57/74], [72/74], [73/74],
  [Focal, held-out: false alterations], [0.836%], [1.690%], [1.318%],
  [Focal, held-out: events detected], [5/18], [15/18], [17/18],
  [Focal, generating: false alterations], [1.074%], [0.518%], [0.169%],
  [Focal, generating: events detected], [11/18], [16/18], [17/18])

With the held-out fitted reference, disabling the mixture gives 2/13/16 detected focal events and 1.356/3.259/3.877% false alterations across increasing depth. At 5%, all previously detected events in this panel are retained. This does not imply uniform improvement: in the low-depth generating-reference control, the confidently wrong fraction increases from 0.487% to 0.542%, and mean event recovery falls from 56.14% to 54.85%. Some low-depth history noise fits reach the lower search bound. The default remains an empirically chosen setting, not a calibration guarantee.

#pagebreak()
= Real-data illustration and implementation

== RBL1: full inference with the current defaults

The 28 September 2026 rerun starts from the validated RBL1 count bundle: 695 cells in 17 barcodes, with 10,201 genes and 92,483 SNP loci. It repeats the association diagnostic, grouping, phase/noise fitting, barcode calls, CN/HF refinement and group pseudobulk calls. The default 5% outlier component is active throughout, including phase/noise fitting. The external reference is the one already present in the input bundle; this rerun does not refit the newly ported reference-selection procedure or repeat BAM preprocessing.

#table(columns: (1.6fr, 1fr, 1fr), inset: 5pt, stroke: .35pt + rgb("d5dfe3"),
  [*Reporting status*], [*Barcodes*], [*Cells*],
  [Group 1], [5], [234],
  [Group 2], [10], [442],
  [Unresolved], [2], [19])

Memberships and unresolved status are unchanged from the previous full joint-PLN run. The barcode-association permutation p-value is 0.005 with 199 shuffles, the smallest possible value at this permutation budget. It supports barcode-associated regional count structure; it does not validate the individual groups. RBL1 has no complete CN ground truth, so these figures show evidence and inference, not a measured purity or CN accuracy.

The run completed in 401.6 seconds (about 6 minutes 42 seconds) on the development machine, using 64 cell bootstraps, 256 CN path draws and seed 270929. The phase-iteration cap was explicitly raised to 180 from the ordinary default of 60. The final pooled phase fit converged after 134 iterations (the initial fit took 44); the fitted dispersions were not at a search boundary. This is numerical fit convergence, not a calibration assessment. Model defaults were retained; the larger phase budget was necessary for this completed example.

```sh
uv run barcodecnv infer rbl1_cells.h5 --out rbl1_current \
  --seed 270929 --phase-iterations 180
```

#pagebreak()
== RBL1 barcode evidence and calls

#figure(image("figures/rbl1-current-summary.png", width: 100%),
  caption: [Full RBL1 inference with the 5% depth-outlier default. From top: cohort-relative smoothed expression, external-reference smoothed expression, centred rolling HF, and barcode CN calls. All modalities share barcode order and group boundaries; unresolved barcodes remain visible. Expression and HF are transformed observations; CN colours and the membership-stability strip are inference summaries. No ground-truth labels determine the grouping or ordering. The full-resolution PNG is bundled with this document.])

The expression panels use the same InferCNV-style transform but different baselines. They show relative, recentered signals, whereas CN calls use the external reference and unsmoothed count likelihoods. Consequently the sign of a smoothed expression block is not itself an absolute gain/loss call. The right-hand stability strip summarizes resampling stability, not a posterior probability of biological clone identity.

#pagebreak()
== RBL1 group summaries

#figure(image("figures/rbl1-current-group-cn.png", width: 100%),
  caption: [Current RBL1 group summaries: cell-weighted barcode consensus, direct group-pseudobulk CN calls, cell fraction with uncertain barcode calls, and confident barcode disagreement with the pooled call. The first two panels use class hue and marginal-probability saturation; the latter two report descriptive member fractions. Group calls condition on the selected memberships and refit noise at the pooled depth. These are production outputs, not ground-truth evaluation.])

Pooling can increase CN support in a selected group, but agreement is not enforced: the individual barcode HMMs and the group-pseudobulk HMMs use different count aggregations. The uncertainty and disagreement panels retain evidence that a pooled profile may conceal. Neither a smooth pooled profile nor low member disagreement establishes that all members are genetically identical.

#pagebreak()
== Command-line workflow

```sh
# From the Python package directory: counts to results in one command.
uv sync --locked
uv run barcodecnv infer --matrix filtered_feature_bc_matrix/ \
  --cells cells.tsv --genes genes.gtf.gz --reference reference.tsv \
  --alleles phased_counts.tsv.gz --genetic-map genetic_maps/ \
  --out results

# Reuse the automatically saved input snapshot:
uv run barcodecnv signal results/prepared.h5 --out signal
uv run barcodecnv infer results/prepared.h5 --out matched_reference \
  --depth-outlier-probability 0.01
```

For Cell Ranger outputs, `barcodecnv run --outs OUTS --cells CELLS --reference REFERENCE --one-block --out RESULTS` runs preprocessing and inference, with console stage messages. It saves intermediate files and a reusable bundle under `RESULTS/preprocessing/`, final tables and plots under `RESULTS/inference/`, and overall status in `RESULTS/pipeline.json`. `barcodecnv preprocess` runs only the cellSNP-lite/Beagle preparation stage. Tools and human hg38 resources are configured separately; normal expression reference fitting is available with `--reference-panel`, while FASTQ alignment remains upstream. Omit `--one-block` when the cell table supplies exchangeability blocks.

The examples above start from existing count matrices and annotation/phased-allele tables. `infer` saves `prepared.h5` inside its output directory, records the input hash and source paths, then performs the diagnostic, inference and reporting. The snapshot survives a subsequent fitting failure. The separate `prepare` command is optional; `signal` can also load count files directly without a reference.

`infer` and end-to-end `run` include the diagnostic unless `--skip-signal` is supplied. `--bootstraps`, `--draws` and `--phase-iterations` control computation; `--no-cn-refinement` and `--no-hf-refinement` omit their respective stages. Outputs must be new paths. For `infer`, use either a prepared bundle or count-input options. Failed inference runs retain their status and reason in `run.json`; end-to-end runs also record the failed stage in `pipeline.json`.

== Normal expression reference fitting

Instead of `--reference`, supply `--reference-panel` to fit a normal expression mixture from an existing Julia-format panel. This works in `run`, `preprocess`, and count-input `infer`/`prepare`; `fit-reference` provides a standalone reusable fit. Checksums, profile ordering, genome compatibility and normalized gene IDs are checked. Only cells selected in the cell map contribute to the pooled observed expression.

The default global fit adapts the Julia implementation of Numbat's `fit_ref_sse`: reference-mean expression must exceed 2 CPM and pooled observed counts must be positive. Observations and the predicted mixture are both normalized over fitting genes before calculating squared log-ratio error; the gradient includes the mixture normalization. This corrects the original port's mismatched normalization. Individual panel columns and the output reference retain their full measured-gene-universe normalization. Free softmax logits use the same Adam settings as Julia (learning rate 0.05, betas 0.9/0.999, epsilon $10^(-8)$). The default 2,000 updates are a budget, not a convergence guarantee; final objective and gradient diagnostics are recorded.

The optional `regional_consensus` method uses chromosome-local 200-gene windows, nuisance scales and BFGS fits, combined by a smoothed geometric median separately for two offset layouts. The two consensus weight vectors are averaged. Uninformative regions are skipped; unidentified layouts and unconverged local fits fail. SciPy BFGS replaces Optim.jl, so line-search differences can produce small numerical differences.

The fitted profile covers the full panel universe. Profile weights with metadata, exact fitting/overlap/absent gene lists, settings and provenance are saved under `reference_fit/`. These fitted reference quantities are subsequently treated as fixed; reference uncertainty is not integrated into CN probabilities. Sample CN changes can influence reference selection, particularly in the global fit.

#pagebreak()
= Interfaces, outputs and reproducibility

== Python interface

```python
from barcodecnv import DepthOptions
from barcodecnv.bundle import read_bundle
from barcodecnv.workflow import run_pbpc

bundle = read_bundle("sample.h5")
result = run_pbpc(
    bundle, replicates=64, draws=256, seed=42,
    depth_options=DepthOptions(outlier_probability=0.05),
)
barcode_probabilities = result["fit"].classes
labels = result["groups"]           # 0 means unresolved
pooled_probabilities = result["group_calls"].pooled
```

`run_pbpc` owns inference and grouping. The separate `barcode_signal_test` function owns the association diagnostic; the CLI composes both with file writing and plotting. Barcode probability axes are barcode, fine-grid marker and broad class; group probability axes are positive group, marker and class. Explicit labels accompany every axis. `fit_phase_and_dispersion`, `infer_barcodes` and `summarize_groups` expose the lower-level fitting boundaries. Barcode inference inherits the pooled fit's depth settings; standalone group reporting accepts `depth_options` explicitly.

== Files and scientific interpretation

- `prepared.h5`: reusable input snapshot for runs started from count files.
- `groups.csv` and `order.csv`: barcode IDs, cell counts, reporting and phase-pooling labels, stability and display order. Node CSVs retain expression, core, CN and HF decisions.
- `barcode_cn_genes.csv.gz`: barcode-by-gene broad-state probabilities and marginal MAP classes, including unresolved barcodes.
- `group_cn_genes.csv.gz`: consensus, pseudobulk probabilities and member uncertainty/conflict fractions. `group_cn_segments.csv.gz` contains fine-grid MAP runs.
- `result.h5`: complete labelled probability tensors, phase summaries, expression/HF features, broad CN distance draws and group summaries.
- `summary.png`, `group_cn.png`: production evidence, calls and disagreement displays. `signal.png`, `signal.json` and `signal_null.csv` preserve the permutation diagnostic.
- `run.json`: input identity, software version, arguments, fitted noise, convergence, runtime and completion status. Depth-mixture probability and scale also appear in HDF5 attributes.

#pagebreak()
== Code responsibilities and verification

Loading, expression grouping, haplotype features, count likelihoods, numerical HMMs, fitting and reporting have separate modules in `src/barcodecnv/`. SciPy supplies adaptive PLN quadrature; Numba compiles its callback and the HMM recursions. `README.md` describes the module boundaries.

At this revision, 113 Python tests pass. They cover R smoothing fixtures, independent PLN integrals, enumerated joint HMM paths with and without the mixture, chromosome resets, coherent path sampling, depth-setting propagation through noise fits, group aggregation, reference-fitting parity with Julia, preprocessing, single-command versus staged equivalence, and CLI failure handling. These checks verify computation, not biological calibration.

`docs/figures/render_calling_evaluation.py` regenerates the saved calling-evaluation figure and the earlier fixed-group RBL1 illustration from bundled data. The current RBL1 production PNGs are copied from the completed full rerun; its manifest and membership comparison are bundled alongside them. `docs/figures/rbl1-current-provenance.json` records source paths, hashes and scope, while `docs/figures/provenance.json` describes the earlier calling comparison. Compilation needs no external benchmark folders. Recomputing the current RBL1 figures requires its original input bundle and inference workflow, not the fixed-group rendering script. Historical experiment scripts are not included in this standalone release.

== References

#text(9pt)[
1. Gao, T., Soldatov, R., Sarkar, H., et al. (2023; online 2022). Haplotype-aware analysis of somatic copy number variations from single-cell transcriptomes. _Nature Biotechnology_ 41, 417-426. #doi("10.1038/s41587-022-01468-y")[doi:10.1038/s41587-022-01468-y].

2. Broad Institute. _InferCNV_, R preprocessing implementation. Numerical reference version 1.28.0; fixture-generation settings and package versions are recorded in `tests/fixtures/infercnv/`. #link("https://github.com/broadinstitute/infercnv/blob/master/R/inferCNV_ops.R")[Source implementation].

3. Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. _Statistical Applications in Genetics and Molecular Biology_ 9(1), Article 39. #doi("10.2202/1544-6115.1585")[doi:10.2202/1544-6115.1585].
]
