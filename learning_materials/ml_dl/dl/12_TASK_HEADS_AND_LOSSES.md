# DL12. Prediction heads, targets and losses

A prediction head maps learned features to the outputs a task needs. One class for a whole example, one label for each token, a numerical value and a segmentation map need different shapes and objectives. The head, target alignment, loss reduction and evaluation metric must agree with one another.

The hidden representation has shape [B,T,D] = [1,3,4]: one example, three positions and four features per position. Choose a target type to inspect its output shape and a separate authored score/loss calculation; no trained head is loaded.

Earlier architectures produce features. This lesson connects those features to an actual prediction task without treating softmax as the right operation for every output. The shared metrics lesson explains general evaluation; a deeper note adds segmentation-specific IoU and Dice.

[course map](../COURSE_MAP.md). Required background: DL05, DL08, ML10.

## 1. Target shapes and valid scoring units

**Key terms.** A **task head** maps features to outputs; **multiclass** selects one class; **multilabel** permits several; **segmentation** predicts per-pixel labels; **detection** predicts object classes and locations. A **mask** can mean a target image or a scoring/visibility rule, so name which.

**Process.** Define the target shape and meaning before constructing the output layer. Produce matching logits/coordinates, apply the correct loss and reduction, then evaluate with metrics that match the task and independent units.

A single image label commonly uses [B,C] logits and [B] integer targets. Semantic segmentation commonly uses [B,C,H,W] logits and [B,H,W] hard-label targets. Multilabel classification commonly uses independent binary logits with matching target indicators. Feeding softmax probabilities into a loss expecting raw logits changes the computation.

Detection adds box representation, matching/assignment, class scores and localization penalties; it is not classification with more categories. Class imbalance can affect sampling, loss weighting and thresholds. A rare-class recall gain can cost false positives, so report both. Aggregate at patient/source/image level when pixels or patches are correlated.

## 2. A worked calculation

**Symbols and assumptions.** IoU is intersection size divided by union size for a declared pair of regions. C is class count, B batch, and H/W spatial dimensions. Per-class and averaged IoU require explicit absent-class handling.

### Region overlap: IoU and Dice

For segmentation, foreground is the set of pixels assigned to the object or region of interest. Intersection counts pixels both prediction and reference include; union counts pixels included by either. Intersection over union (IoU) and the Dice coefficient summarize the same overlap with different denominators, so their values should not be compared as though they were one metric.

If predicted foreground has 7 pixels, reference foreground has 5 and their overlap is 3, union=7+5−3=9 and IoU=3/9=0.3333. Dice overlap is 2×3/(7+5)=0.5: same regions, different metric. A ten-class [8,10] output contains 80 logits, not 80 independent target examples.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Choosing task-specific heads and objectives

Choose the simplest head matching the annotation. Weighted cross-entropy, focal-style objectives and Dice losses have different targets/normalizations and need a specific derivation before implementation. They should not appear as unexplained dropdown names.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: image-level independence

Use authored image masks for arithmetic and shape tests. The first lab can remain image classification; segmentation/detection are bounded theory branches until real datasets, target loaders and metrics are implemented.

**Check.** Why is treating every segmentation pixel as an independent case for uncertainty suspect?

**Answer.** Nearby pixels and pixels in one source image are correlated. Resample independent images or higher-level source groups under the stated protocol.

## 5. Object detection: predict locations and labels together

Object detection returns a variable-size set of object boxes, class labels and scores for an image. A box can use corner coordinates or center/width/height; declare the convention before calculating overlap or losses. Training must match predictions to annotations and combine classification with localization terms. Detection is not image classification with extra class names.

Many detectors propose dense or region-based candidates and use score filtering plus **non-maximum suppression (NMS)**. NMS keeps a high-scoring box and suppresses lower-scoring overlapping boxes under a stated class/overlap rule. It is postprocessing, not an evaluation metric, and not every architecture requires it: DETR uses a set-prediction/matching formulation rather than ordinary NMS.

For continuous-coordinate boxes A=(0,0,2,2) and B=(0.5,0,2.5,2), both areas are 4. Their intersection is 3 and union is 5, giving IoU=0.6. If same-class scores are 0.9 and 0.8 and NMS threshold is 0.5, B is suppressed. Do not add an inclusive-pixel “+1” to this continuous convention. A ground-truth matching threshold is a separate evaluation choice, even if it also uses IoU.

At evaluation, match predictions to ground truth under the benchmark's one-to-one rules, score ordering and overlap thresholds. Unmatched predictions can become false positives, including duplicate detections of one object. Average precision summarizes a precision/recall relationship under specified interpolation, classes and IoU settings. Its value is not interchangeable with a classification accuracy or a single F1.

**Practice.** Could aggressive NMS improve precision but reduce recall? **Answer.** Yes, especially for nearby same-class objects. Inspect the retained boxes and full precision/recall trade-off. Train/validation image or source splits must preserve independence; correlated video frames should not be randomly divided without justification.

[Official NMS definition](https://docs.pytorch.org/vision/main/generated/torchvision.ops.nms.html) · [DETR](https://arxiv.org/abs/2005.12872).

## 6. Semantic and instance segmentation: retain spatial detail

Semantic segmentation assigns a category to each pixel. Instance segmentation additionally distinguishes separate objects of the same category. A third variant, panoptic segmentation, combines instance-aware objects with semantic background regions under a stated label policy. The required annotations and evaluation are different.

A U-Net-style encoder compresses spatial resolution while building features; a decoder upsamples and combines compatible encoder features through skip connections. These spatial skip connections preserve detail and differ from the residual addition introduced in DL06. Concatenating encoder/decoder maps increases feature channels; adding requires matching shapes. Cropping, padding and resize alignment affect which output pixel corresponds to which input location.

For a two-class semantic task on a 64×64 image, one convention returns logits [B,2,64,64] and integer targets [B,64,64]. Cross-entropy consumes the raw logits, not another softmax first. A binary sigmoid formulation instead uses one logit per pixel with a compatible binary loss. State ignored labels and reduction. Weighted losses or soft overlap losses change the optimization target, not the underlying annotation quality.

Use the earlier IoU/Dice note for their denominators rather than defining them twice. To see why pixel accuracy can mislead, suppose only 10 of 1,000 pixels are foreground. Predicting background everywhere gives 99% accuracy but zero foreground recall. A per-class metric, foreground errors and absent-class convention are needed to interpret this result.

Mask R-CNN predicts instance-specific masks associated with detected objects; it is not the same output as one semantic map. Evaluate object matching and mask quality under a compatible protocol. Medical or geographic patches may share a patient/scene; reserve independent sources and resample at that level for uncertainty.

**Practice.** Does a perfectly shaped output tensor prove spatial labels are aligned? **Answer.** No. Inspect input/target pairs after all transforms and test known landmark positions through crop/resize/flip operations. Data augmentation must transform masks and boxes consistently with the image.

[U-Net](https://arxiv.org/abs/1505.04597) · [Mask R-CNN](https://arxiv.org/abs/1703.06870).

## 7. Tracking: keep an identity across video frames

Tracking connects observations across time. A multi-object tracker takes detections, predicts motion, associates new observations with existing tracks and handles track creation, missed detections and termination. It is not another name for detecting the same class independently in every frame. This extension concerns video association, not robotics/control.

SORT supplies a useful baseline built around motion estimation and assignment. Deep SORT adds an appearance representation to help associate detections. An association cost can combine geometry/motion and appearance under a declared gating policy; assigning each detection greedily to its nearest track can conflict with a globally consistent one-to-one assignment.

For an illustrative two-track/two-detection cost table, let costs be [[1,2],[1.1,100]], where lower is better. Greedily giving track 1 its cheapest detection can force track 2 to cost 100, totaling 101. The other one-to-one assignment costs 2+1.1=3.1. This arithmetic illustrates assignment constraints, not a complete tracker. No identity decision is justified from this cost table without knowing how costs were constructed.

Two people can be correctly detected in every frame while their IDs swap when they cross. Detection quality can remain high even though association is wrong. Track metrics therefore need localization/detection and identity continuity; HOTA is one framework for evaluating multiple aspects. Report dataset conventions, matching thresholds and treatment of missing/occluded objects, rather than claiming that image-level accuracy measures tracking.

**Practice.** Should a track disappear permanently after one missing detection? **Answer.** That is a policy choice. A bounded missing-frame allowance may preserve identity through occlusion but can also keep false tracks alive. Choose it on validation sequences and evaluate full held-out sequences, not shuffled individual frames.

[SORT](https://arxiv.org/abs/1602.00763) · [Deep SORT](https://arxiv.org/abs/1703.07402) · [HOTA](https://arxiv.org/abs/2009.07736). The current training lab remains image classification/reconstruction; these tasks have no hidden training implementation.

## Sources

- [PyTorch 2.9 CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html).
- [Stanford CS231n, Spring 2026](https://cs231n.stanford.edu/).
- [Dive into Deep Learning](https://d2l.ai/).
