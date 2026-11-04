# BehavioralTracker — Methods (for a methods section)

**Background and detection.** For every video BT builds an animal-free background from 80–120 frames spread over
the analysed time window (per-pixel median of the frames in which the pixel is not part of a moving animal; a
whole-video "rest percentile" background when the animal hardly moves in the window). A pixel belongs to the
animal when it differs from the background by at least the minimum contrast in the animal's colour direction
(darker / lighter); the threshold is raised next to strong background edges (0.35 × local gradient) so that
apparatus edges and vibrating walls are not detected. The background adapts slowly (time constant ≈ 40 s) to light
drift and quickly (≈ 3 s) when large parts of the arena change at once. Detection runs at ≤ 960 px.

**Tracking.** One animal per arena or chamber: the candidate spots are scored by size (log-normal around the
expected body area — the measured body length when given), contrast and distance from the predicted position;
a new lock needs ~1 s of consistent movement, fixed spots that never move are remembered as artefacts, and a
watchdog re-locks when the tracked spot stops while a moving animal-like spot exists. The body centre is the
tracking point; short gaps (≤ 0.5 s) are interpolated. Zone time uses the body centre; an entry needs ≥ 80 % of
the body (tail removed) inside the zone ("four paws"). Nose and tail base: the two ends of the body's long axis,
the head end decided over the whole track by dynamic programming from the tail, the body shape (narrower snout)
and the walking direction; novel-object exploration = the nose inside the object zone.

**Reflections, shadows and still spots (single animal).** Each candidate's opacity is measured as the lower
quartile of image / background grey in its body (a solid animal blocks the floor's light, ≈ 0.05–0.15; a reflection
or shadow in a clear wall lets the bars of the floor through, 2–3 × more). When the lock is on a see-through copy and
a clearly more solid spot of the animal's size (0.6–1.6 × the learned area, opacity ≤ 0.75 × and ≤ 0.45) stays within
3 body lengths for 0.6 s, the lock moves to the solid one. A jump of more than 1.5 body lengths onto a spot that
does not overlap the animal's previous blob is refused when that spot is a known artefact, more than 1.8 × the
animal's area, more than 2.2 × its mean contrast, or see-through while the animal is solid. The animal's size,
contrast and opacity are learned only while the lock moves (within the last 3 s) at a plausible size. A lock that
has been still for 6 s is released for a moving rival of the background-estimated animal size with at least half its
contrast. The automatic animal colour (darker / lighter) weighs each candidate colour's blobs by count × mass ×
spread and divides by (1 + 2 × the coefficient of variation of their size): the animal has a steady size, flicker
and noise do not.

**Several animals in one arena.** Foreground blobs are matched to identities with per-animal constant-velocity
Kalman predictions and the Hungarian algorithm; merged blobs are split from the predicted positions. Each
animal's look (mean grey of a disc on its body) is learned while the animals are apart; at the first clear frame
after a contact the identities are re-assigned by look when the animals differ by > 4 pooled SD.

**Freezing.** Classic motion: frames are converted to grey, Gaussian-blurred (σ = 1 px,
region scaled to ≤ 640 px) and compared with the frame 1/30 s earlier; a pixel moves when its grey value changes
by at least the cutoff. Auto cutoff = 2 × the 99.99th percentile of the frame-to-frame change of the quietest 75 %
of 40 × 40 px tiles of each frame (75th percentile over frames); when that percentile is > 1.6 × the 99.9th
(spiky compression noise) it is limited to 1.6 × the 99.9th percentile. Motion = changed pixels minus the typical
tile's changed fraction × the region size (camera noise), expressed as % of the body area (body length² / 2.4;
the measured length, the body outline, or the 90th percentile of the tracked size). Freezing = motion below the
threshold for at least the minimum freeze duration. Frames where the typical tile changed > 25 % (light switched)
keep the previous state; runs of frames at the start / end whose brightness is < 0.55 × or > 1.8 × the session's
median are not analysed. "Around the animal": only pixels within one body length of the tracked animal count.
"Body outline points": the animal's outline (adaptive threshold at 40 % of its contrast, calibrated on 16 frames)
with head, four leg areas, trunk and tail points; motion of the head, legs and trunk counts, the tail does not.

**One-pass analysis.** With a measured body length (circle radius known in advance), the motion statistics are
computed from the frames the tracking reads; frames whose final position differs from the one known at that frame
(found later, before the first detection) are measured again from the video. Results are identical to two passes
(checked frame by frame).

**Behaviour classifier.** Per frame: speed of the body centre, motion (% of the body area) of the whole body, head,
leg areas and tail, around the animal and in the whole region, the outline's extent, nose–tail-base and nose–centre
distances (in body lengths) and turning speed; each also as mean and SD over 0.33, 1 and 3 s and min / max over 1 s
(100 features). One random forest per behaviour (40 trees, depth ≤ 10, ≥ 15 frames per leaf, √F features per split,
features binned to 32 quantiles, balanced bootstrap). Probabilities are smoothed over 0.2 s, thresholded at 0.5,
gaps < 0.2 s closed and bouts shorter than the chosen minimum dropped. Tested by leave-one-video-out (or 5 time
blocks with a 3-s guard band) with frame-level precision, recall, F1 and Cohen's kappa.

**Statistics.** Experimental unit = the animal (repeated videos of one animal are averaged). Two groups: Welch's
t-test when every group passes Shapiro–Wilk (α = 0.05), else Mann–Whitney U; ≥ 3 groups: one-way ANOVA or
Kruskal–Wallis; effect sizes Cohen's d / η². Benjamini–Hochberg FDR across metrics. Repeated measures: paired
t-test / Wilcoxon (2 time points), Friedman (more). Linear mixed models (statsmodels MixedLM, ML): metric ~ group +
time + group:time + (1 | animal) [+ dam / cage variance components]; when the animal variance is 0 (singular), OLS
with animal-clustered standard errors.

**Validation against hand scoring.** Frame-by-frame agreement and Cohen's kappa per video; across videos Pearson r,
ICC(A,1) (Shrout & Fleiss ICC(2,1)) and Bland–Altman bias with 95 % limits of agreement.

**Reproducibility.** Every result file records the BT version, all settings and a hash-chained audit trail. The
Classic freezing measure is published as a short open program (reference_freezing.py, numpy + OpenCV) that gives the
same frame-by-frame result as BT; the source edition of BT can be made available to reviewers on request.
