# Visually Indicated Sounds — 2-Day Project Plan

**Goal:** End-to-end pipeline that takes silent video clips of drumstick impacts and synthesizes plausible impact audio. Trained and evaluated on the *Greatest Hits* dataset (Owens et al., 2016).

**Guiding principle:** Get an ugly working pipeline end-to-end on Day 1, then improve it on Day 2. A working bad model beats a half-built good one. If you fall behind, cut features in the order listed at the bottom of this doc.

---

## Setup checklist (do this before Day 1)

- [ ] Python env with: `torch`, `torchvision`, `numpy`, `scipy`, `librosa`, `pycochleagram` (from McDermott lab GitHub), `matplotlib`, `scikit-learn`, `opencv-python`, `tqdm`
- [ ] Download the Greatest Hits dataset from `vis.csail.mit.edu` — both videos and audio tracks
- [ ] GPU available (Colab T4 is fine if no local GPU)
- [ ] Project repo with subdirs: `data/`, `src/`, `checkpoints/`, `outputs/`, `figures/`
- [ ] Sanity-check one video plays and its audio loads at the expected sample rate (paper uses 96 kHz capture; resample to 22050 Hz to keep things fast)

---

## Day 1 — End-to-end pipeline

### Morning Block (~4 hrs): Data loading + audio representation

**Why this block first:** If your audio representation can't round-trip (cochleagram → waveform → recognizable sound), your model literally cannot succeed. Lock this down before training anything.

#### 1.1 Onset detection (~45 min)

Detect impact times in the audio track. Owens et al. use:
1. Threshold the amplitude gradient to find peak candidates
2. Mean-shift merge nearby peaks
3. Non-max suppression so onsets are ≥ 0.25 s apart

**Simplification for v1:** Use `scipy.signal.find_peaks` on the audio envelope with `distance=0.25*sr` and a height threshold tuned by eyeballing 2–3 videos. Skip mean-shift unless results look bad.

```
Inputs:  waveform (1D array), sample rate
Outputs: list of onset timestamps (seconds)
```

Verify by overlaying detected peaks on a waveform plot for 3 random videos. If onsets line up with audible hits, move on.

#### 1.2 Cochleagram extraction + inversion (~2 hrs)

This is the highest-risk step. Do not skip the inversion check.

**Forward (waveform → cochleagram):**
- 40 bandpass filters on ERB scale (+ low-pass + high-pass) — use `pycochleagram.cochleagram.human_cochleagram`
- Take Hilbert envelope of each subband (this is built into the library)
- Downsample envelopes to 90 Hz (≈ 3 samples per video frame at 30 fps)
- Compress with exponent 0.3 (`s ** 0.3`)

**Inversion (cochleagram → waveform):**
- Use `pycochleagram.cochleagram.invert_cochleagram` (Griffin-Lim style iterative reconstruction)
- Or implement the paper's parametric synthesis: impose subband envelopes on white noise, one iteration

**Critical sanity check:** Take a real audio clip from the dataset → forward → invert → save as `.wav` → listen. If the inverted version sounds like a recognizable degraded version of the original, you're good. If it sounds like static, debug before continuing.

#### 1.3 Dataset class (~1 hr)

Build a PyTorch `Dataset` that yields `(video_clip, cochleagram)` pairs centered on impact onsets.

- Window: 15 frames around the onset (≈ 0.5 s at 30 fps), with the onset at frame 7 or 8
- Cochleagram: matching 0.5 s window, shape ≈ `(42, 45)` after downsampling
- Train/test split: 75/25 at the **video level** (not the clip level — clips from the same video must not leak across splits)
- Cache cochleagrams to disk on first epoch; recomputing them every epoch is the #1 way to waste training time

### Afternoon Block (~4 hrs): Visual features + simplest model

#### 1.4 Visual feature extraction (~1.5 hrs)

For each frame in the 15-frame clip, compute two things:

1. **RGB feature:** Pretrained ResNet18 (or AlexNet if course used it) penultimate layer (`avgpool` output, 512-dim for ResNet18). Just the first frame is enough — paper does this too, "to reduce computational cost."
2. **Spacetime image feature:** For each frame `t`, build a 3-channel image where channels are grayscale frames `t-1, t, t+1`. Run through the same CNN. This encodes motion cheaply without optical flow.

Concatenate per-frame features into a tensor of shape `(15, 1024)`. Cache these — you'll reuse them.

#### 1.5 V1 model (~1 hr)

**Keep this simple. No LSTM in v1.**

```
Input:  visual features (15, 1024)
        → flatten to (15360,)
        → Linear(15360, 512) + ReLU
        → Linear(512, 256) + ReLU
        → Linear(256, 10)        # predicts PCA-reduced cochleagram
Output: 10-dim vector
```

PCA: fit on training-set cochleagrams (flattened to `42*45 = 1890`-dim), keep top 10 components. Predict in PCA space, invert PCA at eval time. Paper does this — it makes the regression target much easier.

Loss: MSE in PCA space. Use Adam, lr 1e-4, batch size 32.

#### 1.6 Train v1 (~1.5 hrs)

- Train for ~20 epochs or until validation loss plateaus
- Save checkpoint
- On a held-out clip: predict PCA vec → invert PCA → invert cochleagram → write `.wav`
- **Listen to 5 examples.** They will sound bad. That's expected. Confirm: (a) audio is non-silent, (b) loud sections roughly align with hits, (c) softer materials sound vaguely softer than harder materials.

### Evening Block (~2 hrs): Buffer + commit

- Fix whatever broke (something will have)
- Commit working code, push to git
- Write a 5-line "what works / what's broken" note for tomorrow

**End-of-Day-1 success criterion:** You can run one command and produce a `.wav` file from a held-out silent video clip. Quality doesn't matter yet.

---

## Day 2 — Improve, evaluate, write

### Morning Block (~4 hrs): One improvement, then freeze

Pick **exactly one**. Listed in priority order:

#### Option A — Example-based synthesis (RECOMMENDED, highest payoff)

Instead of inverting your predicted cochleagram directly, use it as a nearest-neighbor query against real training cochleagrams and play back the matched real waveform.

```
1. Build a library: for every training-set onset, store
   (cochleagram_vec, waveform_clip)
2. At inference: predict cochleagram → L1 nearest neighbor in library
3. Output the corresponding real waveform
```

This is ~50 lines on top of what you have. Paper shows it's the single biggest factor in human-fooling rate (40% with example-based vs. 30% with parametric).

#### Option B — Add temporal modeling

Replace the MLP with a small GRU (1 layer, hidden dim 128) over the 15-frame feature sequence. Take the final hidden state → Linear → 10-dim PCA prediction. Use GRU not LSTM — fewer params, trains faster, basically equivalent here.

Only do this if Option A's results still sound temporally jumbled.

#### Option C — Two-stream input

If you skipped spacetime images on Day 1, add them now. Probably already done — skip this.

**Stop improving by noon.** Whatever you have at noon is what you evaluate.

### Afternoon Block (~3 hrs): Evaluation

Three evaluations, in order:

#### 2.1 Quantitative auditory metrics (~45 min)

For each held-out clip, compute on real vs. predicted cochleagrams:

- **Loudness error:** `|max(L2_norm(pred_subbands)) - max(L2_norm(true_subbands))|`, averaged over clips
- **Spectral centroid error:** center of mass of frequency channels in a 1-frame window at the impact center; report MSE and Pearson correlation across clips

Compare against two baselines:
- Random impact sound from training set
- Mean cochleagram (predict the same thing every time)

If your model doesn't beat both baselines, something is wrong — debug.

#### 2.2 Material-informativeness probe (~45 min)

The cleverest experiment in the paper, and almost free:

1. Train a linear SVM on **real** cochleagram features to predict material class (use the dataset's material labels)
2. Apply that SVM to your **predicted** cochleagrams
3. Report class-averaged accuracy

If accuracy is meaningfully above chance (1/N_classes), your model has learned material-discriminative features even though it was only trained on raw audio regression. Paper gets ~22%; you're doing well if you get >15%.

#### 2.3 Mini listening study (~1 hr)

**Do not attempt MTurk in 2 days.** Instead:

- Recruit 5–10 classmates/friends (in person or over text)
- 2AFC: show two videos of the same scene, one with real audio, one with predicted; ask which is real
- 10–15 trials per participant
- Report mean fooling rate with confidence intervals
- **Be honest in the writeup** about the small N and the convenience-sample bias

#### 2.4 Make figures (~30 min)

- One cochleagram comparison plot: 4 examples, real (top) vs. predicted (bottom). Mimic Owens Figure 8
- One bar chart of loudness error and centroid error vs. baselines
- One small table of material-probe accuracy

### Evening Block (~3 hrs): Write up

You already have a solid draft introduction. Fill in:

- **Methods** — what you actually built. Be explicit about simplifications from the paper (no LSTM, smaller listening study, etc.). Justify each one (training time, scope).
- **Results** — the three evaluations from this afternoon. One paragraph per evaluation, plus the figures.
- **Discussion** — what failed, what surprised you, what the material-probe accuracy suggests about whether the model "understood" materials. Move the "real-world video generalization" goal from your draft intro into Future Work — it didn't fit in 2 days, and saying so is fine.
- **Update the AI statement** to reflect any AI use during the project itself, not just bibliography.

---

## What to cut if you fall behind

In priority order — drop from the top of this list first:

1. **GRU/temporal model** (Option B). The MLP is enough.
2. **Spacetime images.** RGB-only works; just note it as a limitation.
3. **Material SVM probe.** Nice to have, not essential.
4. **Example-based synthesis.** Direct cochleagram inversion is acceptable.
5. **Mini listening study.** Auditory metrics alone can carry the eval section, but mention the listening study as planned future work.

**Non-negotiable:** working cochleagram pipeline + one trained model + at least one quantitative evaluation. If you have those, you have a paper.

---

## Reference values (from Owens et al. 2016) for sanity checks

| Quantity | Paper value | Acceptable for you |
|---|---|---|
| Cochleagram filters | 40 (+ LP/HP) | 40 |
| Compression exponent | 0.3 | 0.3 |
| Envelope sample rate | 90 Hz | 90 Hz |
| Clip length (centered) | 15 frames / 0.5 s | 15 frames |
| PCA dim for prediction target | 10 | 10 |
| Material classifier on real sounds | 45.8% (chance 5.9%) | >25% |
| Material classifier on predicted sounds | 22.7% | >15% |
| Human fooling rate (full system) | 40.0% | >30% |
| Human fooling rate (random sound) | 19.8% | this is your floor |

---

## Common failure modes to watch for

- **Silent predictions.** Loss collapses to predicting near-zero. Fix: check that your PCA target isn't normalized to ~0 mean; use a robust loss (paper uses `log(eps + r^2)`).
- **All predictions sound the same.** Model collapsed to predicting the mean. Fix: smaller learning rate, more data augmentation (random crops, horizontal flips), check that visual features actually vary across inputs.
- **Onsets misaligned.** Your detected onsets don't line up with hits. Fix: visualize peaks on waveform; tune `find_peaks` height/distance.
- **Cochleagram inversion sounds like noise.** Bug in the forward pass — most often wrong filterbank or skipped envelope step. Round-trip a real clip and bisect.
- **Train/test leakage.** Splitting at the clip level instead of the video level. Symptoms: suspiciously low test loss. Fix: split by video filename.

---

## Ethical notes (for the writeup)

You raised these in your prep assignment — keep them in the final paper, even briefly:
- Risk of audio deepfakes; mitigation via watermarking generated audio
- Risk of recovering redacted audio; out of scope here since model only handles impact sounds, but worth flagging
- Training-set provenance: Greatest Hits is collected by the paper authors with consent; you're not introducing any new identifiable subjects