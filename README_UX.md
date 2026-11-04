# BehavioralTracker 1.0.0

Animal behaviour tracking and analysis: open field, EPM, Y / T maze, light/dark box, NOR, social interaction,
freezing (fear conditioning), live tracking from a camera, event scoring and statistics.
How to use it: ❔ Help in the app (also `docs/USER_MANUAL.md`); contact: ✉ Contact and support in the app.

* `pro_ui/behavioral_tracker.ui` – the Qt-Designer design (**unchanged**).
* `pro_ui/ux_controller.py` – the **UX file**: loads the design and gives every screen its behaviour on top of
  the engine in `tracking/`, `analysis/` and `output/`.

## Install

**Windows (recommended):** double-click `BehavioralTracker_Setup_1.0.0.exe` → Next → choose the shortcuts
(Desktop / Start menu) → Install → Finish. No administrator rights needed; uninstall from *Settings ▸ Apps*.
* `BehavioralTracker_Setup_1.0.0.exe` (~10 MB) brings its own Python and **downloads the packages
  (~220 MB) while installing** – keep the PC online for a few minutes.
* `BehavioralTracker_Setup_1.0.0_offline.exe` (~90 MB) has everything inside – for PCs without internet.
* Windows may show "Windows protected your PC" (the Setup is not code-signed): *More info ▸ Run anyway*.
* If the program ever does not start, the reason is written to `%LOCALAPPDATA%\BehavioralTracker\crash.log`.

**From source:**
```
pip install -r requirements.txt
python main_pro.py
```
(On Windows keep the folder in a short path such as `C:\BT\BehavioralTracker_Pro`.)

**Build the Setup files yourself** (Ubuntu / WSL): `sudo apt-get install nsis mingw-w64`, then
`bash installer/build_installer.sh` → `installer/Output/` (online + offline Setup).

---

## Tamil quick guide (சுருக்கம்)

* **Experiment** (single / individual tracking): test type select pannunga, videos add pannunga. Indha videos
  **Batch Processing-la poagaadhu** — rendum thani. Video vendamna list-la select panni *✕ Remove selected*
  (illa Delete key). Tracking page mela irukkura box-la videos-a maathalaam.
* **Zone varaiya**: Video Setup ▸ *＋ Draw* ▸ Polygon / Freehand / Rectangle / Ellipse.
* **Preset zone edit / extend**: zone-a click → yellow points-a drag; edge double-click = pudhu point;
  point right-click = delete; *⛶ All* → corner handles resize, round handle rotate. Ctrl+Z undo.
* **⚡ Auto**: video-la apparatus-a kandupidichu zones fit pannum. Test type thappa irundha (udharanam
  plus-maze video, aana *Open Field* select pannirundha) app warning kudukkum.
* **🎯 Guided** (auto sariya varalana idha use pannunga): EPM-la 4 arm-oda kadaisi end-a click pannunga
  (first 2 = OPEN arms, next 2 = CLOSED arms), Y-maze-la 3 arm ends, T-maze-la stem → left → right,
  box-la floor-oda 4 corners. Arm width video-la irundhu automatic-a measure aagum; click konjam
  thappa irundhaalum arm-oda nadu line-ku snap aagum.
* **Auto-mask**: *Auto-mask outside zones* tick panna, zones-ku veliya irukkura kai / reflection / shadow
  track aagadhu. Mask zone line-oda exact-a (nool pudicha maadhiri) start aagum — gap illa (⚙ Settings ▸
  *Gap between zones and mask* la venumna konjam gap kudukkalaam).
* **📏 Measure the mouse** (Tracking page, *Minimum animal size* keezha): video-va pause pannunga (mouse
  neetama irukkura frame), button press panni mouse-oda **nose tip**, appuram **tail base** click pannunga.
  Mouse length (px) varum → *Minimum animal size* automatic-a set aagum (body area-la ¼). Right-click = cancel.
* **Minimum contrast** = mouse floor-ai vida ethana grey levels dark/light-a irukkanum-nu. Romba high-na
  dim place-la mouse miss aagum (lost frames, wobbly track); romba low-na shadow / noise mouse kooda
  serum. Nalla value = mouse-oda weakest contrast-la paadhi. **🎯 Auto-set contrast** press pannunga —
  40 frames check panni recommend pannum (*Use* press panna set aagum).
* **Light / dark box – Dark box**: *Covered (lid)* = mouse ulla theriyadhu, track aagadhu, time "dark box"-la
  count aagum. *Open (dim light)* = ulla irukkura mouse-aiyum track pannum.
* **Low-light boost** (Off / Auto / Low / Medium / Strong ≈5 lux): dark-ana video / zone-a brighten pannum —
  tracking-kum screen-la paakkavum. *Auto* = oru zone dark-a irundha mattum. Dark area noise-um measure
  aagum, so false detection varaadhu.
* **BORIS-style plots** (Event Scoring ▸ *📊 Plots & analysis…*, manual scoring-ku mattum): events plot,
  time budget, time bins, transitions, Cohen's kappa (2 observers).
* **Tracking**: *🔍 Test detection (at the start time)* – analysis odaadhu; start time-la oru still picture
  varum, adhula detect aana mouse yellow-a + marker-oda theriyum (play / scrub panni matha idangalaiyum
  paakkalaam). *Animal colour* thappa irundhaa (black mouse-ku "Lighter than floor") app warning kudukkum,
  sariyaana colour-ku maatha kekkum — thappaana colour-la tracker mouse utkaarndha empty spot-a follow pannum.
  Zone lines ellaa idathulayum nool maadhiri thin (1 px). Body centre point mattum dhaan track aagum.
  Light maarinaalum / camera konjam nagarndhaalum background adhai follow pannum (⚙ Settings ▸ Light changes). *Preview images per video* –
  ethana preview images save aaganum-nu neengale sollunga (0 = none). Tracking engine: *Robust* (default)
  illa *Classic*.
* **Batch**: videos add pannunga → oru video select panni keezha settings panel-la: *✏ Set zones…*
  (ovvoru videovukkum thani zones – Save & next video), Start / End time, Tracking engine, Detection mode,
  Animal colour. *Apply to all videos* = ellaa videos-kkum copy. Ellaam set pannittu *Start Batch* – ovvoru
  video mudinjadhum adhoda results thani folder-la save aagum (*📂 Open* / *⤓ Export*).
  **Excel – organise panniyirukku** (ovvoru video `<video>_Analysis.xlsx`, and `BatchSummary_*.xlsx`), ore
  order: *Video & tracking info* → *Time & percentage* → *Entries & transitions* → *Distance, speed & latency*
  → *Indices* (ovvoru index pakkathula formula) → *Settings* → *Column guide* (ovvoru column enna + formula).
  Oru video Excel-la kooda Entry sequence, Time bins, Zone coordinates, Raw tracking. **Entries** = ellaa
  test-kum main; **Transitions** = light/dark box-ku. Case maatum vithiyaasamaana zone names (Whole_Arena /
  Whole_arena) oru column-a serum.
  Batch odumbodhu endha video-va click pannalum (illa *👁 Live view*) adhoda tracking live-a paakkalaam —
  animal marker, zones, video time; mudinja video-kku last tracked frame.
* Preset zones: EPM / T-maze centre **square**, Y-maze centre **triangle** (arms adhoda sides-la start aagum);
  T / Y presets 'T' / 'Y' shape-laye (upright) varum.
* Trajectory / heat-map images original video maadhiriye irukkum (auto-mask dim illa).
* **↻ (round arrow)**: Tracking page-la gear pakkathula, Analysis page-la Export pakkathula. Analysis
  mudinjadhum idha click pannina tracking view-um results-um clear aagum, pudhu run-ku ready. Save aana
  files delete aagaadhu.
* **Sidebar order**: Dashboard → Experiment → Video Setup → Tracking → **Batch Processing** → Freezing – Individual
  → Freezing – Batch → Analysis → Results → Event Scoring.
* **Experimental Design**: column name-a maatha column header-a **double-click** pannunga (thani *Rename column*
  button illa).
* **❄ Freezing – Individual / Freezing – Batch** (sidebar-la rendu thani option, Analysis mela; Dashboard-la
  *Freezing* tile-um irukku): fear conditioning maadhiri
  test-la freezing automatic-a kandupidikkum (Classic motion method). Video open pannunga, venumna
  *⬠ Region* (polygon — corners click pannunga, double-click / Enter = mudinjadhu, right-click = undo, Esc = cancel)
  podunga, *▨ Mask* vachu reflection / light streak / cable / lamp area-va exclude pannunga (neraiya mask podalaam;
  *Clear ▾* = region / last mask / all masks / all). **Tracking** group-la *Animal colour*, *Min contrast*;
  *🎯 Test detection* = andha frame-la mouse kandupidikkudha-nu paarkkalaam (green marker), appuram
  *❄ Analyse this video*. Motion graph, freezing bouts,
  red-la moving pixels, FREEZING / ACTIVE ellaam varum. Threshold (% of body), minimum freeze (s), ignore blips,
  epochs (tones / CS, ITI — *＋ Tones…*), time bins maathina udane result maarum (video marupadi read
  pannaadhu). **Start / End (s)** kai-la type pannunga (*now* buttons illa). **Colour: Gray / RGB** — RGB = ovvoru
  colour channel-um paarkkum (grey-la theriyaadha colour change-um motion-a varum). Analyse pannumbodhu video
  **live-a** canvas-la theriyum (tracking maadhiri — FREEZING / STILL / ACTIVE badge); keezha *Measuring motion…
  NN %* periya-a theriyum. (Ellaa CPU cores-um use pannum — video pieces-a parallel-a odum, live picture first
  piece-a follow pannum.) *Count motion*: **Whole region** (default, side / top view rendukkum), **Around the animal**
  (BT tracking circle — top view-la cable sway-a ignore pannum) illa **Body dots** (top view): mouse mela small
  dots — nose, 4 legs, body centre, tail base + tail neelam-varaikkum tail dots — line-a connect aagi mouse
  maadhiri theriyum. Body move aanaa dhaan "moved"; **tail mattum aadunaa "Tail moved"** (body moved illa,
  freezing break aagaadhu). Mouse outline mouse-oda sondha contrast-la (40 %) kandupidikkum — bedding, shadow, wall
  serradhu illa; body size-um adhula irundhu (Body dots mode-la). Tail thin dark line-a follow pannum. Ovvoru
  dot-um andha part move aanaa dhaan nagarum (still-a irundha nagaraadhu). Dots canvas-la chinna-va, sharp-a
  varaiyum. Mouse time window muzhukka asaiyaama irundhaalum (freeze / thoongudhu) tracking kandupidikkum
  (background-a muzhu video-la irundhu marupadi edukkum). Leg dots body outline-la (paw veliya therinjaa adhai follow pannum; chinna video-la paw thani-a
  theriyaadhu — paw detection illa). Motion cutoff *Auto* = video-voda camera noise
  × 2 (classic calibration rule, empty video thevai illa; *Empty video…* vachu empty chamber video-la calibrate pannalaam).
  Body size tracking-la irundhu, illa *📏 Measure* (nose → tail base, 2 click). Light on / off frames-a
  *Light-change guard* skip pannum. Result-a *📊 Open in Analysis & Results* — anga Overview, Motion trace, Epochs,
  Bouts, Time bins, Frame review graphs ellaam varum (tracking result maadhiriye *Showing* list-la select
  pannalaam). Ovvoru analysis-um **`.btf`** (BT Freezing file) save aagum — video illaama kooda open pannalaam.
  **Freezing – Batch**: neraiya videos, Subject / Group type pannalaam, ovvoru video-kkum
  thani `results/<video>/freezing/` folder + `.btf` + `Freezing_BatchSummary_*.xlsx` (Statistics page-la open
  pannalaam). Batch odumbodhu video-va **double-click** panna (illa *👁 Live view*) freezing-a live-a paakkalaam;
  mudinja video double-click = adhoda result Analysis page-la.
* **▦ Chambers** (Video Setup, Calibrate keezha): orey video-la thani thani chamber-la mice (2–4 varaikkum) irundha.
  Tile click panna mattum chamber options varum (illana ellaam pazhaya maadhiri). *⚡ Find* chambers-a video-la
  irundhu kandupidikkum (illa *▭ Draw* — rectangle drag pannunga); ovvoru chamber-ukkum **Animal ID**.
  Mela chamber tabs: oru chamber select panni adhukku zones podunga (draw / ⚡ Auto / ▦ Preset) — oru chamber-la
  evlo zones venumnaalum podalaam (`C1 · Centre`, `C1 · Corner` …). *⧉ Copy zones to all chambers*.
  **Tracking** page-la *Settings for: All | C1 | C2 …* — oru chamber select panni adhukku mattum start / end time,
  detection mode, animal colour, low-light boost, minimum contrast maathalaam (bold = own value). *Start Tracking*
  = ellaa chamber-um **orey time-la** (video oru thadava dhaan read aagum). Ovvoru animal-ukkum thani folder +
  `.bta` (`<video>_Chamber1_M1/`), plus `<video>_chambers_summary.xlsx` (oru row = oru animal). Analysis page
  *Showing* list-la chamber-a select pannunga. **Batch**: *Scale* keezha *Chambers* (Video Setup / Off / Find n /
  **Own (Set zones…)**) + **Chambers…** = ovvoru chamber-kkum Animal ID + **thani start / end time** (mm:ss
  type pannalaam; kaali vitta video-voda time). *Own* select panna *✏ Set zones…* editor-la right side
  **Chambers** panel varum — *⚡ Find* / *▭ Draw*, chamber tabs, ovvoru chamber-kkum zones / preset / copy. Batch
  table-la ⏱ = chamber-kku own time irukku. BatchSummary-la oru row = oru animal (Video + Chamber + Subject).
* **Animals in one arena** (Tracking page): 1 = normal. Thani thani cage-la 2–4 mice irundha **▦ Chambers** use
  pannunga. Oru arena-la 2–3 mice *together* (social test) irundha 2 / 3 select pannunga — contact-ku apram
  identity look (coat / dye mark) vachu marupadi fix aagum; preview video check pannunga.
* **Event Scoring** (sidebar, eppavume kadaisi): keyboard vachu behaviours-a manual-a score pannradhu.
  *＋ Add* vachu **Behavioral Inventory** (behaviour catalogue / behavioural repertoire) create pannunga — ovvoru behaviour-ukkum oru **key** and **type**: *State* (key
  press = start, marupadi press = stop) illa *Point* (oru press = oru occurrence). Video play panni (Space),
  behaviour nadakkumbodhu adhoda key-a press pannunga. Events table-la time-oda varum; per-behaviour bouts,
  total / mean duration, latency summary; Excel / CSV export. Behavioral Inventory-a save / load pannalaam (adhuvaa
  remember pannidum).

---

## What is in 1.0.0

### Tracking engine (`tracking/robust_tracker.py`, wired in `tracking/location.py`)
| Problem of the old tracker | Fix |
|---|---|
| Animal resting > 50 % of the time ended up **inside the background** ("ghost"); the empty spot was tracked | Polarity-aware background (dark animal → bright percentile) + animal-removal refinement |
| After a lost frame the tracker **stayed at the stale position**; an animal re-appearing far away was never found | Search radius grows while lost; global re-acquisition after ~0.3 s; short gaps (≤ 0.5 s) interpolated |
| A hand / reflection outside the maze could grab the lock | Auto-mask outside zones + animal-sized first lock + robust polarity vote |
| HD video: mouse bigger than the fixed "max area" → **always lost** | Animal size measured automatically (0 = auto) |
| Every frame read with a seek (slow) | Sequential reading — 2–4× faster |
| Multi-mouse: threshold sign bug, no crop / masks / zones | Fixed; ghost-free background, auto animal size, per-mouse zone tables |

Also: camera-shake compensation, flicker / exposure compensation, oversized-blob splitting.
*Classic* engine = the simple centre-of-mass method (percentile threshold + optional search window).

**Benchmark** (synthetic videos with ground truth, % frames within ⅓ body length):

| Scene | old tracker | 1.0.0 |
|---|---|---|
| Light/dark box, long rest in dark side | 92.9 | **100** |
| EPM, long rest + experimenter hand | 0.0 | **100** |
| White mouse on black floor | 99.5 | **100** |
| Y-maze with 3 occlusions | 52.8 | **100** |
| HD 1280×720, large mouse | 37.4 | **100** |
| Low-contrast (IR-like) dark side | 49.9 | **99.4** |
| Camera shake + moving shadow | 99.9 | **100** |

On a real practice clip (tethered mouse on a plus maze) the old position was on the animal in 26 % of the
frames, the new one in 100 %.

### Video Setup (zone editor)
* Custom zones: polygon, freehand trace, rectangle, ellipse; any number, named with test-specific suggestions.
* Edit any zone (presets included): drag / add / delete points, move, resize, rotate one zone or the whole
  layout; duplicate, rename, delete; undo/redo; zoom & pan; lock.
* **⚡ Auto** – apparatus detected on an animal-free background: EPM / Y / T mazes (any rotation), open field
  (square or round), NOR (+ the 2 objects), light/dark box, 3-chamber. Warns when the video does not match
  the selected test type.
* **🎯 Guided fit** – a few clicks (arm ends or floor corners) → exact zones; arm width measured on the video,
  arm axes snapped to the real arms, perspective corrected for boxes, round arenas from 3+ edge points.
* Trace outline → zones, crop (zones carried over), save / load layouts (JSON).

### Tracking page
Engine (Robust / Classic), start / end, detection mode, animal colour, **low-light boost**
(Off / Auto / Low / Medium / Strong: a brightening curve applied before the blur, the same for the
background and every frame; with it the noise of every place is measured and the thresholds follow it),
**dark box covered / open** (light/dark box), minimum contrast with **🎯 Auto-set** (40 frames: the mouse's
contrast incl. its weakest moments and the noise → recommended value), smoothing, minimum size with
**📏 Measure the mouse** (2 clicks nose → tail base: body length → minimum size ¼ of the body area; the
length is always used by the tracker as the expected animal size), **preview images per
video**, ⚙ advanced settings, Test detection (a jump to another frame runs the ~1 s before it first, like
the real run), live view with the time / frame shown in black on a white box.
*Animals in one arena*: 1 normally; separate cages in one video = **Chambers** (one animal per chamber, up to 4,
Video Setup or the batch *Chambers* row); 2–3 animals together = the multi-animal engine (Kalman + Hungarian,
merged blobs split from the predictions, identities re-assigned by look after every contact —
tracking/two_mouse.py `Appearance`). The body centre is the tracking point; for the novel object test the nose
(head end of the body, tracking/body_points.py) counts exploration (⚙ Settings ▸ Object exploration). The background is **adaptive**
(⚙ Settings ▸ Light changes, on by default): it follows slow light drift, a place that has looked different for
~2 s without being the animal (light changed there, camera nudged, a shadow / dropping that stays) becomes floor,
and when large parts of the arena change at once the picture is renewed around the animal straight away. An
animal colour that does not match the video (or "Either" for a clearly dark / light mouse) gives a warning.
The first lock looks ~1 s ahead and follows every candidate (a still shadow / label cannot take it); a
lock that sits still for 3 s while an animal-like spot keeps moving is switched to it. With *Keep it where
it was last seen* (default) lost frames keep the last position, so the zone times add up to the whole
time window you set (a note says so when the video ends before the end time).

### Analysis
Excel workbooks are organised by kind of value: Video & tracking info, Time & percentage (every zone's time,
then every zone's %, then the test's time measures), Entries & transitions (zone entries, the test's entry
measures incl. alternation, zone → zone transitions), Distance, speed & latency, Indices (each with its formula
in the next column), Settings, Column guide (meaning + formula of every column); per video also Entry
sequence, Time bins, Zone coordinates, Raw tracking and All values (one row, read back by the program). The
BatchSummary has the same sheets with one row per video; Statistics merges the data sheets per video.
Per-zone time / % / entries / latency / mean visit / distance for every test, plus EPM open-arm %, Y-maze
spontaneous alternation (SAR / AAR), T-maze goal alternation, light-dark transitions and latencies,
open-field centre measures, NOR discrimination index (*Novel* / *Familiar* zones), sociability index.
Trajectory and heat-map are drawn on a picture that looks like the original video (no mask dimming).
Files: `<video>_Analysis.xlsx`, `zone_summary.csv`, `arm_entries.csv`, `raw_tracking.csv`, charts,
`background.png`, `preview/`.

### Batch
Settings panel for the selected video(s): **Set zones…** (zone editor with each video's own frame: auto,
guided, preset, draw, copy from another video; *Save & next* walks through the queue), **start / end time**,
**tracking engine / detection mode / animal colour / chambers**, *Apply to all videos* (one compact grid). Results folder
(next to each video or one chosen folder); every video is exported to its own sub-folder as soon as it is
finished (*📂 Open*, *⤓ Export*); `BatchSummary_*.xlsx` with Subject / Group for the Statistics tab.
**👁 Live view**: while a batch runs, click any video (or *Live view*) to watch its tracking live (marker, zones, video time); finished videos show their last tracked frame.
**Chambers** (below *Scale*): *Video Setup* (the chambers marked there, mapped to each video's size), *Off*,
*Find 1–4* (found in that video) or *Own (Set zones…)* (chambers drawn / found for this video in the zone
editor's **Chambers** panel: find, draw, chamber tabs, zones / preset / copy per chamber). **Chambers…** per video:
Animal ID and its **own start / end time** for every chamber (seconds or mm:ss; empty = the video's time; ⏱ in
the table). A chamber video gives one BatchSummary row per animal (Video, Chamber, Subject = Animal ID).

### Chambers — several arenas in one video (`tracking/chambers.py`, `pro_ui/chambers_ui.py`)
* A chamber is a rectangle of the (straightened) video; each chamber is tracked **exactly as if the video had been
  cropped to it** (own background model, tracker, zones, settings, results folder and `.bta`) — checked: the
  positions are identical to tracking the cropped video on its own.
* All chambers are tracked **at the same time from one reading of the video**; long videos in up to 4 pieces at
  once (each piece warms up first; frame-by-frame when a stop condition or a covered zone needs it).
* Zones: name prefix `C<n> · ` (several zones per chamber), or by where their centre lies; stripped in that
  chamber's results. Masks apply where they fall.
* Per-chamber settings (Tracking ▸ *Settings for*): start / end time, detection mode, animal colour, low-light
  boost, minimum contrast (or Classic threshold); 🎯 *Auto-set contrast* measures every chamber.
* *⚡ Find*: where the animals move (90th percentile of the frame-to-median difference over 48 frames), then the
  floor of the same colour around each moving area (chroma distance, Otsu), + a margin for the walls; an even
  split when the animals cannot be told apart.
* Output: `results/<video>/<video>_Chamber<n>_<ID>/` (every usual file + `.bta` with its own video copy — all
  copies written from one reading of the video) and `results/<video>/<video>_chambers_summary.xlsx / .csv`.
* *Test detection* and the live view show every chamber (colour frame, name, animal marker, OK / SEARCHING).

### Freezing Analysis (new — `tracking/freezing.py`, `pro_ui/freezing_page.py`, `pro_ui/freezing_results.py`, `output/freezing_report.py`, `output/freezing_graphs.py`)
Two side-bar pages, **Freezing – Individual** and **Freezing – Batch**, after Batch Processing and above Analysis &
Results (Event Scoring stays last); a *Freezing* tile on the Dashboard opens Freezing – Individual.
* Start / End are typed in (seconds). The progress text sits above a thin bar in large type.
* **Live**: Freezing – Individual shows the video on the canvas while it is measured (motion pixels, body dots,
  FREEZING / STILL / ACTIVE / LIGHT CHANGE badge); the video is read in pieces on all cores and the picture
  follows the first piece. Batch: double-click / *👁 Live view*.
* **Animal that never moves in the window**: when BT tracking finds the animal in < 60 % of the frames, it is
  tracked again with the background sampled from the whole video (120 samples; the first floor estimate is the
  97th percentile for a dark animal, so the floor under a resting animal is found from the few samples without
  it there, then masked out as usual). The new run is kept only when it finds the animal in ≥ 20 % more frames
  and at an animal's size (≥ half the size of the first run). Ordinary videos are tracked exactly as before.
* **Colour: Gray / RGB** — RGB counts a pixel as changed when any colour channel changes by ≥ the cutoff
  (the largest of the three channel differences), so a colour change that is flat in grey is still motion.
* **Body dots** (*Count motion* ▸ *Body dots — tail ignored*, needs BT tracking, top view): before the motion
  pass 16 frames are used to measure the animal's contrast, the floor noise and the body size (outline area
  without the tail → length = √(2.4 × area)); in Body-dots mode this is the body size for "% of body" (*Body size
  from: Body outline*). Every frame: the outline is where the picture is 40 % of the way from the floor to the
  animal (its own contrast at the tracked centre) — textured bedding, the shadow and the wall stay outside; an
  outline much bigger than the animal is cut again higher. Body = outline without thin parts (opening ≈ 0.2 body
  length). Tail = a thin dark line followed pixel by pixel from the body edge near one end (thin-line filter, floor
  level measured on the video); the end with the tail is the back, the nose is the outline point farthest from the
  tail base; without a visible tail the head stays where it was. Dots: nose, front left / right leg, body centre,
  hind left / right leg, tail base, a dot every 0.2 body length along the tail (up to 6). Leg dots sit on the full
  outline (lateral extremes at 15–42 % / 58–85 % from the nose), so a paw that sticks out pulls its dot; they are not
  paw detection. A dot keeps its place while it moves less than the picture's jitter (≈ 0.04 body length), so a
  still part keeps still dots and a moving part moves them. The dots are drawn by the viewer at screen resolution
  (small, sharp). Motion is counted per part (body, head, legs, tail); the freezing decision uses the **body**
  only, so a tail that moves alone gives *Tail moved*, not body motion. Frame table: Body / Head / Legs / Tail
  motion %, Moving parts, Nose / Body centre / Tail base X/Y.
* Results open on the **Analysis & Results** page (*Showing* list: tracking and freezing results): Overview
  (epochs, time bins, motion trace + 4 key numbers), Motion trace, Epochs, Bouts, Time bins, Frame review,
  Statistics; Export = Excel / CSV / PNG / all as .zip / `.btf`.
* **`.btf`** (BT Freezing file, like `.bta`): motion data, settings, results, audit trail and a small video copy —
  opens without the original video; re-tuning + Save adds an audit entry.
* Freezing – Batch: double-click a running video (or *👁 Live view*) = live freezing view; a finished one =
  its result on the Analysis page.
* **Region = polygon + masks** (*⬠ Region*, *▨ Mask*, *Clear ▾*): only pixels inside the polygon and outside
  every mask count (motion, the noise estimate for the auto cutoff, "around the animal", body dots); tracking stays
  inside the polygon and never takes a masked place (reflections in clear walls, light streaks, a lamp, a cable).
  A box (older `.btf` files) or a rectangle polygon without masks is analysed exactly as before. Saved in the
  settings of the result and in the `.btf` (`region`: `{"poly": …, "masks": …}`).
* **📏 Measured length** (nose → tail base of the mouse, not its reflection) is also given to BT tracking: the
  tracker expects a mouse of that size (always used when measured).
  With *Whole region* and a measured length no tracking is needed at all (the threshold is % of that body).
* **Tracking** group: *Animal colour*, *Min contrast* (grey levels; higher = fewer reflections / shadows taken for
  the animal) and **🎯 Test detection**: BT tracking of ~1 s up to the shown frame, with the background of the
  analysis time window (as in the real run) → "ANIMAL FOUND" (green marker) or "NOT FOUND". Changing them asks
  for a new analysis (the stored motion was tracked with the old values).
* Speed: Freezing – Individual reads the video in pieces on all cores (the live picture follows the first piece);
  body dots are computed only in *Body dots* mode. An *Auto* cutoff at its maximum (40) gives a warning (noisy
  video: mask bright / flickering parts, use Gray).
* **Motion** = pixels whose grey value changes by ≥ the motion cutoff between a frame and the frame 1/30 s
  before (1 frame at 25–45 fps, 2 at 60 fps — same numbers at any frame rate), after a σ = 1 blur, inside the
  region. The change of a typical (median) 40 × 40 px tile is subtracted, so flicker and compression
  pulses are not movement. Given as **% of the animal's body area** (body size from BT tracking = 90th
  percentile of the tracked size, or measured: area = length² ÷ 2.4) → the threshold does not depend on the
  camera distance or video size.
* **Whole region** (default, Classic — the classic frame-difference count, frame for frame) or **Around the animal** (a circle of 1 body length around the BT-tracked animal).
* **Motion cutoff Auto** = 2 × the 99.99th percentile of the frame-to-frame change of the quiet 75 % of the
  tiles (75th percentile over frames) — the classic calibration rule on the video itself; *Empty video…* = the
  classic empty-chamber calibration (300 frames, 10 000 random pixels).
* **Freezing** = motion < threshold for ≥ minimum freeze (s); *ignore blips* (s, default off) fills short
  movements inside a freeze; *light-change guard*: frames where a typical tile changed > 25 % keep the state
  of the frame before.
* The video is read once (per-frame grey-change histograms are kept), so cutoff, threshold, durations,
  epochs and bins update instantly. Reading runs in up to 4 pieces at once; the pieces' overlapping frames are
  fingerprinted and, when seeking is not exact (MPEG-1, WMV …), the video is read in one piece instead.
* Output per video (`results/<video>/freezing/`): `<video>_Freezing.xlsx` (Freezing results with formula +
  meaning, Epochs, Time bins, Freezing bouts, Settings, Frame by frame, Column guide), `_Freezing_frames.csv`,
  `_Freezing_plot.png`. Batch: `Freezing_BatchSummary_<time>.xlsx` (Freezing summary — readable by Statistics —,
  Epochs, Time bins, Video & tracking info, Settings, Column guide).
* Tracking change that came with it: when part of the video is much brighter / darker (light cue, house
  light), the background samples are brought to the usual brightness first (`_even_brightness`; ordinary
  videos are not touched).

### Event Scoring ▸ 📊 Plots & analysis (BORIS-style, manual scoring only)
Events plot (ethogram), time budget (occurrences, total / mean / SD / min / max duration, % of time, inter-event
interval, latency, rate) + plot, time bins, transitions (counts + probabilities), inter-observer agreement
(Cohen's kappa on time slices, with the second observer's .btms). *Save plots + tables* = Excel + PNG (300 dpi)
+ SVG. The export of the scoring page also writes the time budget, 60 s bins and transitions.

### Event Scoring (new)
A keyboard behaviour-scoring page (`EventScoringPage` in `pro_ui/ux_controller.py`): define a Behavioral Inventory
(behaviours + keys + State/Point type), play the video and press keys to log events live; editable event
list, per-behaviour summary (bouts, total / mean duration, latency) and Excel / CSV export; inventories save /
load and the last one is remembered. Independent of the tracking pipeline. Single-tracking videos (Experiment
page) and the Batch-Processing queue are kept separate. A ↻ button on the Tracking and Analysis pages clears
the current results and view for a fresh run (saved files are never deleted).

### Removed
Automatic grooming / rearing classifier, ML training/inference and their files. (Behaviour scoring is now the
manual, keyboard-driven Event Scoring page above.)

---

Shortcuts: Ctrl+O open video · Ctrl+1…6 pages · Space play/pause · ←/→ frame step · Enter finish shape ·
Esc cancel · Ctrl+Z / Ctrl+Y undo/redo · Ctrl+A select all zones · Ctrl+D duplicate · Delete delete zone ·
Ctrl+arrows nudge · 0 reset zoom.
