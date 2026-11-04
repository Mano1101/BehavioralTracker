# Statistical Design Safeguards — BehavioralTracker Pro v7

## Added safeguards
- Detects Subject/subject/Subject_ID/Animal identifiers when available.
- Uses the subject/animal as the experimental unit for between-group inference.
- Repeated rows/videos from the same subject are averaged before group tests to reduce pseudoreplication.
- Subjects with conflicting Group labels are excluded rather than silently reassigned.
- Reports analysis unit, subject column, repeated-row aggregation, and excluded conflicting subjects.
- Benjamini–Hochberg FDR correction is applied across tested metrics.
- If no subject identifier is present, the output explicitly says `row/video` analysis and does not claim animal-level inference.
- No Group or Subject assignments are invented by the software.

## Statistical scope
The application provides analysis helpers, not a substitute for a prespecified statistical analysis plan. Paired/repeated-measures designs, nested litters/cages, longitudinal models, and complex factorial designs require researcher-selected methods outside this automatic helper.
