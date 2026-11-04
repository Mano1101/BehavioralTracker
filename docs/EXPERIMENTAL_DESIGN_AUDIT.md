# Experimental Design Layer v10

## Implemented

The Experiment page now exposes **Experimental Design…** and stores explicit research metadata:

- Animal ID
- Group
- Sex
- Dam/Cage
- Session
- Test
- Timepoint
- Measurement ID

## Repeated measurements

Multiple rows can be created for the same Animal ID across sessions/timepoints. The application preserves these rows rather than treating them as independent animals.

## Validation safeguards

Required fields: Animal ID, Group, Session, Test.
Duplicate Animal/Session/Test/Measurement ID combinations are rejected.
Missing relationships are not inferred automatically.

## Output connection

Each successful tracking result can now receive:

- `experimental_design.json`
- `experimental_design.csv`

The metadata is carried in the tracking setup and therefore remains available for downstream analysis/statistics.

## Statistical boundary

This layer records design structure; it does not automatically choose a mixed-effects/repeated-measures model. Dam/cage nesting and longitudinal dependence require researcher-level statistical specification.
