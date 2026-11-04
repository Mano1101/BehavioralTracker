# Nested Experimental Design Audit

Version: v12

## Explicit hierarchy
- Animal ID = experimental subject/unit
- Cage ID = housing unit
- Dam ID = maternal/litter unit
- Group, Sex, Dam ID, Cage ID are required to remain consistent for repeated records of an Animal ID.
- Repeated records are keyed by Animal ID + Session + Test + Measurement ID.

## Statistical safeguard
The application does not infer missing Dam/Cage relationships. When both Dam ID and Cage ID are available, the mixed-effects engine constructs an explicit Dam→Cage variance component and keeps Animal ID as the subject random intercept. If the variance component is singular, the model reports that limitation instead of silently claiming that the nesting was modeled.

## Validation
Python compilation passed. GUI rendering was not claimed in this runtime.
