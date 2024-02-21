# Feature Timeline

## For the end of Feb
 - Policy evaluation tool
     - <del>Oracle </del> MPC with perfect model?
  
     - <del>Return distribution
     - Reproducibility Metric
     - <del>Real plot axis naming
 - Customisation
    - <del> Model parameters
    - <del> Model Dynamics
    - <del> Constraint Functions
    - Cusomisation Documentation
      - Params
      - Model
      - Constraints
  - Model Reformulation as Python classes
    - <del>Allow disturbances for JAX models
    - <del>Expose model details (i.e m.info returns variable names for states, controls etc.)
    - <del>Change SP, Constraints, and disturbances to use variable names instead of '0', '1' etc.
    - <del>Allow for non-sequential definition of disturbances/constraints
    - <del>First Order system and Multistage extraction reformulation

## Feature Ideas
  - Policy evaluation
    - Learning curve plot
    - cross-validation
    - Plot custom constraints
  - Customisation
    - Reward function
  - Oracle
    - IMC Tuned FB controller (i.e. if MPC fails to converge this could be 
       used as a backup?)
    - Option to allow/disallow disturbance and setpoint foresight
  - Other
    - Ability to specify observable states
    - Leaderboard / Hackathon
