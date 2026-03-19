# Sean's First 2 Weeks Starter Pack (USV EKF Baseline)

You are responsible for the baseline estimator so your team can compare In-EKF against something fair.

If you only do one thing, do this:
- Build a simple EKF that estimates `x, y, heading, speed`.
- Make it run on either fake data or simulator data.
- Output error metrics and one plot.

---

## Your role in one sentence

Create a **working baseline EKF/Error-State EKF** pipeline with clear metrics so the team can evaluate In-EKF performance under different sea states and currents.

---

## 14-day checklist (copy this and execute)

### Week 1: Get a clean baseline working

Day 1
- Confirm team conventions: coordinate frame, units, timestep, simulator I/O format.
- Create a tiny assumptions note in markdown.

Day 2
- Implement EKF prediction step and covariance propagation.
- Use a simple 2D motion model first.

Day 3
- Add GPS position update.
- Add heading update (compass/IMU yaw).

Day 4
- Run a synthetic straight-line test and a constant-turn test.
- Save plots for estimated vs true trajectory.

Day 5
- Add RMSE metrics (`x, y, heading`) and final drift metric.
- Push code + one short result summary.

### Week 2: Turn it into benchmark-quality output

Day 6
- Connect EKF to simulator data format (or mock same format if simulator is late).

Day 7
- Run calm-water scenario and verify stable behavior.

Day 8
- Run disturbed scenario (stronger wave/current) and record metric changes.

Day 9
- Add one robustness test: sensor dropout, bias, or increased noise.

Day 10
- Generate comparison plots/tables for at least 2 scenarios.
- Write 1-page methods/results note for team report.

---

## Minimum deliverables by end of week 2

- `ekf_baseline.py` (or equivalent) that runs end-to-end.
- Config values used for process/measurement noise.
- 3 to 5 plots:
  - trajectory
  - heading over time
  - error over time
  - optional covariance trace
- Metric table: RMSE and drift across scenarios.
- Short methods note documenting assumptions and equations.

---

## Questions to ask teammates right now

Send this in Discord:

1) "Can we lock state definition and units by tonight? I need this to finalize EKF interfaces."
2) "What exact simulator output fields and frequency will I receive?"
3) "What sea state/current cases are required for report figures?"

---

## Weekly update template (copy/paste)

"I am building the baseline EKF comparator (`x, y, heading, speed`) so we can evaluate In-EKF fairly.  
This week I will deliver: (1) running EKF pipeline, (2) RMSE/drift metrics, (3) first comparison plots for calm vs disturbed scenarios."

---

## Fast fallback if you get stuck

If integration blocks you:
- Use synthetic trajectory + noise injection.
- Demonstrate EKF behavior changes with noise strength.
- Still produce metrics/plots and mark simulator integration as next step.

This still counts as meaningful technical progress and gives your team reusable evaluation tooling.
