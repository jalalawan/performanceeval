# performanceeval
EPA vs. LCS (Low Cost Sensor) Data Performance Evaluation

Background:

- Guidance on low-cost sensor siting, deployment, and performance evaluation is recent and evolving (latest information in Enhanced Air Sensors Guidebook, 2022) which is used as the basis for this work.

From overall review of LCS Literature:

- Lack of long-term performance evaluations against reference monitors (typically > 1-year deployment) 
- “Multiple Deployments” refer to at least 3 sensors deployed in at least 2 cities and/or 2 EPA Regions
- Collocation refers to sensors located within ~6 miles and at a ± 1m height with an FRM / FEM
    - Average distance in SM is ~15 miles from FRM/FEM monitors recording (and sharing) PM, R.H, and Temp at hourly resolution
    - Less distance (~5 miles) between FRM / LCS sensors in PGH.
- Fundamental safeguards for the use of sensors (even for NSIM* applications) should include protocols that address data handling and initial processing, outlier detection and removal, data QA/QC for both FRM/LCS, and data completeness checks. 

![image](https://github.com/jalalawan/performanceeval/assets/39367591/b2901c00-d4b0-43dc-9578-74c54e259acc)

Preliminary Findings:

LCS performance w.r.t. both precision (w.r.t. collocated LCS) and accuracy (w.r.t. FRM/FEM) in SM failed to meet EPA recommended targets (esp. true for linearity and RMSE).
EPA U.S wide correction factor improves most performance metrics but still fails to meet most EPA targets.
Performance generally deteriorates after Q3 (i.e. post 9-months deployment)
Quadratic PM corrections with R.H as a covariate provides best correction results in SM.
Sensor / Reference R.H and Temp are positively correlated (R2 > ~0.7), R.H is positively correlated with PM2.5 consistent with literature. Sensors underestimate w.r.t. FRM/FEM but bias (slope/intercept) is reduced with correction equations. 
Precision metrics improve between 25-50% using EPA correction eq. Moderate, but unquantifiable improvement in accuracy metrics.
Lots of caveats / assumptions (FRM collocation unclear, averaging mechanism suggested by EPA may need improvement, linearity assumption should be checked with residuals / quintile regression, lack of long-term evaluations to compare results etc.)
PGH analysis shows better performance w.r.t both accuracy and precision. However, performance for most metrics deteriorates after Q3 (in-progress)
Overall results reveals (in)consistency in trends in EPA climate regions. Future work may include statistical tests such as ANOVA can be performed to systematically assess trends in LCS vs. FRM monitors at different collocation distances & metereologiesfor determining neighborhood level measurements from LCS (e.g. comparison in trends between SM and PGH at alpha = 0.05) (e.g. study here)
Future work may consider efficacy of over-the-cloud calibration based on collocated monitors, ML techniques such as classification / clustering for improved sensor siting, IP-rated sensor casings to avoid deliquescence-related errors, and more research using longer-duration deployments to estimate end-of-life. 
These steps may help move the needle towards a global certification regime for LCS sensors, and their fitness-for-use especially in ESJ communities.![image](https://github.com/jalalawan/performanceeval/assets/39367591/baeabe62-a2ee-456e-94ce-0af5aa2cdcf7)
