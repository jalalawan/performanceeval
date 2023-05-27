# performanceeval
**EPA vs. LCS (Low Cost Sensor) Data Performance Evaluation
**
The repository contains analysis for Santa Monica sensors (n=16) for Quarter 0 (Oct - Dec 2018). Steps presented here were repeated for Santa Monica from Jan 2019 - Dec 2019, and for Pittsburgh from July 2019 - July 2020. Detailed analysis and datasets may be shared on request.

Note: Conda environment suggested for EPA's sensortoolkit package. Please follow instructions in .yml file.

![image](https://github.com/jalalawan/performanceeval/assets/39367591/171171c2-baa0-4656-9f70-b011ab173c9e)

**Background**:

- Guidance on low-cost sensor siting, deployment, and performance evaluation is recent and evolving (latest information in Enhanced Air Sensors Guidebook, 2022) which is used as the basis for this work.

From review of LCS Literature:

- Lack of long-term performance evaluations against reference monitors (typically > 1-year deployment) 
- “Multiple Deployments” refer to at least 3 sensors deployed in at least 2 cities and/or 2 EPA Regions
- Collocation refers to sensors located within ~6 miles and at a ± 1m height with an FRM / FEM
    - Average distance in SM is ~15 miles from FRM/FEM monitors recording (and sharing) PM, R.H, and Temp at hourly resolution
    - Less distance (~5 miles) between FRM / LCS sensors in PGH.
- Fundamental safeguards for the use of sensors (even for NSIM* applications) should include protocols that address data handling and initial processing, outlier detection and removal, data QA/QC for both FRM/LCS, and data completeness checks. 

![image](https://github.com/jalalawan/performanceeval/assets/39367591/b2901c00-d4b0-43dc-9578-74c54e259acc)

**Data Preprocessing (Step 0):
**![image](https://github.com/jalalawan/performanceeval/assets/39367591/d4e4374f-a918-4824-b4a3-939491fdbaf6)

**Data Preprocessing (Step 1):
**![image](https://github.com/jalalawan/performanceeval/assets/39367591/a125cc61-6486-4b98-9f3a-5442d7977b38)

**Data Preprocessing (Step 2-4):
**![image](https://github.com/jalalawan/performanceeval/assets/39367591/3c80ca3e-7cbd-4652-a318-d9f6f049a521)
![image](https://github.com/jalalawan/performanceeval/assets/39367591/0affcf31-6a9f-43bc-9e9c-c6221938e837)

**Performance Assessment Using EPA's "sensortoolkit" package (minor changes made to accommodate specific requirements, including calibration eqs. and visual analysis of results) / Data Preprocessing partly done using RStudio Desktop 2021.09:
**![image](https://github.com/jalalawan/performanceeval/assets/39367591/4058bb84-20f2-4335-89b6-e023319d525b)


![image](https://github.com/jalalawan/performanceeval/assets/39367591/c7313710-2ba4-4e33-b6e0-126b222ff4ee)


