# Documentation for CTU-CHB Intrapartum Cardiotocography Database v1.0.0

## 1. Overview

The CTU-CHB Intrapartum Cardiotocography (CTG) Database contains 552 selected CTG recordings from a total of 9,164 collected between 2010 and 2012 at University Hospital Brno (UHB), in collaboration with Czech Technical University (CTU) in Prague.

- **Recordings:** 552 CTG recordings
- **Duration:** Up to 90 minutes, starting no more than 90 minutes before delivery
- **Signals:**
  - Fetal Heart Rate (FHR)
  - Uterine Contraction (UC)
- **Sampling frequency:** 4 Hz

---

## 2. Inclusion Criteria

Recordings included in the dataset meet the following criteria:

- Singleton pregnancy
- Gestational age over 36 weeks
- No known fetal developmental defects
- Second stage of labor duration ≤ 30 minutes
- FHR signal quality > 50% in every 30-minute window
- Available biochemical analysis of umbilical artery blood (e.g., pH)
- Majority vaginal deliveries; 46 cesarean section (CS) recordings

---

## 3. Additional Available Data

The dataset also includes clinical and outcome data:

- **Maternal data:** age, parity, gravidity
- **Delivery data:** delivery type (vaginal, operative vaginal, cesarean), labor duration, meconium-stained amniotic fluid, measurement type (ultrasound or scalp electrode)
- **Fetal data:** sex, birth weight
- **Fetal outcome data:** umbilical artery blood analysis (pH, pCO2, pO2, base excess, BDecf), Apgar scores, neonatal evaluation (oxygen requirement, seizures, NICU admission)
- **Expert CTG evaluation:** by 9 obstetricians based on FIGO guidelines (not yet available)

---

## 4. Purpose of the Dataset

This dataset was created to provide a homogeneous, high-quality CTG data source for:

- Research and development of CTG signal analysis methods
- Monitoring fetal condition during labor
- Developing early detection algorithms for fetal hypoxia and labor complications

---

## 5. Fetal Hypoxia and Classification

### What is Fetal Hypoxia?

Fetal hypoxia is a condition of insufficient oxygen supply to the fetus during labor, potentially causing serious complications.

### Hypoxia Indicators

- The primary indicator is **umbilical artery blood pH**.
- Hypoxia is associated with **metabolic acidosis**, reflected by decreased pH.

### pH Thresholds for Hypoxia

- pH < 7.20: mild hypoxia or acidosis
- pH < 7.10: moderate to severe hypoxia
- pH < 7.00: severe hypoxia with high risk of complications

### Studies and Classification

- The dataset provides pH values as ground truth for hypoxia.
- Researchers have developed machine learning and deep learning models using CTG signals (FHR and UC) to predict hypoxia based on pH.
- The dataset does not include explicit hypoxia labels; labels must be derived from pH values.

---

## 6. Example: Labeling Hypoxia Based on pH

Assuming you have pH data for each CTG recording in a table:
