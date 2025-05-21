# ML-Optimized-Microstrip-Patch-Antenna

Welcome! This project explores how Machine Learning (ML) can be used to predict the performance of **flag-shaped microstrip patch antennas**—specifically **return loss** without relying on time-consuming electromagnetic simulations (like HFSS).


---

## Project Objective

Antenna design is often an iterative and computationally heavy task. By training an ML model on previously simulated antenna data, we aim to:

- Speed up the design and optimization process
- Make accurate predictions on antenna behavior based on geometric parameters
- Avoid running full-wave simulations every time

---

## What's Inside the Repo?

| File | Description |
|------|-------------|
| `Code.ipynb` | The main Jupyter notebook where all the data preprocessing, model training, and evaluation takes place. |
| `flag_shaped_antenna_dataset.xlsx` | The dataset containing various geometric features and their corresponding antenna performance values. |
| `HFSS simulation vs Model Prediction for a sample data.png` | Visual comparison of ML model predictions vs HFSS simulations. |
| `Workflow.png` | A helpful diagram showing the workflow of the entire project from data to prediction. |

---

## How It Works (ML Workflow)

1. **Input Data**: Geometric parameters of the antenna
2. **Output Labels**: Antenna performance metrics:
   - Return Loss (dB)
3. **Model Used**: Support Vector Regression and Random Forest, with and without Genetic Algorithm
4. **Training and Testing**:
   - Dataset split into training/testing (90% training 10% testing)
   - Model trained to learn mapping from geometry → performance
5. **Results Evaluated** using metrics like MSE and R² score.

---

## Dataset Info

- **Format**: Excel file (`.xlsx`)
- **Number of Samples**:
  - Total samples : 325
  - Training samples : 292
  - Testing samples : 33
- **Target Variable**:
  - Return Loss
- **Units**:
  - Dimensions: mm
  - Frequency: GHz

*Data generated through HFSS simulations.*

---

## Sample Output

![HFSS simulation vs Model Prediction for a sample data](https://github.com/user-attachments/assets/048475cb-5a0c-4eba-8851-e9f81af7748c)

The above graph shows how close the ML model’s predictions are to actual simulated results—showing the power and potential of ML in antenna engineering.





