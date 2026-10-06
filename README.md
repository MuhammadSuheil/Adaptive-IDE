# Adaptive IDE Interface based on Developer's Cognitive Load using Machine Learning and Multimodal Sensor Fusion

## Overview
This repository contains the software components and research context for developing an Adaptive Integrated Development Environment (IDE) extension. The core objective of this research is to apply Machine Learning (ML) and Artificial Intelligence (AI) to dynamically alter the IDE interface to reduce cognitive load and assist developers based on their current cognitive state.

## Research Team
**Head of Research:**
- Hadipurnawan Satria, Ph.D

**Researchers:**
1. Anggina Primanita, S.Kom., M.I.T., Ph.D.
2. Julian Supardi, S.Pd., M.T., Ph.D
3. Alvi Syahrini Utami, S. Si., M.Kom

**Student Researchers:**
- **Eye Tracking Sub-team:** M. Suheil Ichma Putra, M. Rabyndra Janitra Binello, Monica Amrina Rosyada
- **Heart Rate Sub-team:** Fitran Husein, Muhammad Alif Berri Rossi

## Project Components & Quick Start

1. **Code Arrange (Evaluation Platform)**
   A web-based Java code arrangement platform (`codearrange-web`) designed to serve as a controlled testing environment. It records developer interactions and evaluates cognitive load during specific programming tasks.
   - **How to Run:**
     ```powershell
     cd codearrange-web
     npm install # if not installed yet
     node server.js "Participant Name" "exam-set-a" # add --review to enable difficulty rating
     ```
     Then open `http://localhost:3000` in your browser.

2. **Multimodal Sensor Data Collection & Fusion**
   A system to capture physiological data in real-time for cognitive state inference.

   - **Eye Tracking:** Tracks eye gaze on the screen using a Webcam & MediaPipe.
     ```powershell
     python eye_tracking_prototype/eye_tracking_prototype.py
     ```
   - **Heart Rate Variability (HRV):** Records RR intervals from a PPG sensor (BLE) to extract RMSSD/SDNN features.
     ```powershell
     python -m hrv_monitor_prototype.calibrate
     ```
   - **Multimodal Fusion:** Runs both Eye Tracking and HRV synchronously. Once completed, the data from both sensors is automatically fused into a single timeline.
     ```powershell
     python run_multimodal.py --participant P01 --task coding_task
     ```

*(For more details regarding calibration, parameters, and architecture, refer to the `README.md` in their respective folders).*

## Research Goal (Adaptive Interface)
The inferred cognitive state data acquired from the modules above will eventually be used to trigger dynamic UI/UX adaptations in an Integrated Development Environment (either as a new simple IDE or an extension). The primary goal is to help developers reduce cognitive load (e.g., auto-enabling Zen Mode when overloaded, increasing font size during confusion, etc.).
