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

## General Workflow & Methodology

The research utilizes a multimodal sensor fusion approach to detect the developer's cognitive state while they code. The system gathers data from two primary physiological sources:

1. **Heart Rate (HR / HRV):**
   - Utilizing a PPG sensor (such as the Coospo HW9 HRM Arm Band) to collect Heart Rate Variability (HRV) data.
   - A dedicated application communicates with the device via Bluetooth Low Energy (BLE) to gather physiological responses that correlate with stress or high cognitive load.

2. **Eye Movement (Eye Tracking):**
   - Utilizing standard or high-framerate (60fps) webcams to track iris movements.
   - Processing the video feed using Machine Learning (e.g., MediaPipe) to map eye gaze to specific coordinates on the screen and editor lines.

### System Architecture Concept
- **Data Collection:** Dedicated services run locally to capture HR and Eye Tracking data in real-time.
- **Data Fusion & Processing:** The streams of data are synchronized (Multimodal Sensor Fusion). A machine learning model or algorithmic logic infers the cognitive state of the developer (e.g., *Focused, Confused, Scanning, Overloaded*).
- **Adaptive IDE Extension (VS Code):** A custom IDE extension acts as the client that receives the inferred state. It automatically triggers UI/UX changes. For instance:
  - Hiding the sidebar and entering Zen Mode during deep focus.
  - Enlarging font sizes or suggesting documentation when the developer appears confused or encounters errors.
  - Simplifying the interface to reduce visual noise when the developer is overloaded.