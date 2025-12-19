# TinyML-based power quality disturbance detection using ESP32

This repository contains the implementation of a TinyML-based approach for power quality disturbance detection and classification using an ESP32 microcontroller.

The project supports the experimental results presented in the associated academic article and includes both the training pipeline and the embedded inference implementation.

## Repository structure

- `python/`  
  Scripts and notebooks for data generation, feature extraction, model training and evaluation.

- `esp32/`  
  Embedded implementation of the trained TinyML models using TensorFlow Lite and EloquentTinyML.

## Disturbances considered

- Normal (pure sinusoid)
- Voltage sag
- Voltage swell
- Harmonics
- Oscillatory transients

## Hardware

- ESP32 microcontroller

## Notes

The embedded tests are intended for functional validation of the inference pipeline and do not represent statistical field evaluation.
