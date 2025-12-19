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

# TinyML para Detecção de Distúrbios de Qualidade de Energia

Este repositório contém a implementação de uma abordagem baseada em TinyML para detecção e classificação de distúrbios de Qualidade de Energia utilizando um microcontrolador ESP32.

O projeto dá suporte aos resultados experimentais apresentados no artigo acadêmico associado e inclui tanto o pipeline de treinamento dos modelos quanto a implementação de inferência embarcada.

## Estrutura do repositório

- `python/`  
  Scripts e notebooks para geração de dados, extração de características, treinamento e avaliação dos modelos.

- `esp32/`  
  Implementação embarcada dos modelos TinyML utilizando TensorFlow Lite e EloquentTinyML.

## Distúrbios considerados

- Normal (senoide pura)
- Afundamento de tensão (sag)
- Elevação de tensão (swell)
- Harmônicos
- Transitórios oscilatórios

## Hardware

- Microcontrolador ESP32

## Observações

Os testes embarcados têm como objetivo a validação funcional do pipeline de inferência e não representam uma avaliação estatística em campo.
