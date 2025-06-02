# CAISO Electricity Price Forecasting with LSTM

![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Colab](https://img.shields.io/badge/Open%20in-Colab-yellow)

## Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [License](#-license)

## 📌 Project Overview

This project implements an LSTM neural network to forecast electricity prices (Locational Marginal Prices) across California ISO zones using historical price data.

**Key Specifications:**
- Input: 24 periods (6 hours) of historical data
- Output: 16 periods (4 hours) of forecasted prices
- Zones: SP-15, NP-15, ZP26, PGE-TDLMP, SCE-TDLMP

## 🛠️ Features

- **Multi-step forecasting** predicts 4 hours of prices at once
- **Automatic data preprocessing** handles timestamps and normalization
- **Multiple zone support** processes all major CAISO zones
- **Visualization tools** for comparing predictions vs actuals

## 📦 Installation

### Prerequisites
- Python 3.8+
- Google Colab account (for cloud execution)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/caiso-electricity-forecast.git
cd caiso-electricity-forecast