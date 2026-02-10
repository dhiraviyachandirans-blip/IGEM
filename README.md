An AI-powered web application for predicting peptide taste profiles and physicochemical properties, developed as part of the SaltenPep project for iGEM 2024 (Gold Medal Winner).
🔗 Live Demo: https://peptastepredictor01.streamlit.app/

📋 Table of Contents

Overview
Features
Technology Stack
Installation
Usage
Project Background
Model Architecture
Results
Contributing
Acknowledgments
Contact
License


🎯 Overview
PepTaste Predictor is a machine learning-powered platform designed to predict the taste characteristics of peptide sequences. This tool supports the development of salt-reducing peptides for healthier food formulations by enabling rapid screening of peptide candidates based on their predicted taste profiles.
Key Capabilities:

🧪 Peptide Sequence Analysis - Input custom peptide sequences for taste prediction
📊 Physicochemical Property Calculation - Automatic computation of molecular weight, charge, hydrophobicity, and more
🤖 AI-Driven Predictions - Machine learning models trained on peptide-taste relationship data
📈 Interactive Visualization - Real-time visualization of predictions and molecular properties
💾 Data Export - Download prediction results for further analysis


✨ Features
🔬 Core Functionality

Multi-Taste Classification: Predicts multiple taste categories (salty, umami, bitter, sweet)
Sequence-to-Properties Pipeline: Automated feature extraction from peptide sequences
Batch Processing: Analyze multiple peptides simultaneously
Real-time Predictions: Instant results upon sequence input

📊 Advanced Analytics

Physicochemical Profiling:

Molecular weight calculation
Isoelectric point (pI) estimation
Hydrophobicity index
Net charge at pH 7
Aromatic amino acid content
Aliphatic index



🎨 User Experience

Intuitive Interface: Clean, user-friendly Streamlit interface
Responsive Design: Works seamlessly across devices
Visual Feedback: Interactive charts and graphs
Help Documentation: Integrated tooltips and examples


🛠️ Technology Stack
Programming & Frameworks

Python 3.8+ - Core programming language
Streamlit - Web application framework
Pandas - Data manipulation and analysis
NumPy - Numerical computing

Machine Learning

Scikit-learn - Model training and evaluation
XGBoost/Random Forest - Classification algorithms (specify your actual model)
Feature Engineering - Custom peptide descriptor extraction

Data Visualization

Plotly - Interactive visualizations
Matplotlib - Statistical plots
Seaborn - Enhanced data visualization

Bioinformatics

Biopython - Peptide sequence analysis
Custom algorithms - Physicochemical property calculations
