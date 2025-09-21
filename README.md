# Real Estate Analytics

A comprehensive data science project for analyzing real estate properties in Gurgaon, India. This project includes web scraping, data cleaning, exploratory data analysis, feature engineering, and machine learning model development for property price prediction.

## 📋 Project Overview

This project focuses on analyzing real estate data from Gurgaon, Haryana, with the goal of understanding property market trends and building predictive models for property prices. The project encompasses the complete data science pipeline from data collection to model deployment.

## 🏗️ Project Structure

```
Real Estate Analytics/
├── Data Cleaning/                    # Data preprocessing and cleaning notebooks
│   ├── data-preprocessing-flats.ipynb
│   ├── data-preprocessing-houses.ipynb
│   ├── data-preprocessing-level-2.ipynb
│   ├── feature-engineering.ipynb
│   └── *.csv                        # Cleaned datasets
├── Data Visualisation/              # Data visualization and analysis
│   ├── data-visualization.ipynb
│   ├── data_viz1.csv
│   ├── feature_text.pkl
│   └── latlong.csv
├── EDA/                            # Exploratory Data Analysis
│   ├── eda-multivariate-analysis.ipynb
│   ├── eda-univariate-analysis.ipynb
│   └── merge-flats-and-house.ipynb
├── Feature Selection/              # Feature engineering and selection
│   ├── feature-selection-and-feature-engineering.ipynb
│   ├── feature-selection.ipynb
│   └── *.csv                       # Post-feature selection datasets
├── Model Selection/                # Machine learning model development
│   ├── model-selection.ipynb
│   ├── df.pkl
│   └── pipeline.pkl
├── Outlier Removal and mising value imputation/  # Data quality improvement
│   ├── missing-value-imputation.ipynb
│   └── outlier-treatment.ipynb
└── Web Scraping/                   # Data collection from web sources
    ├── apartments.ipynb
    ├── flats_appartment.ipynb
    ├── Independent_house.ipynb
    └── *.csv                       # Raw scraped data
```

## 🚀 Features

### Data Collection
- **Web Scraping**: Automated data collection from 99acres.com
- **Property Types**: Apartments, flats, and independent houses
- **Data Sources**: Multiple property listing websites

### Data Processing
- **Data Cleaning**: Comprehensive cleaning of raw scraped data
- **Missing Value Imputation**: Advanced techniques for handling missing data
- **Outlier Treatment**: Statistical methods for outlier detection and treatment
- **Feature Engineering**: Creation of new features like luxury scores, area calculations

### Data Analysis
- **Exploratory Data Analysis (EDA)**: Univariate and multivariate analysis
- **Data Visualization**: Interactive plots and charts using Plotly, Matplotlib, and Seaborn
- **Geographic Analysis**: Location-based insights with latitude/longitude data

### Machine Learning
- **Feature Selection**: Advanced feature selection techniques
- **Model Comparison**: Multiple regression models including:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - XGBoost
  - Neural Networks
  - Support Vector Regression
- **Model Evaluation**: Cross-validation and performance metrics

## 📊 Dataset Information

### Main Datasets
- **Flats Dataset**: 11,502+ apartment/flat listings
- **Houses Dataset**: Independent house listings
- **Combined Dataset**: Merged dataset with 3,800+ properties

### Key Features
- **Property Details**: Type, society, sector, price, area, bedrooms, bathrooms
- **Location Data**: Address, sector, nearby locations, coordinates
- **Property Specifications**: Floor number, facing direction, age, possession status
- **Amenities**: Furnishing details, features, luxury score
- **Pricing**: Price per square foot, total price

## 🛠️ Technologies Used

### Programming Languages
- **Python**: Primary language for data analysis and machine learning

### Libraries and Frameworks
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost
- **Web Scraping**: requests, beautifulsoup4
- **Data Analysis**: jupyter notebooks

### Data Sources
- **99acres.com**: Primary data source for property listings
- **Geographic Data**: Latitude/longitude coordinates for location analysis

## 📈 Key Insights

### Property Market Analysis
- Price distribution across different sectors in Gurgaon
- Correlation between property features and pricing
- Location-based price variations
- Impact of amenities and furnishing on property values

### Machine Learning Results
- Feature importance analysis for price prediction
- Model performance comparison
- Cross-validation results for model reliability

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required Python packages (see individual notebooks for specific requirements)

### Installation
1. Clone the repository
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost requests beautifulsoup4
   ```
3. Open Jupyter Notebook and run the notebooks in order

### Running the Project
1. Start with **Web Scraping** notebooks to collect fresh data
2. Run **Data Cleaning** notebooks to preprocess the data
3. Execute **EDA** notebooks for exploratory analysis
4. Use **Feature Selection** notebooks for feature engineering
5. Run **Model Selection** notebooks for machine learning

## 📝 Notebook Execution Order

1. **Web Scraping** → Data collection
2. **Data Cleaning** → Data preprocessing
3. **Outlier Removal and Missing Value Imputation** → Data quality improvement
4. **EDA** → Exploratory analysis
5. **Feature Selection** → Feature engineering
6. **Model Selection** → Machine learning model development
7. **Data Visualisation** → Results visualization

## 📊 Data Quality

- **Missing Values**: Handled using advanced imputation techniques
- **Outliers**: Detected and treated using statistical methods
- **Data Validation**: Comprehensive validation checks throughout the pipeline
- **Feature Engineering**: Created meaningful features for better model performance

## 🤝 Contributing

This is a data science project focused on real estate analysis. Contributions are welcome for:
- Additional data sources
- Improved feature engineering
- New visualization techniques
- Model optimization
- Documentation improvements

## 📄 License

This project is for educational and research purposes. Please ensure compliance with web scraping terms of service when collecting data.

## 📞 Contact

For questions or suggestions about this project, please open an issue in the repository.

---

**Note**: This project is focused on the Gurgaon real estate market. Results and insights may vary for other locations and time periods.
