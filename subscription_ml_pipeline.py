"""
Subscription Management System - Complete ML Model Training Script
================================================================

This script provides a complete end-to-end ML pipeline for subscription management
including churn prediction, plan recommendation, usage analysis, and pricing optimization.

Author: AI Assistant
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
import pickle
import os

class SubscriptionMLPipeline:
    """Complete ML Pipeline for Subscription Management System"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.models = {}
        self.scalers = {}
        self.results = {}

    def generate_dataset(self, n_customers=10000):
        """Generate realistic subscription management dataset"""
        print("🔄 Generating subscription management dataset...")

        # Customer demographics
        customer_data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(35, 12, n_customers).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n_customers),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_customers, p=[0.5, 0.3, 0.2]),
            'income_bracket': np.random.choice(['Low', 'Medium', 'High'], n_customers, p=[0.3, 0.5, 0.2]),
            'family_size': np.random.poisson(2.5, n_customers) + 1,
        }

        # Subscription details
        subscription_types = ['Fibernet_Basic', 'Fibernet_Premium', 'Broadband_Copper_Basic', 'Broadband_Copper_Premium']
        contract_types = ['Monthly', 'Quarterly', 'Yearly']

        subscription_data = {
            'subscription_type': np.random.choice(subscription_types, n_customers),
            'contract_type': np.random.choice(contract_types, n_customers, p=[0.6, 0.25, 0.15]),
            'monthly_charge': np.random.uniform(25, 150, n_customers),
            'data_quota_gb': np.random.choice([50, 100, 200, 500, 1000], n_customers),
            'tenure_months': np.random.exponential(18, n_customers).astype(int),
        }

        # Usage patterns
        usage_data = {
            'avg_monthly_usage_gb': np.random.lognormal(4, 1, n_customers),
            'peak_usage_hours': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_customers, p=[0.15, 0.25, 0.45, 0.15]),
            'support_tickets_3m': np.random.poisson(1.5, n_customers),
            'payment_delays_6m': np.random.poisson(0.5, n_customers),
            'auto_renew': np.random.choice([0, 1], n_customers, p=[0.3, 0.7]),
        }

        # Create DataFrame
        df = pd.DataFrame({**customer_data, **subscription_data, **usage_data})

        # Create realistic churn based on multiple factors
        churn_probability = (
            0.1 +  # base churn rate
            0.3 * (df['monthly_charge'] > 100).astype(int) +  # high price
            0.2 * (df['support_tickets_3m'] > 3).astype(int) +  # many support issues
            0.15 * (df['payment_delays_6m'] > 1).astype(int) +  # payment issues
            0.1 * (df['contract_type'] == 'Monthly').astype(int) +  # monthly contracts
            0.1 * (df['avg_monthly_usage_gb'] > df['data_quota_gb']).astype(int) -  # quota exceeded
            0.15 * (df['auto_renew'] == 1).astype(int) -  # auto renew reduces churn
            0.1 * (df['tenure_months'] > 24).astype(int)  # loyalty
        )

        df['churn'] = np.random.binomial(1, np.clip(churn_probability, 0, 1), n_customers)

        # Add derived features
        df['usage_quota_ratio'] = df['avg_monthly_usage_gb'] / df['data_quota_gb']
        df['price_per_gb'] = df['monthly_charge'] / df['data_quota_gb']
        df['is_heavy_user'] = (df['usage_quota_ratio'] > 0.8).astype(int)
        df['is_premium_customer'] = (df['monthly_charge'] > 80).astype(int)
        df['clv'] = (df['monthly_charge'] * df['tenure_months'] * (1 - df['churn'] * 0.5)).round(2)

        print(f"✅ Dataset generated: {len(df):,} customers, {df['churn'].mean():.2%} churn rate")
        return df

    def preprocess_data(self, df):
        """Preprocess data for ML models"""
        print("🔄 Preprocessing data...")
        df_processed = df.copy()

        # Encode categorical variables
        le_gender = LabelEncoder()
        df_processed['gender_encoded'] = le_gender.fit_transform(df_processed['gender'])

        # One-hot encode multi-category variables
        df_encoded = pd.get_dummies(df_processed, 
                                   columns=['location', 'income_bracket', 'subscription_type', 'contract_type', 'peak_usage_hours'], 
                                   prefix=['loc', 'income', 'sub', 'contract', 'peak'])

        # Drop original categorical columns
        df_encoded = df_encoded.drop(['gender'], axis=1)

        # Feature engineering
        df_encoded['price_efficiency'] = df_encoded['data_quota_gb'] / df_encoded['monthly_charge']
        df_encoded['support_intensity'] = df_encoded['support_tickets_3m'] / (df_encoded['tenure_months'] + 1)
        df_encoded['loyalty_score'] = df_encoded['tenure_months'] * (1 - df_encoded['payment_delays_6m'] * 0.1)

        print(f"✅ Data preprocessing complete: {df_encoded.shape[1]} features")
        return df_encoded

    def train_churn_model(self, df_processed):
        """Train churn prediction models"""
        print("🤖 Training churn prediction models...")

        # Prepare features and target
        X = df_processed.drop(['customer_id', 'churn'], axis=1)
        y = df_processed['churn']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models to train
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
        }

        results = {}
        for name, model in models.items():
            print(f"  Training {name}...")

            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
                'Model': model
            }

        # Find best model
        results_df = pd.DataFrame({k: {metric: v[metric] for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']} 
                                  for k, v in results.items()}).T

        best_model_name = results_df['AUC-ROC'].idxmax()
        best_model = results[best_model_name]['Model']

        print(f"✅ Best churn model: {best_model_name} (AUC-ROC: {results_df.loc[best_model_name, 'AUC-ROC']:.3f})")

        self.models['churn'] = best_model
        self.scalers['churn'] = scaler
        self.results['churn'] = results_df

        return best_model, results_df, X.columns

    def build_recommendation_system(self, df):
        """Build plan recommendation system"""
        print("🎯 Building plan recommendation system...")

        df_rec = df.copy()
        le_income = LabelEncoder()
        le_location = LabelEncoder()

        df_rec['income_encoded'] = le_income.fit_transform(df_rec['income_bracket'])
        df_rec['location_encoded'] = le_location.fit_transform(df_rec['location'])

        # Customer feature matrix
        feature_cols = ['age', 'family_size', 'avg_monthly_usage_gb', 'income_encoded', 
                       'location_encoded', 'is_heavy_user', 'support_tickets_3m']

        X_customers = df_rec[feature_cols].values

        # Standardize and cluster
        scaler_rec = StandardScaler()
        X_customers_scaled = scaler_rec.fit_transform(X_customers)

        kmeans = KMeans(n_clusters=5, random_state=self.random_state)
        customer_segments = kmeans.fit_predict(X_customers_scaled)
        df_rec['customer_segment'] = customer_segments

        # Plan recommendation logic
        def recommend_plan(usage_gb, family_size, income_encoded, age):
            if usage_gb > 200 and income_encoded == 2:  # High usage, high income
                return 'Fibernet_Premium'
            elif usage_gb > 200 and income_encoded == 1:  # High usage, medium income
                return 'Fibernet_Basic'
            elif family_size > 4:  # Large family
                return 'Fibernet_Premium'
            elif usage_gb < 50 and income_encoded == 0:  # Low usage, low income
                return 'Broadband_Copper_Basic'
            elif 50 <= usage_gb <= 200:  # Medium usage
                return 'Broadband_Copper_Premium'
            else:
                return 'Fibernet_Basic'

        df_rec['recommended_plan'] = df_rec.apply(
            lambda row: recommend_plan(row['avg_monthly_usage_gb'], row['family_size'], 
                                      row['income_encoded'], row['age']), axis=1
        )

        # Evaluate recommendations
        recommendation_accuracy = (df_rec['subscription_type'] == df_rec['recommended_plan']).mean()

        print(f"✅ Recommendation system built: {recommendation_accuracy:.3f} accuracy")

        self.models['recommendation'] = kmeans
        self.scalers['recommendation'] = scaler_rec

        return df_rec, kmeans, recommendation_accuracy

    def train_clv_model(self, df_processed):
        """Train Customer Lifetime Value prediction model"""
        print("💎 Training CLV prediction model...")

        clv_features = ['age', 'family_size', 'monthly_charge', 'data_quota_gb', 'avg_monthly_usage_gb',
                       'tenure_months', 'support_tickets_3m', 'payment_delays_6m', 'auto_renew',
                       'usage_quota_ratio', 'price_per_gb', 'is_heavy_user', 'is_premium_customer']

        X_clv = df_processed[clv_features]
        y_clv = df_processed['clv']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_clv, y_clv, test_size=0.2, random_state=self.random_state)

        # Scale features
        scaler_clv = StandardScaler()
        X_train_scaled = scaler_clv.fit_transform(X_train)
        X_test_scaled = scaler_clv.transform(X_test)

        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
        }

        results = {}
        for name, model in models.items():
            print(f"  Training {name}...")

            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            results[name] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2_Score': r2_score(y_test, y_pred),
                'Model': model
            }

        results_df = pd.DataFrame({k: {metric: v[metric] for metric in ['RMSE', 'MAE', 'R2_Score']} 
                                  for k, v in results.items()}).T

        best_model_name = results_df['R2_Score'].idxmax()
        best_model = results[best_model_name]['Model']

        print(f"✅ Best CLV model: {best_model_name} (R²: {results_df.loc[best_model_name, 'R2_Score']:.3f})")

        self.models['clv'] = best_model
        self.scalers['clv'] = scaler_clv
        self.results['clv'] = results_df

        return best_model, results_df

    def analyze_pricing_strategy(self, df):
        """Analyze and optimize pricing strategy"""
        print("💰 Analyzing pricing strategy...")

        # Plan pricing analysis
        plan_pricing = df.groupby('subscription_type').agg({
            'monthly_charge': 'mean',
            'data_quota_gb': 'mean',
            'price_per_gb': 'mean',
            'churn': 'mean',
            'customer_id': 'count'
        }).round(2)

        plan_pricing.columns = ['Avg_Price', 'Avg_Quota_GB', 'Price_per_GB', 'Churn_Rate', 'Customer_Count']

        # Generate recommendations
        recommendations = []
        for plan_type in plan_pricing.index:
            churn_rate = plan_pricing.loc[plan_type, 'Churn_Rate']

            if churn_rate > 0.25:
                recommendations.append(f'{plan_type}: REDUCE price by 10-15% (High churn: {churn_rate:.2%})')
            elif churn_rate < 0.15:
                recommendations.append(f'{plan_type}: INCREASE price by 5-10% (Low churn: {churn_rate:.2%})')
            else:
                recommendations.append(f'{plan_type}: MAINTAIN current pricing')

        print(f"✅ Pricing analysis complete: {len(recommendations)} recommendations generated")

        return plan_pricing, recommendations

    def save_models(self, output_dir='subscription_models'):
        """Save trained models and results"""
        print(f"💾 Saving models to {output_dir}...")

        os.makedirs(output_dir, exist_ok=True)

        # Save models
        for name, model in self.models.items():
            with open(f'{output_dir}/{name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)

        # Save scalers
        for name, scaler in self.scalers.items():
            with open(f'{output_dir}/{name}_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

        # Save results
        for name, result in self.results.items():
            if isinstance(result, pd.DataFrame):
                result.to_csv(f'{output_dir}/{name}_results.csv')

        print(f"✅ Models saved successfully to {output_dir}/")

    def run_complete_pipeline(self, n_customers=10000):
        """Run the complete ML pipeline"""
        print("🚀 STARTING SUBSCRIPTION MANAGEMENT ML PIPELINE")
        print("=" * 60)

        # 1. Generate dataset
        df = self.generate_dataset(n_customers)

        # 2. Preprocess data
        df_processed = self.preprocess_data(df)

        # 3. Train churn model
        churn_model, churn_results, feature_names = self.train_churn_model(df_processed)

        # 4. Build recommendation system
        df_with_recs, rec_model, rec_accuracy = self.build_recommendation_system(df)

        # 5. Train CLV model
        clv_model, clv_results = self.train_clv_model(df_processed)

        # 6. Analyze pricing strategy
        pricing_analysis, pricing_recs = self.analyze_pricing_strategy(df)

        # 7. Save models
        self.save_models()

        # Final summary
        print("\n🎉 PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"📊 Dataset: {len(df):,} customers")
        print(f"🤖 Churn Model: {churn_results['AUC-ROC'].max():.3f} AUC-ROC")
        print(f"🎯 Recommendation Accuracy: {rec_accuracy:.3f}")
        print(f"💎 CLV Model: {clv_results['R2_Score'].max():.3f} R²")
        print(f"💰 Pricing Recommendations: {len(pricing_recs)} generated")

        return {
            'dataset': df,
            'processed_data': df_processed,
            'churn_results': churn_results,
            'recommendation_accuracy': rec_accuracy,
            'clv_results': clv_results,
            'pricing_analysis': pricing_analysis,
            'pricing_recommendations': pricing_recs
        }

def main():
    """Main execution function"""
    print("📱 SUBSCRIPTION MANAGEMENT SYSTEM ML PIPELINE")
    print("=" * 60)
    print("This script trains machine learning models for:")
    print("• Customer churn prediction")
    print("• Subscription plan recommendation")
    print("• Customer lifetime value prediction")
    print("• Pricing strategy optimization")
    print()

    # Initialize pipeline
    pipeline = SubscriptionMLPipeline()

    # Run complete pipeline
    results = pipeline.run_complete_pipeline(n_customers=10000)

    print("\n🎯 READY FOR DEPLOYMENT!")
    print("Models can be loaded using pickle and used for real-time predictions.")

if __name__ == "__main__":
    main()
