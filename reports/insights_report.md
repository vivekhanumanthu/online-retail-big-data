# Insight Report: UCI Online Retail

## Executive Summary
The pipeline converts transaction logs into actionable retention and cost-efficiency insights. The key decision is balancing model uplift against infrastructure spend.

## Insight 1: Data Quality and Returns Behavior
- Non-positive prices, missing customer IDs, and malformed transactions were quarantined.
- Return-heavy segments can distort revenue trends if not isolated.
- Recommendation: monitor return ratio by country weekly.

## Insight 2: Repeat-Purchase Prediction Signal
- Retail-specific engineered signals (`avg_item_value`, `items_per_unit_price`, `log_invoice_total`) improve behavior modeling.
- Tree-based models are typically strongest for mixed numeric/categorical invoice features.

## Insight 3: Distributed vs Single-Node Modeling
- sklearn baseline is useful for quick controls on sampled data.
- MLlib is preferred for full-scale CV and repeatable distributed training.

## Insight 4: Scaling Tradeoff
- Fastest runtime does not always minimize cost per 1000 rows.
- Mid-resource profiles often produce better cost-performance balance.

## Insight 5: Operational Decisioning
- Region/country level performance and risk scores support targeted retention campaigns.
- Use expected profit metric from evaluation output for campaign threshold selection.

## Clear Mapping: How to Show 4 Dashboards

### Dashboard 1: Data Quality and Pipeline Trust
- Goal: prove source data is reliable before model and business decisions.
- Data sources:
  - `outputs/tableau/dq_metrics_csv/*.csv`
  - `outputs/tableau/broadcast_compare_csv/summary.csv`
- Sheets:
  - KPI cards: `SUM(rows_after_cleaning)`, `AVG(duplicate_ratio)`, `AVG(avg_unit_price)`.
  - Country quality bar chart: `country` vs `rows_after_cleaning`.
  - Data anomaly view: `country` vs `duplicate_ratio` (sorted desc).
  - Pipeline join mode comparison: `mode` vs `runtime_sec`.
- Filters/controls: `country`, `mode`.
- Action: click `country` to filter Dashboard 3.

### Dashboard 2: Model Performance and Selection
- Goal: choose the best production model.
- Data sources:
  - `outputs/tableau/model_metrics_csv/*.csv`
  - `outputs/tableau/evaluation_csv/*.csv`
- Sheets:
  - Model score bar chart: `model_name` vs `auc`.
  - Model balance chart: `model_name` vs `f1`.
  - Evaluation KPI cards: `test_auc`, `test_f1`, `auc_ci_low_95`, `auc_ci_high_95`.
  - Profit KPI card: `expected_profit` (or calculated `expected_profit / test_count * 1000`).
- Filters/controls: `model_name`.
- Action: selected `model_name` drives narrative annotation in Dashboard 3 (campaign confidence).

### Dashboard 3: Business Impact and Targeting
- Goal: show where to run retention campaigns.
- Data sources:
  - `outputs/tableau/sales_metrics_csv/*.csv`
  - `outputs/tableau/evaluation_csv/*.csv` (for profit context)
- Sheets:
  - Map or filled chart: `country` colored by `gross_revenue`.
  - Segment scatter: `return_line_ratio` vs `gross_revenue` by `country`.
  - Region ranking table: `region`, `invoice_count`, `customer_count`, `avg_line_total`.
  - Risk/opportunity highlight: top countries by high `return_line_ratio` and high revenue.
- Filters/controls: `region`, `country`.
- Action: click `country` to open Dashboard 4 with cost/performance view for operational feasibility.

### Dashboard 4: Scalability and Cost Efficiency
- Goal: justify runtime and cloud-cost profile for deployment.
- Data sources:
  - `outputs/tableau/scaling_csv/*.csv`
  - `outputs/tableau/broadcast_compare_csv/summary.csv`
- Sheets:
  - Runtime by profile: `resource_profile` vs `runtime_sec`.
  - Cost efficiency chart: `resource_profile` vs `cost_per_1000_rows`.
  - Throughput chart: `resource_profile` vs `throughput_rows_per_sec`.
  - Strong scaling efficiency: `resource_profile` vs `strong_efficiency` (mode = `strong`).
  - Broadcast vs non-broadcast card: `mode` vs `runtime_sec`.
- Filters/controls: `mode`, `resource_profile`.
- Action: add parameter switch for recommendation label (Fastest / Cheapest / Balanced).

## Story Flow (Presentation Order)
1. Dashboard 1: establish trust in cleaned data and stable pipeline behavior.
2. Dashboard 2: show which model performs best on AUC/F1 and test metrics.
3. Dashboard 3: translate model quality into country/region business targeting.
4. Dashboard 4: prove operational cost and scaling plan for production rollout.
