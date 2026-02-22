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

## Final Recommendations
1. Use tuned distributed RF (`mllib_rf_tuned`) for primary deployment.
2. Keep strict ingestion validation and quarantine monitoring.
3. Run broadcast and non-broadcast comparisons as regression checks after pipeline changes.
4. Select cluster profile by cost-per-1000-rows under SLA constraints.
