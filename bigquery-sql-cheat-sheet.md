## BigQuery SQL Cheat Sheet

A practical SQL reference for ML engineering workflows on Google Cloud: preparing features, building labels, validating data, and doing analytics directly in **BigQuery**.

### Table of Contents

- [Getting Started](#getting-started)
- [Core Query Patterns](#core-query-patterns)
- [Joins & Data Leakage Prevention](#joins--data-leakage-prevention)
- [Window Functions](#window-functions)
- [Dates & Timestamps](#dates--timestamps)
- [Safe Casting & Null Handling](#safe-casting--null-handling)
- [Arrays & Structs (Nested Data)](#arrays--structs-nested-data)
- [JSON & Semi-Structured Data](#json--semi-structured-data)
- [BigQuery Scripting](#bigquery-scripting)
- [Query Power Tools](#query-power-tools)
- [Text Manipulation & Regex](#text-manipulation--regex)
- [Data Quality Checks](#data-quality-checks)
- [Performance & Cost Optimization](#performance--cost-optimization)
- [BigQuery ML (BQML)](#bigquery-ml-bqml)
- [Stakeholder KPI Queries](#stakeholder-kpi-queries)
- [Real-World Analytics Patterns](#real-world-analytics-patterns)
- [Exam-Style Reminders](#exam-style-reminders)

---

## Getting Started

### Official Documentation

Quick links to the most useful BigQuery docs:

- **Query Syntax**: <a href="https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax">BigQuery Query Syntax</a>
- **Functions & Operators**: <a href="https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators">BigQuery Functions & Operators</a>
- **Scripting**: <a href="https://cloud.google.com/bigquery/docs/reference/standard-sql/scripting">BigQuery Scripting</a>
- **Partitioned Tables**: <a href="https://cloud.google.com/bigquery/docs/partitioned-tables">Partitioned Tables Documentation</a>
- **Performance Best Practices**: <a href="https://cloud.google.com/bigquery/docs/best-practices-performance-overview">BigQuery Performance Best Practices</a>

### BigQuery SQL Mental Model

**Key concepts:**

- **Standard SQL** (default) with BigQuery-specific types and functions
- Handles **nested data** with **STRUCT** and **ARRAY** types
- Scales best when you:
  - Filter on **partition** columns
  - Use **clustering** on common join/filter columns
  - Avoid unintentional data explosions (UNNEST mistakes)

#### How BigQuery Executes Queries

Understanding BigQuery's execution model helps you write efficient queries:

- BigQuery is a **distributed columnar** warehouse: you pay primarily for **bytes scanned**
- `WHERE` filters on the **partition column** enable **partition pruning** (big cost saver)
- `SELECT *` on wide tables can scan and return unnecessary columns
- Joins and UNNEST can multiply rows; always sanity-check row counts after joins

**💡 Cost Tip:** Filter early and filter on partition columns to minimize bytes scanned!

---

## Core Query Patterns

### Basic SELECT with Aggregation

```sql
-- Count events and purchases per user in the last 30 days
SELECT
  user_id,
  COUNT(*) AS events,
  COUNTIF(event_name = 'purchase') AS purchases
FROM `project.dataset.events`
WHERE event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY user_id
ORDER BY purchases DESC
LIMIT 100;
```

**What it does:** Counts total events and purchases per user, then ranks users by purchase count.

**Key functions:**

- `COUNTIF(condition)` - Count rows matching a condition
- `ANY_VALUE(col)` - Get "some value" when grouping (useful when all values are the same)
- `APPROX_COUNT_DISTINCT(col)` - Faster approximate distinct count
- `APPROX_QUANTILES(col, n)` - Get approximate percentiles (p50/p90/p99)

### DISTINCT: Count Unique Values

```sql
-- How many unique customers?
SELECT COUNT(DISTINCT customer_id) AS unique_customers
FROM `project.dataset.orders`;
```

**What it does:** Counts unique customers who placed at least one order.

### HAVING: Filter After Aggregation

```sql
-- Find power users (10+ sessions)
SELECT user_id, COUNT(*) AS sessions
FROM `project.dataset.sessions`
GROUP BY user_id
HAVING sessions >= 10
ORDER BY sessions DESC;
```

**What it does:** Filters to users with 10+ sessions. **Remember:** `HAVING` applies after `GROUP BY`, while `WHERE` applies before.

### CASE: Create Business Segments

```sql
-- Bucket users by purchase count
SELECT
  user_id,
  purchases,
  CASE
    WHEN purchases = 0 THEN '0'
    WHEN purchases BETWEEN 1 AND 2 THEN '1-2'
    WHEN purchases BETWEEN 3 AND 5 THEN '3-5'
    ELSE '6+'
  END AS purchase_bucket
FROM `project.dataset.user_purchase_counts`;
```

**What it does:** Converts raw counts into readable segments for reporting and analysis.

### UNION ALL: Combine Tables

```sql
-- Combine web and store orders
SELECT order_id, created_at, 'web' AS source FROM `project.dataset.web_orders`
UNION ALL
SELECT order_id, created_at, 'store' AS source FROM `project.dataset.store_orders`;
```

**What it does:** Combines two sources into one dataset. **Note:** `UNION ALL` doesn't remove duplicates (use `UNION` if you need deduplication).

---

## Joins & Data Leakage Prevention

### Basic Feature Join

```sql
-- Attach features to labeled rows
SELECT
  b.user_id,
  b.label,
  f.last_30d_purchases,
  f.last_30d_sessions
FROM `project.dataset.base_labels` b
LEFT JOIN `project.dataset.user_features_30d` f
USING (user_id);
```

**What it does:** Attaches feature columns to each labeled row, keeping all base rows even when features are missing.

### Join Types: When to Use Which

| Join Type      | Use Case                                                                            |
| -------------- | ----------------------------------------------------------------------------------- |
| **INNER JOIN** | Keep only rows that match in both tables (good for "must have feature" cases)       |
| **LEFT JOIN**  | Keep all labels/base rows even if features are missing (common for training tables) |
| **CROSS JOIN** | Cartesian product (almost always accidental unless you mean "expand")               |

**⚠️ Common Pitfalls:**

- **Many-to-many joins** silently multiply rows → always validate row counts
- **Future leakage** in time-series joins → ensure feature windows end _before_ label timestamp

### Time-Safe Feature Join (Anti-Leakage Pattern)

```sql
-- Build features using only events BEFORE the label timestamp
SELECT
  l.user_id,
  l.label_ts,
  l.label,
  COUNTIF(e.event_name = 'purchase') AS purchases_before_label
FROM `project.dataset.labels` l
LEFT JOIN `project.dataset.events` e
  ON e.user_id = l.user_id
 AND e.event_ts <= l.label_ts  -- CRITICAL: only use past events
 AND e.event_ts >= TIMESTAMP_SUB(l.label_ts, INTERVAL 30 DAY)
GROUP BY l.user_id, l.label_ts, l.label;
```

**What it does:** Builds time-safe features by using only events that occurred before the label timestamp. This prevents **data leakage** that can inflate offline metrics.

**💡 Exam Tip:** Always check that feature timestamps ≤ label timestamps in your training data!

---

## Window Functions

### Deduplicate: Keep Latest Record Per Key

```sql
-- Keep the most recent row per user
SELECT * EXCEPT(rn)
FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
  FROM `project.dataset.users`
)
WHERE rn = 1;
```

**What it does:** Keeps the most recent row per user (typical "latest state" table cleanup).

### Rolling Window: Last 7 Rows

```sql
-- Rolling count of events in the last 7 rows (not time-based)
SELECT
  user_id,
  event_ts,
  COUNT(*) OVER (
    PARTITION BY user_id
    ORDER BY event_ts
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS last_7_events
FROM `project.dataset.events`;
```

**What it does:** Rolling count over the current + previous 6 rows per user (not a time-duration window).

### Key Window Functions

| Function                                                     | Purpose                              |
| ------------------------------------------------------------ | ------------------------------------ |
| `ROW_NUMBER()`                                               | Sequential numbering (1, 2, 3...)    |
| `RANK()`                                                     | Ranking with gaps (1, 2, 2, 4...)    |
| `DENSE_RANK()`                                               | Ranking without gaps (1, 2, 2, 3...) |
| `LAG(col, n)`                                                | Value from n rows before             |
| `LEAD(col, n)`                                               | Value from n rows ahead              |
| `SUM() OVER (...)`, `AVG() OVER (...)`, `COUNT() OVER (...)` | Aggregations over window             |

### Window Frames: ROWS vs RANGE

- **ROWS**: "N physical rows before/after" (depends on row ordering, not time gaps)
- **RANGE**: "value-based window" (e.g., timestamps within a range). Use with care.

**Example:** `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` = last 7 rows, regardless of time gaps.

---

## Dates & Timestamps

### Common Date Transformations

```sql
-- Convert and bucket timestamps
SELECT
  DATE(event_ts) AS event_date,
  TIMESTAMP_TRUNC(event_ts, HOUR) AS event_hour,
  DATE_TRUNC(DATE(event_ts), WEEK(MONDAY)) AS week_start
FROM `project.dataset.events`;
```

**Common date functions:**

- `CURRENT_DATE()`, `CURRENT_TIMESTAMP()` - Get current date/time
- `DATE_ADD(date, INTERVAL n DAY)`, `DATE_SUB(date, INTERVAL n DAY)` - Add/subtract days
- `TIMESTAMP_ADD(ts, INTERVAL n HOUR)`, `TIMESTAMP_SUB(ts, INTERVAL n HOUR)` - Add/subtract time
- `*_TRUNC` - Bucket by day/week/month/etc.

**💡 Tip:** Use `DATE_TRUNC` for clean weekly/monthly aggregations!

---

## Safe Casting & Null Handling

### Safe Operations to Avoid Pipeline Breakage

```sql
SELECT
  SAFE_CAST(value AS INT64) AS value_int,  -- Returns NULL instead of error
  IFNULL(col, 0) AS col_filled,            -- Replace NULL with 0
  COALESCE(col1, col2, 'unknown') AS first_non_null  -- First non-NULL value
FROM `project.dataset.table`;
```

**Useful safe functions:**

- `SAFE_CAST(value AS TYPE)` - Returns NULL instead of error on cast failure
- `SAFE_DIVIDE(a, b)` - Returns NULL if `b = 0` (prevents divide-by-zero)
- `NULLIF(x, 0)` - Returns NULL if `x = 0` (useful before division)
- `IFNULL(col, default)` - Replace NULL with default value
- `COALESCE(col1, col2, ...)` - Return first non-NULL value

**💡 Best Practice:** Use `SAFE_DIVIDE` instead of regular division to avoid errors!

---

## Arrays & Structs (Nested Data)

### UNNEST: Explode Arrays

```sql
-- Expand order items into individual rows
SELECT
  user_id,
  item.item_id,
  item.price
FROM `project.dataset.orders` o,
UNNEST(o.items) AS item;
```

**What it does:** Expands nested arrays into individual rows for analysis.

### Build Arrays/Structs

```sql
-- Aggregate events into an array per user
SELECT
  user_id,
  ARRAY_AGG(STRUCT(event_name, event_ts) ORDER BY event_ts DESC LIMIT 10) AS last_events
FROM `project.dataset.events`
GROUP BY user_id;
```

**What it does:** Creates nested structures for feature generation or export.

**⚠️ Common UNNEST Traps:**

- `FROM t, UNNEST(arr)` is a cross join → row explosion if you don't mean it
- If `arr` can be NULL, consider `UNNEST(IFNULL(arr, []))`

### Aggregating After UNNEST

```sql
-- Calculate total order value from nested items
SELECT
  order_id,
  SUM(item.price * item.qty) AS order_value
FROM `project.dataset.orders` o,
UNNEST(IFNULL(o.items, [])) AS item
GROUP BY order_id;
```

**What it does:** Aggregates nested data after expanding it. Common pattern for feature engineering.

---

## JSON & Semi-Structured Data

### JSON Extraction

```sql
-- Extract values from JSON fields
SELECT
  JSON_VALUE(payload, '$.user.id') AS user_id,
  JSON_QUERY(payload, '$.items') AS items_json
FROM `project.dataset.events_json`;
```

**Functions:**

- `JSON_VALUE(json, path)` - Extract scalar value (string/number)
- `JSON_QUERY(json, path)` - Extract JSON object/array

**Note:** Function availability depends on whether the field is JSON type or STRING type.

---

## BigQuery Scripting

Useful when you need multi-step SQL jobs, variables, and dynamic SQL:

```sql
-- Declare variables and use them in queries
DECLARE start_date DATE DEFAULT DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY);
DECLARE end_date DATE DEFAULT CURRENT_DATE();

CREATE TEMP TABLE tmp AS
SELECT *
FROM `project.dataset.events`
WHERE DATE(event_ts) BETWEEN start_date AND end_date;

-- Dynamic SQL example
EXECUTE IMMEDIATE FORMAT("""
  CREATE OR REPLACE TABLE `project.dataset.features_%s` AS
  SELECT user_id, COUNT(*) AS events
  FROM `project.dataset.events`
  WHERE DATE(event_ts) BETWEEN DATE('%s') AND DATE('%s')
  GROUP BY user_id
""", FORMAT_DATE('%Y%m%d', end_date), start_date, end_date);
```

**Use cases:**

- Parameterized queries for pipelines
- Multi-step transformations
- Dynamic table creation

---

## Query Power Tools

### QUALIFY: Filter on Window Functions

```sql
-- Keep latest row per user (simpler than nested query)
SELECT
  user_id,
  updated_at,
  ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
FROM `project.dataset.users`
QUALIFY rn = 1;
```

**What it does:** Filters to the latest row per user using a window function in a single query. Much cleaner than nested queries!

### SELECT \* EXCEPT / REPLACE

```sql
-- Keep all columns but normalize one field
SELECT
  * EXCEPT(raw_text),
  LOWER(raw_text) AS raw_text
FROM `project.dataset.table`;
```

**What it does:** Avoids rewriting long SELECT lists when you only need to modify a few columns.

### WITH (CTEs): Readable Multi-Step Transformations

```sql
-- Structure complex logic into named steps
WITH base AS (
  SELECT * FROM `project.dataset.events`
),
agg AS (
  SELECT user_id, COUNT(*) AS events
  FROM base
  GROUP BY user_id
)
SELECT * FROM agg;
```

**What it does:** Makes complex queries readable by breaking them into logical steps.

### PIVOT: Rows to Columns

```sql
-- Turn event types into columns (wide feature table)
SELECT *
FROM (
  SELECT user_id, event_name, 1 AS cnt
  FROM `project.dataset.events`
)
PIVOT (SUM(cnt) FOR event_name IN ('view', 'add_to_cart', 'purchase'));
```

**What it does:** Produces a wide per-user feature table with one column per event type.

### UNPIVOT: Columns to Rows

```sql
-- Convert wide features back to long form
SELECT user_id, feature_name, feature_value
FROM `project.dataset.user_features_wide`
UNPIVOT(feature_value FOR feature_name IN (f1, f2, f3));
```

**What it does:** Converts wide features back to long form for easier inspection and comparisons.

---

## Text Manipulation & Regex

### Common Text Cleaning Patterns

```sql
SELECT
  LOWER(email) AS email_norm,
  REGEXP_REPLACE(phone, r'\\D', '') AS digits_only,
  REGEXP_CONTAINS(url, r'utm_') AS has_utm
FROM `project.dataset.users`;
```

**Common functions:**

- `LOWER()`, `UPPER()`, `TRIM()` - Case and whitespace normalization
- `REGEXP_REPLACE(str, pattern, replacement)` - Replace matching patterns
- `REGEXP_CONTAINS(str, pattern)` - Check if pattern matches
- `REGEXP_EXTRACT(str, pattern)` - Extract matching substring

---

## Data Quality Checks

### Quick Sanity Checks

```sql
-- Check null rate and duplicates
SELECT
  COUNT(*) AS rows,
  COUNTIF(user_id IS NULL) / COUNT(*) AS null_user_id_rate,
  COUNT(DISTINCT user_id) AS distinct_users
FROM `project.dataset.table`;
```

**What it does:** Quick sanity check on size, missing key rate, and unique-key cardinality.

### Leakage Detection

```sql
-- Ensure feature timestamp <= label timestamp
SELECT
  COUNTIF(feature_ts > label_ts) AS leakage_rows
FROM `project.dataset.training_view`;
```

**What it does:** Detects "future info" leakage that can inflate offline metrics. Should return 0!

---

## Performance & Cost Optimization

### BigQuery Performance Best Practices

**Key strategies:**

1. **Partition your large fact tables** (often by date) and always filter on the partition column
2. **Cluster on common filters/joins** (e.g., `user_id`, `account_id`) for faster scans
3. **SELECT only needed columns** (avoid `SELECT *` on wide tables)
4. **Avoid accidental blowups** with UNNEST and many-to-many joins
5. **Use `EXPLAIN`** to understand query stages

**Common tips:**

- **Filter early** and avoid scanning unnecessary partitions
- Avoid **cross joins** unless intended (including accidental UNNEST explosions)
- Prefer **approximate** aggregation functions where exactness isn't required

### Partition Pruning Example

```sql
-- Good: partition filter (assumes DATE partition column named event_date)
SELECT COUNT(*)
FROM `project.dataset.events_partitioned`
WHERE event_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) AND CURRENT_DATE();
```

**What it does:** Filters by partition to reduce scanned bytes (faster and cheaper).

**💡 Cost Tip:** Always filter on partition columns in WHERE clauses!

### Cost/Debug Tools

- Use `EXPLAIN` to understand query stages
- Run queries with limits and narrowed partitions first to validate logic
- Prefer pre-aggregating expensive transforms into feature tables if reused

---

## BigQuery ML (BQML)

### Create a Model

```sql
-- Create a logistic regression model
CREATE OR REPLACE MODEL `project.dataset.my_model`
OPTIONS(
  model_type='LOGISTIC_REG',
  auto_class_weights=TRUE,
  input_label_cols=['label']
) AS
SELECT
  * EXCEPT(example_id)
FROM `project.dataset.training_table`;
```

**Common model types:**

- `LOGISTIC_REG` - Binary classification
- `LINEAR_REG` - Regression
- `BOOSTED_TREE_CLASSIFIER` - Gradient boosting for classification
- `BOOSTED_TREE_REGRESSOR` - Gradient boosting for regression

### Predict with Model

```sql
-- Generate predictions
SELECT *
FROM ML.PREDICT(MODEL `project.dataset.my_model`, (
  SELECT * FROM `project.dataset.eval_table`
));
```

**💡 Exam Tip:** BQML doesn't support CNN or some advanced model types. Check the docs for supported models!

### MERGE: Upserts for Feature Tables

```sql
-- Maintain a "latest features" table with upserts
MERGE `project.dataset.user_features` T
USING `project.dataset.user_features_new` S
ON T.user_id = S.user_id
WHEN MATCHED THEN
  UPDATE SET
    last_30d_purchases = S.last_30d_purchases,
    updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
  INSERT (user_id, last_30d_purchases, updated_at)
  VALUES (S.user_id, S.last_30d_purchases, CURRENT_TIMESTAMP());
```

**What it does:** Upsert pattern to keep a "latest features" table current without full rebuilds.

### Views vs Materialized Views

| Type                  | Use Case                                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| **View**              | Saved query; recomputed each time (cheap to create; cost depends on underlying query)               |
| **Materialized View** | Precomputed results maintained by BigQuery; good for repeated aggregations (subject to constraints) |

**When to use:**

- **View**: Feature logic you want centralized and always "latest"
- **Materialized view**: Frequently queried aggregates that are expensive to recompute

---

## Stakeholder KPI Queries

These are "business analyst" patterns: clean metrics, clear definitions, easy to explain.

### Revenue and Orders by Week (Trend)

```sql
SELECT
  DATE_TRUNC(DATE(order_ts), WEEK(MONDAY)) AS week_start,
  COUNT(DISTINCT order_id) AS orders,
  SUM(order_amount) AS revenue
FROM `project.dataset.orders`
WHERE DATE(order_ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 WEEK)
GROUP BY week_start
ORDER BY week_start;
```

**What it does:** Weekly trend line for orders and revenue over the last 12 weeks.

### Conversion Funnel

```sql
WITH per_user AS (
  SELECT
    user_id,
    MAX(event_name = 'view') AS did_view,
    MAX(event_name = 'add_to_cart') AS did_add_to_cart,
    MAX(event_name = 'purchase') AS did_purchase
  FROM `project.dataset.events`
  WHERE DATE(event_ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  GROUP BY user_id
)
SELECT
  COUNTIF(did_view) AS viewers,
  COUNTIF(did_add_to_cart) AS add_to_cart_users,
  COUNTIF(did_purchase) AS purchasers,
  SAFE_DIVIDE(COUNTIF(did_add_to_cart), COUNTIF(did_view)) AS view_to_cart_rate,
  SAFE_DIVIDE(COUNTIF(did_purchase), COUNTIF(did_add_to_cart)) AS cart_to_purchase_rate
FROM per_user;
```

**What it does:** One-row funnel summary with step counts and conversion rates (safe divide prevents errors).

### Top Customers (Pareto-Style)

```sql
SELECT
  customer_id,
  SUM(order_amount) AS revenue
FROM `project.dataset.orders`
WHERE DATE(order_ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
GROUP BY customer_id
ORDER BY revenue DESC
LIMIT 50;
```

**What it does:** Identifies top revenue customers over the last 90 days (useful for retention/sales prioritization).

---

## Real-World Analytics Patterns

### Cohort Retention (Weekly)

```sql
-- Cohort users by first week seen, then compute retention by week offset
WITH first_seen AS (
  SELECT
    user_id,
    DATE_TRUNC(MIN(DATE(event_ts)), WEEK(MONDAY)) AS cohort_week
  FROM `project.dataset.events`
  GROUP BY user_id
),
activity AS (
  SELECT
    user_id,
    DATE_TRUNC(DATE(event_ts), WEEK(MONDAY)) AS activity_week
  FROM `project.dataset.events`
  GROUP BY user_id, activity_week
),
joined AS (
  SELECT
    f.cohort_week,
    a.user_id,
    DATE_DIFF(a.activity_week, f.cohort_week, WEEK) AS week_offset
  FROM first_seen f
  JOIN activity a
  USING (user_id)
  WHERE a.activity_week >= f.cohort_week
)
SELECT
  cohort_week,
  week_offset,
  COUNT(DISTINCT user_id) AS active_users
FROM joined
GROUP BY cohort_week, week_offset
ORDER BY cohort_week, week_offset;
```

**What it does:** Tracks how many users from each cohort return in subsequent weeks.

### A/B Test Analysis

```sql
-- Compute conversion rate by experiment group
WITH per_user AS (
  SELECT
    user_id,
    ANY_VALUE(variant) AS variant,  -- 'control' or 'treatment'
    MAX(event_name = 'purchase') AS converted
  FROM `project.dataset.experiment_events`
  WHERE experiment_id = 'exp_123'
  GROUP BY user_id
)
SELECT
  variant,
  COUNT(*) AS users,
  COUNTIF(converted) AS converters,
  SAFE_DIVIDE(COUNTIF(converted), COUNT(*)) AS conversion_rate
FROM per_user
GROUP BY variant;
```

**What it does:** Compares conversion rates between control and treatment groups.

### Sessionization (30-Minute Inactivity Gap)

```sql
-- Assign session_id based on a 30-minute inactivity threshold
WITH ordered AS (
  SELECT
    user_id,
    event_ts,
    LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) AS prev_ts
  FROM `project.dataset.events`
),
flags AS (
  SELECT
    *,
    IF(
      prev_ts IS NULL OR TIMESTAMP_DIFF(event_ts, prev_ts, MINUTE) > 30,
      1,
      0
    ) AS is_new_session
  FROM ordered
),
sessions AS (
  SELECT
    user_id,
    event_ts,
    SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY event_ts) AS session_id
  FROM flags
)
SELECT * FROM sessions;
```

**What it does:** Groups events into sessions based on 30-minute inactivity gaps.

### Incremental Backfill Pattern

```sql
-- Backfill one date (or a date range) into a partitioned table
DECLARE run_date DATE DEFAULT DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY);

INSERT INTO `project.dataset.daily_features` (event_date, user_id, events)
SELECT
  run_date AS event_date,
  user_id,
  COUNT(*) AS events
FROM `project.dataset.events`
WHERE DATE(event_ts) = run_date
GROUP BY user_id;
```

**What it does:** Incrementally processes one day at a time for partitioned feature tables.

### Anomaly Detection: Day-over-Day Change

```sql
-- Detect day-over-day changes in revenue
WITH daily AS (
  SELECT
    DATE(order_ts) AS d,
    SUM(order_amount) AS revenue
  FROM `project.dataset.orders`
  WHERE DATE(order_ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  GROUP BY d
)
SELECT
  d,
  revenue,
  LAG(revenue) OVER (ORDER BY d) AS prev_revenue,
  SAFE_DIVIDE(revenue - LAG(revenue) OVER (ORDER BY d), LAG(revenue) OVER (ORDER BY d)) AS pct_change
FROM daily
ORDER BY d;
```

**What it does:** Calculates day-over-day percentage changes to detect anomalies.

---

## Exam-Style Reminders

**Key points for the ML Engineer exam:**

- BigQuery is excellent for **feature aggregation** and large-scale analytics; it often beats moving data to Spark for simple transforms
- For time series, beware **random splits** and ensure your SQL joins respect time boundaries
- Always filter on **partition columns** to reduce costs
- Use **SAFE_DIVIDE** and **SAFE_CAST** to avoid pipeline failures
- Check for **data leakage** by ensuring feature timestamps ≤ label timestamps
- **BQML doesn't support CNN** - use Vertex AI for deep learning models

---

**💡 Tip:** Use Ctrl+F (or Cmd+F) to quickly search this document for specific SQL patterns or functions!
