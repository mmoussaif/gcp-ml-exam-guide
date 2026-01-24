## BigQuery SQL Manipulation + Cheat Sheet (ML Engineer Focus)

This is a practical SQL reference for the workflows that show up constantly in ML engineering on Google Cloud: preparing features, building labels, validating data, and doing lightweight analytics directly in **BigQuery**.

### Official docs (high-signal starting points)

- Query syntax (Standard SQL): [BigQuery Query Syntax](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax)
- Functions & operators (Standard SQL): [BigQuery Functions & Operators](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators)
- Procedural language / scripting: [BigQuery Scripting](https://cloud.google.com/bigquery/docs/reference/standard-sql/scripting)
- Partitioned tables: [Partitioned Tables Documentation](https://cloud.google.com/bigquery/docs/partitioned-tables)
- Query performance best practices: [BigQuery Performance Best Practices](https://cloud.google.com/bigquery/docs/best-practices-performance-overview)

### 0) BigQuery “SQL mental model”

- **Standard SQL** (default) with BigQuery-specific types and functions.
- Handles **nested data** with **STRUCT** and **ARRAY**.
- Scales best when you:
  - filter on **partition** columns
  - use **clustering**
  - avoid exploding data unintentionally (UNNEST mistakes)

#### How BigQuery executes queries (why some patterns are expensive)

- BigQuery is a **distributed columnar** warehouse: you pay primarily for **bytes scanned**.
- `WHERE` filters on the **partition column** enable **partition pruning** (big cost saver).
- `SELECT *` on wide tables can scan and return unnecessary columns.
- Joins and UNNEST can multiply rows; always sanity-check row counts after joins.

---

### 1) Core query patterns (SELECT / WHERE / GROUP BY)

```sql
-- Basic pattern
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

**Explanation:** Per user, count total events and purchases in the last 30 days, then rank users by purchases.

**Useful functions**

- `COUNTIF(condition)`
- `ANY_VALUE(col)` (when you need “some value” to satisfy grouping)
- `APPROX_COUNT_DISTINCT(col)` (faster, approximate)
- `APPROX_QUANTILES(col, n)` (for p50/p90/p99-ish summaries)

#### Core SQL clauses you must know (stakeholder reporting)

```sql
-- DISTINCT: how many unique customers?
SELECT COUNT(DISTINCT customer_id) AS unique_customers
FROM `project.dataset.orders`;
```

**Explanation:** Counts unique customers who placed at least one order.

```sql
-- HAVING: filter after aggregation (e.g., power users)
SELECT user_id, COUNT(*) AS sessions
FROM `project.dataset.sessions`
GROUP BY user_id
HAVING sessions >= 10
ORDER BY sessions DESC;
```

**Explanation:** Filters to users with 10+ sessions (HAVING applies after GROUP BY).

```sql
-- CASE: business-friendly segments
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

**Explanation:** Converts raw counts into readable segments for reporting.

```sql
-- UNION ALL: combine two tables with the same schema (don’t deduplicate)
SELECT order_id, created_at, 'web' AS source FROM `project.dataset.web_orders`
UNION ALL
SELECT order_id, created_at, 'store' AS source FROM `project.dataset.store_orders`;
```

**Explanation:** Combines two sources into one dataset without removing duplicates.

---

### 2) Joins (and ML data leakage traps)

```sql
-- Typical feature join: base table + aggregates
SELECT
  b.user_id,
  b.label,
  f.last_30d_purchases,
  f.last_30d_sessions
FROM `project.dataset.base_labels` b
LEFT JOIN `project.dataset.user_features_30d` f
USING (user_id);
```

**Explanation:** Attach feature columns to each labeled row; keep all base rows even when features are missing.

**Common pitfalls**

- **Many-to-many joins** silently multiply rows → always validate row counts.
- **Future leakage** in time-series joins → ensure feature windows end _before_ label timestamp.

#### Join types: when to use which

- **INNER JOIN**: keep only rows that match in both tables (good for “must have feature” cases).
- **LEFT JOIN**: keep all labels/base rows even if features are missing (common for training tables).
- **CROSS JOIN**: cartesian product (almost always accidental unless you mean “expand”).

#### A “time-safe” feature join pattern (anti-leakage)

```sql
-- For each label event, aggregate features using only prior events.
-- (Pattern: join on user_id AND event_ts <= label_ts, then aggregate.)
SELECT
  l.user_id,
  l.label_ts,
  l.label,
  COUNTIF(e.event_name = 'purchase') AS purchases_before_label
FROM `project.dataset.labels` l
LEFT JOIN `project.dataset.events` e
  ON e.user_id = l.user_id
 AND e.event_ts <= l.label_ts
 AND e.event_ts >= TIMESTAMP_SUB(l.label_ts, INTERVAL 30 DAY)
GROUP BY l.user_id, l.label_ts, l.label;
```

**Explanation:** Builds time-safe features by using only events that occurred before the label timestamp (anti-leakage).

---

### 3) Window functions (ranking, rolling stats, dedup)

```sql
-- Deduplicate: keep latest record per key
SELECT * EXCEPT(rn)
FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
  FROM `project.dataset.users`
)
WHERE rn = 1;
```

**Explanation:** Keeps the most recent row per user (typical “latest state” table cleanup).

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

**Explanation:** Rolling count over the current + previous 6 rows per user (not a time-duration window).

Key window functions:

- `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`
- `LAG()`, `LEAD()`
- `SUM() OVER (...)`, `AVG() OVER (...)`, `COUNT() OVER (...)`

#### Window frames: ROWS vs RANGE (quick intuition)

- **ROWS**: “N physical rows before/after” (depends on row ordering, not time gaps).
- **RANGE**: “value-based window” (e.g., timestamps within a range). Use with care.

---

### 4) Dates + timestamps (common transformations)

```sql
-- Convert and bucket
SELECT
  DATE(event_ts) AS event_date,
  TIMESTAMP_TRUNC(event_ts, HOUR) AS event_hour,
  DATE_TRUNC(DATE(event_ts), WEEK(MONDAY)) AS week_start
FROM `project.dataset.events`;
```

Common helpers:

- `CURRENT_DATE()`, `CURRENT_TIMESTAMP()`
- `DATE_ADD`, `DATE_SUB`, `TIMESTAMP_ADD`, `TIMESTAMP_SUB`
- `*_TRUNC` for bucketing

---

### 5) Safe casting + null handling (avoid pipeline breakage)

```sql
SELECT
  SAFE_CAST(value AS INT64) AS value_int, -- returns NULL instead of error
  IFNULL(col, 0) AS col_filled,
  COALESCE(col1, col2, 'unknown') AS first_non_null
FROM `project.dataset.table`;
```

Also useful:

- `SAFE_DIVIDE(a, b)` (returns NULL if `b = 0`)
- `NULLIF(x, 0)` to prevent divide-by-zero

---

### 6) Arrays + structs (nested data) + UNNEST

```sql
-- Explode an array safely
SELECT
  user_id,
  item.item_id,
  item.price
FROM `project.dataset.orders` o,
UNNEST(o.items) AS item;
```

```sql
-- Build arrays/structs (for feature generation or export)
SELECT
  user_id,
  ARRAY_AGG(STRUCT(event_name, event_ts) ORDER BY event_ts DESC LIMIT 10) AS last_events
FROM `project.dataset.events`
GROUP BY user_id;
```

**Common UNNEST traps**

- `FROM t, UNNEST(arr)` is a cross join → row explosion if you don’t mean it.
- If `arr` can be NULL, consider `UNNEST(IFNULL(arr, []))`.

#### Aggregating back after UNNEST (common feature pattern)

```sql
-- Example: total order value from nested items
SELECT
  order_id,
  SUM(item.price * item.qty) AS order_value
FROM `project.dataset.orders` o,
UNNEST(IFNULL(o.items, [])) AS item
GROUP BY order_id;
```

---

### 7) JSON + semi-structured fields

```sql
-- JSON extraction (function availability depends on type: JSON vs STRING)
SELECT
  JSON_VALUE(payload, '$.user.id') AS user_id,
  JSON_QUERY(payload, '$.items') AS items_json
FROM `project.dataset.events_json`;
```

---

### 7.1) BigQuery scripting (procedural SQL) for pipelines

Useful when you need multi-step SQL jobs, variables, and dynamic SQL:

```sql
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

---

### 7.2) Query syntax power tools (often used in production SQL)

```sql
-- QUALIFY lets you filter on window functions without nesting
SELECT
  user_id,
  updated_at,
  ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
FROM `project.dataset.users`
QUALIFY rn = 1;
```

**Explanation:** Filter to the latest row per user using a window function in a single query.

```sql
-- SELECT * EXCEPT / REPLACE is common in feature pipelines
SELECT
  * EXCEPT(raw_text),
  LOWER(raw_text) AS raw_text
FROM `project.dataset.table`;
```

**Explanation:** Keep all columns but normalize one field (avoid rewriting long SELECT lists).

```sql
-- WITH (CTEs) for readable multi-step transformations
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

**Explanation:** Structure complex logic into named steps for readability and reuse.

```sql
-- PIVOT: turn rows into columns (handy for feature wide tables)
SELECT *
FROM (
  SELECT user_id, event_name, 1 AS cnt
  FROM `project.dataset.events`
)
PIVOT (SUM(cnt) FOR event_name IN ('view', 'add_to_cart', 'purchase'));
```

**Explanation:** Produce a wide per-user feature table with one column per event type.

```sql
-- UNPIVOT: turn wide columns back into rows (useful for analysis/debug)
SELECT user_id, feature_name, feature_value
FROM `project.dataset.user_features_wide`
UNPIVOT(feature_value FOR feature_name IN (f1, f2, f3));
```

**Explanation:** Convert wide features back to long form for easier inspection and comparisons.

### 8) Text manipulation + regex (common cleaning)

```sql
SELECT
  LOWER(email) AS email_norm,
  REGEXP_REPLACE(phone, r'\\D', '') AS digits_only,
  REGEXP_CONTAINS(url, r'utm_') AS has_utm
FROM `project.dataset.users`;
```

---

### 9) Data quality checks (fast sanity queries)

```sql
-- Null rate + duplicates
SELECT
  COUNT(*) AS rows,
  COUNTIF(user_id IS NULL) / COUNT(*) AS null_user_id_rate,
  COUNT(DISTINCT user_id) AS distinct_users
FROM `project.dataset.table`;
```

**Explanation:** Quick sanity check on size, missing key rate, and unique-key cardinality.

```sql
-- Leakage check idea: ensure feature timestamp <= label timestamp
SELECT
  COUNTIF(feature_ts > label_ts) AS leakage_rows
FROM `project.dataset.training_view`;
```

**Explanation:** Detects “future info” leakage that can inflate offline metrics.

---

### 10) Performance + cost cheat sheet (BigQuery-specific)

- **Partition your large fact tables** (often by date) and always filter on the partition column.
- **Cluster on common filters/joins** (e.g., `user_id`, `account_id`) for faster scans.
- Prefer **SELECT only needed columns** (avoid `SELECT *` on wide tables).
- Avoid accidental blowups with `UNNEST` and many-to-many joins.
- Use `EXPLAIN` to understand query stages.

Common performance tips from the official guidance:

- **Filter early** and avoid scanning unnecessary partitions.
- Avoid **cross joins** unless intended (including accidental `UNNEST` explosions).
- Prefer **approximate** aggregation functions where exactness isn’t required.

#### Partition pruning example (the pattern you want)

```sql
-- Good: partition filter (example assumes a DATE partition column named event_date)
SELECT COUNT(*)
FROM `project.dataset.events_partitioned`
WHERE event_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) AND CURRENT_DATE();
```

**Explanation:** Filters by partition to reduce scanned bytes (faster and cheaper).

#### Cost/debug tools (practical)

- Use `EXPLAIN` to understand stages.
- Consider running queries with limits and narrowed partitions first to validate logic.
- Prefer pre-aggregating expensive transforms into feature tables if reused.

---

### 11) BigQuery ML (BQML) SQL quick hits

```sql
-- Create a model (example shape; pick the right model type for your use case)
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

```sql
-- Predict
SELECT *
FROM ML.PREDICT(MODEL `project.dataset.my_model`, (
  SELECT * FROM `project.dataset.eval_table`
));
```

### 11.1) MERGE (upserts) for feature tables

Useful when you maintain a “latest features” table:

```sql
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

**Explanation:** Upsert pattern to keep a “latest features” table current without full rebuilds.

### 11.2) Views vs materialized views (when to use)

- **View**: saved query; recomputed each time (cheap to create; cost depends on underlying query).
- **Materialized view**: precomputed results maintained by BigQuery; good for repeated aggregations (subject to constraints).

Use cases:

- **View**: feature logic you want centralized and always “latest”.
- **Materialized view**: frequently queried aggregates that are expensive to recompute.

---

### 12) “Exam-style” BigQuery SQL reminders

- BigQuery is excellent for **feature aggregation** and large-scale analytics; it often beats moving data to Spark for simple transforms.
- For time series, beware **random splits** and ensure your SQL joins respect time boundaries.

---

### 13) Stakeholder-ready KPI queries (examples you can put in a deck)

These are “business analyst” patterns: clean metrics, clear definitions, easy to explain.

#### Revenue and orders by week (trend)

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

**Explanation:** Weekly trend line for orders and revenue over the last 12 weeks.

#### Conversion funnel (simple event funnel)

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

**Explanation:** One-row funnel summary with step counts and conversion rates (safe divide prevents errors).

#### Top customers (Pareto-style)

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

**Explanation:** Identifies top revenue customers over the last 90 days (useful for retention/sales prioritization).

---

### 14) Real-world analytics & experimentation patterns

#### 14.1 Cohort retention (weekly)

```sql
-- Cohort users by first week seen, then compute retention by week offset.
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

#### 14.2 A/B test (simple lift + confidence-friendly summaries)

```sql
-- Compute conversion rate by experiment group.
WITH per_user AS (
  SELECT
    user_id,
    ANY_VALUE(variant) AS variant, -- 'control' or 'treatment'
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

#### 14.3 Sessionization (30-minute inactivity gap)

```sql
-- Assign session_id based on a 30-minute inactivity threshold.
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

#### 14.4 Incremental backfill into a partitioned table (date parameter)

```sql
-- Pattern: backfill one date (or a date range) into a partitioned table.
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

#### 14.5 Anomaly scan (day-over-day change for a KPI)

```sql
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
