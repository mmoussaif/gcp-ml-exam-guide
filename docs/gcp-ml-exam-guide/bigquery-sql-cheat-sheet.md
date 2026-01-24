## BigQuery SQL Manipulation + Cheat Sheet (ML Engineer Focus)

This is a practical SQL reference for the workflows that show up constantly in ML engineering on Google Cloud: preparing features, building labels, validating data, and doing lightweight analytics directly in **BigQuery**.

### Official docs (high-signal starting points)

- Query syntax (Standard SQL): `https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax`
- Functions & operators (Standard SQL): `https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators`
- Procedural language / scripting: `https://cloud.google.com/bigquery/docs/reference/standard-sql/scripting`
- Partitioned tables: `https://cloud.google.com/bigquery/docs/partitioned-tables`
- Query performance best practices: `https://cloud.google.com/bigquery/docs/best-practices-performance-overview`

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

**Useful functions**

- `COUNTIF(condition)`
- `ANY_VALUE(col)` (when you need “some value” to satisfy grouping)
- `APPROX_COUNT_DISTINCT(col)` (faster, approximate)
- `APPROX_QUANTILES(col, n)` (for p50/p90/p99-ish summaries)

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

```sql
-- SELECT * EXCEPT / REPLACE is common in feature pipelines
SELECT
  * EXCEPT(raw_text),
  LOWER(raw_text) AS raw_text
FROM `project.dataset.table`;
```

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

```sql
-- PIVOT: turn rows into columns (handy for feature wide tables)
SELECT *
FROM (
  SELECT user_id, event_name, 1 AS cnt
  FROM `project.dataset.events`
)
PIVOT (SUM(cnt) FOR event_name IN ('view', 'add_to_cart', 'purchase'));
```

```sql
-- UNPIVOT: turn wide columns back into rows (useful for analysis/debug)
SELECT user_id, feature_name, feature_value
FROM `project.dataset.user_features_wide`
UNPIVOT(feature_value FOR feature_name IN (f1, f2, f3));
```

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

```sql
-- Leakage check idea: ensure feature timestamp <= label timestamp
SELECT
  COUNTIF(feature_ts > label_ts) AS leakage_rows
FROM `project.dataset.training_view`;
```

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
