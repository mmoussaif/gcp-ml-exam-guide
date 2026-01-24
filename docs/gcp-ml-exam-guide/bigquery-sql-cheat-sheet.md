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

---

### 12) “Exam-style” BigQuery SQL reminders

- BigQuery is excellent for **feature aggregation** and large-scale analytics; it often beats moving data to Spark for simple transforms.
- For time series, beware **random splits** and ensure your SQL joins respect time boundaries.
