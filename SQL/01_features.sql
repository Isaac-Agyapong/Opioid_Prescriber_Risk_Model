-- =====================================================================
-- Early-warning feature table (schema ml, database opioid_analytics)
--
-- One row per prescriber per snapshot year T (2020, 2021, 2022):
--   population : individual prescribers who are eligible in year T (100+ Part D claims, unsuppressed opioid
--                count) and are NOT a peer outlier in year T, and who are eligible again in T+1 and T+2
--                (so the outcome can be observed)
--   label      : becomes a peer outlier in T+1 or T+2
--   features   : only information available at the end of year T (years T and T-1)
--
-- Source tables come from the Medicare_Opioid_Prescribing project (core.*, analytics.prescriber_benchmark).
-- =====================================================================

DROP SCHEMA IF EXISTS ml CASCADE;
CREATE SCHEMA ml;

CREATE TABLE ml.prescriber_snapshot AS
WITH b AS (
    SELECT data_year, npi, specialty_id, state_fips, rural, total_claims, opioid_claims, la_opioid_claims,
           opioid_day_supply, opioid_rate, peer_count, peer_median_rate, peer_p99_rate, peer_percentile,
           is_high_outlier
    FROM analytics.prescriber_benchmark
),
future AS (
    SELECT s.data_year AS t, s.npi,
           count(*)                                   AS future_years_observed,
           bool_or(f.is_high_outlier)                 AS became_outlier
    FROM b s
    JOIN b f ON f.npi = s.npi AND f.data_year IN (s.data_year + 1, s.data_year + 2)
    GROUP BY s.data_year, s.npi
)
SELECT
    c.data_year                                                     AS snapshot_year,
    c.npi,
    -- ------------------------------------------------ label
    fu.became_outlier::int                                          AS label,
    -- ------------------------------------------------ context
    d.specialty_group,
    st.census_region,
    coalesce(c.rural, false)::int                                   AS rural,
    c.peer_count,
    -- ------------------------------------------------ year T prescribing
    c.total_claims,
    c.opioid_claims,
    c.opioid_rate,
    c.peer_percentile,
    c.opioid_rate / nullif(c.peer_median_rate, 0)                   AS rate_vs_peer_median,
    c.opioid_rate / nullif(c.peer_p99_rate, 0)                      AS rate_vs_peer_p99,
    coalesce(c.la_opioid_claims, 0) / nullif(c.opioid_claims, 0)    AS long_acting_share,
    c.opioid_day_supply / nullif(c.opioid_claims, 0)                AS days_per_opioid_claim,
    -- ------------------------------------------------ patients (year T)
    fp.total_beneficiaries,
    fp.opioid_beneficiaries::numeric / nullif(fp.total_beneficiaries, 0) AS opioid_patient_share,
    fp.bene_avg_age,
    fp.bene_avg_risk_score,
    -- ------------------------------------------------ change from T-1 (NULL if not eligible in T-1)
    (p.npi IS NOT NULL)::int                                        AS present_prev_year,
    c.opioid_rate - p.opioid_rate                                   AS rate_change_1y,
    c.peer_percentile - p.peer_percentile                           AS percentile_change_1y,
    c.opioid_claims / nullif(p.opioid_claims, 0) - 1                AS opioid_claims_growth_1y,
    coalesce(p.is_high_outlier, false)::int                         AS was_outlier_prev_year
FROM b c
JOIN future fu               ON fu.t = c.data_year AND fu.npi = c.npi AND fu.future_years_observed = 2
LEFT JOIN b p                ON p.npi = c.npi AND p.data_year = c.data_year - 1
JOIN core.dim_specialty d    ON d.specialty_id = c.specialty_id
JOIN core.dim_state st       ON st.state_fips = c.state_fips
JOIN core.fact_prescriber_year fp ON fp.npi = c.npi AND fp.data_year = c.data_year
WHERE c.data_year IN (2020, 2021, 2022)
  AND NOT c.is_high_outlier;

ALTER TABLE ml.prescriber_snapshot ADD PRIMARY KEY (snapshot_year, npi);

-- base rates per snapshot (reported in the README)
CREATE VIEW ml.label_summary AS
SELECT snapshot_year, count(*) AS prescribers, sum(label) AS positives,
       round(100.0 * avg(label), 3) AS positive_rate_pct
FROM ml.prescriber_snapshot
GROUP BY snapshot_year;
