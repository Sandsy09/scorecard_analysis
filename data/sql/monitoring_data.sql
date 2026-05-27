SELECT
    c.customer_id,
    c.application_date,
    c.annual_income,
    c.employment_status,
    c.credit_bureau_score,
    c.months_at_address,
    d.ltv_ratio,
    d.loan_term_months,
    d.deposit_pct,
    d.vehicle_age_years,
    s.model_score,
    s.pd_estimate,
    c.default_flag

FROM
    dbo.customers c
    INNER JOIN dbo.deals d
        ON c.customer_id = d.customer_id
    LEFT JOIN dbo.model_scores s
        ON  c.customer_id = s.customer_id
        AND s.model_name  = 'PD_CUST_V1'

WHERE
    c.application_date BETWEEN :start_date AND :end_date