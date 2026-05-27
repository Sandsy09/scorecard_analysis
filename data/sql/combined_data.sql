SELECT
    c.customer_id,
    c.application_date,

    -- Customer variables (PD_cust inputs)
    c.annual_income,
    c.employment_status,
    c.credit_bureau_score,
    c.months_at_address,
    c.months_in_employment,
    c.num_credit_searches_6m,
    c.num_county_court_judgements,

    -- Deal variables (f(deal) inputs)
    d.ltv_ratio,
    d.loan_term_months,
    d.deposit_pct,
    d.vehicle_age_years,
    d.loan_amount,
    d.balloon_payment_flag,

    -- Target
    c.default_flag

FROM
    dbo.customers c
    INNER JOIN dbo.deals d
        ON c.customer_id = d.customer_id

WHERE
    c.application_date BETWEEN :start_date AND :end_date
    AND c.default_flag IS NOT NULL