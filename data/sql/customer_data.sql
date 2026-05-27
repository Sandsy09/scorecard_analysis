SELECT
    c.customer_id,
    c.annual_income,
    c.employment_status,
    c.credit_bureau_score,
    c.months_at_address,
    c.months_in_employment,
    c.num_credit_searches_6m,
    c.num_county_court_judgements,
    c.default_flag

FROM
    dbo.customers c

WHERE
    c.application_date BETWEEN :start_date AND :end_date
    AND c.default_flag IS NOT NULL