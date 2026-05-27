SELECT
    d.customer_id,
    d.ltv_ratio,
    d.loan_term_months,
    d.deposit_pct,
    d.vehicle_age_years,
    d.loan_amount,
    d.balloon_payment_flag

FROM
    dbo.deals d

WHERE
    d.application_date BETWEEN :start_date AND :end_date