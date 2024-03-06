SELECT COUNT(DISTINCT acc.display_name)
FROM
account AS acc,
so_user AS su,
badge AS b1,
badge AS b2
WHERE
acc.website_url != '' AND
b1.name = 'Illuminator' AND
b2.name = 'Nice Question' AND
acc.id = su.account_id AND
b1.site_id = su.site_id AND
b1.user_id = su.id AND
b2.site_id = su.site_id AND
b2.user_id = su.id ;