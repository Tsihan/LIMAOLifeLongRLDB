select count(distinct acc.display_name) from account AS acc, so_user AS u1, badge AS b1, badge AS b2 where
acc.website_url != '' and
acc.id = u1.account_id and
b1.site_id = u1.site_id and
b1.user_id = u1.id and
b1.name = 'Illuminator' and
b2.site_id = u1.site_id and
b2.user_id = u1.id and
b2.name = 'Nice Question' and
b2.date > b1.date + '9 months'::interval
