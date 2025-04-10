select count(distinct acc.id) from
account AS acc, site AS s1, so_user AS u1, question AS q1, post_link AS pl, tag AS t1, tag_question AS tq1 where
not exists (select * from answer AS a where a.site_id = q1.site_id and a.question_id = q1.id) and
s1.site_name = 'stackoverflow' and
s1.site_id = q1.site_id and
pl.site_id = q1.site_id and
pl.post_id_to = q1.id and
t1.name = 'tensorflow' and
t1.site_id = q1.site_id and
q1.creation_date > '2018-01-01'::date and
tq1.site_id = t1.site_id and
tq1.tag_id = t1.id and
tq1.question_id = q1.id and
q1.owner_user_id = u1.id and
q1.site_id = u1.site_id and
u1.reputation > 113 and
acc.id = u1.account_id and
acc.website_url != '';
