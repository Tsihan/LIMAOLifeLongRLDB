SELECT COUNT(DISTINCT acc.id) 
FROM
account AS acc, 
site AS s, 
so_user AS su, 
question AS q, 
post_link AS pl, 
tag AS t, 
tag_question AS tq 
WHERE
s.site_name = 'stackoverflow' AND
t.name = 'firebase-realtime-database' AND
q.creation_date > '2011-01-01'::date AND
su.reputation > 146 AND
acc.website_url != '' AND
s.site_id = q.site_id AND
pl.site_id = q.site_id AND
pl.post_id_to = q.id AND
t.site_id = q.site_id AND
tq.site_id = t.site_id AND
tq.tag_id = t.id AND
tq.question_id = q.id AND
q.owner_user_id = su.id AND
q.site_id = su.site_id AND
acc.id = su.account_id ;