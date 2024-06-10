SELECT COUNT(DISTINCT acc.id)
FROM account AS acc, site AS s1, so_user AS u1, question AS q1, post_link AS pl, tag AS t, tag_question AS tq1
WHERE
NOT EXISTS (
    SELECT *
    FROM answer AS a1
    WHERE a1.site_id = q1.site_id AND a1.question_id = q1.id
) AND
s1.site_name = 'stackoverflow' AND
s1.site_id = q1.site_id AND
pl.site_id = q1.site_id AND
pl.post_id_to = q1.id AND
t.name = 'screenshot' AND
t.site_id = q1.site_id AND
q1.creation_date > '2013-01-01'::date AND
tq1.site_id = t.site_id AND
tq1.tag_id = t.id AND
tq1.question_id = q1.id AND
q1.owner_user_id = u1.id AND
q1.site_id = u1.site_id AND
u1.reputation > 59 AND
acc.id = u1.account_id AND
acc.website_url != '';
