SELECT COUNT(*)
FROM
site AS s1,
so_user AS u1,
question AS q1,
answer AS a1,
tag AS t1,
tag_question AS tq1,
badge AS b,
account AS acc
WHERE
s1.site_id = q1.site_id
AND s1.site_id = u1.site_id
AND s1.site_id = a1.site_id
AND s1.site_id = t1.site_id
AND s1.site_id = tq1.site_id
AND s1.site_id = b.site_id
AND q1.id = tq1.question_id
AND q1.id = a1.question_id
AND a1.owner_user_id = u1.id
AND t1.id = tq1.tag_id
AND b.user_id = u1.id
AND acc.id = u1.account_id
AND (s1.site_name in ('mathoverflow','ru'))
AND (t1.name in ('homotopy-theory','функции'))
AND (q1.view_count >= 10)
AND (q1.view_count <= 1000)
AND (u1.reputation >= 0)
AND (u1.reputation <= 100)
AND (b.name in ('Autobiographer','Scholar','Student'))
