SELECT acc.location, count(*)
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
AND (s1.site_name in ('stats'))
AND (t1.name in ('model-selection','optimization','p-value','poisson-distribution'))
AND (q1.view_count >= 100)
AND (q1.view_count <= 100000)
AND (u1.downvotes >= 10)
AND (u1.downvotes <= 100000)
AND (b.name in ('Announcer','Custodian','Necromancer','Nice Question'))
GROUP BY acc.location
ORDER BY COUNT(*)
DESC
LIMIT 100
