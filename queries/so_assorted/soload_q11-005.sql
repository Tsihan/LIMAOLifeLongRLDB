SELECT COUNT(*)
FROM
tag AS t1,
site AS s1,
question AS q1,
tag_question AS tq1
WHERE
t1.site_id = s1.site_id
AND q1.site_id = s1.site_id
AND tq1.site_id = s1.site_id
AND tq1.question_id = q1.id
AND tq1.tag_id = t1.id
AND (s1.site_name in ('cs'))
AND (t1.name in ('algorithm-analysis','complexity-theory','computability','data-structures','formal-languages','graphs'))
AND (q1.view_count >= 100)
AND (q1.view_count <= 100000)
