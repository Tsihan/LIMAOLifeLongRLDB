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
AND (s1.site_name in ('stackoverflow'))
AND (t1.name in ('api','audio','events','file-io','forms','functional-programming','java-ee','serialization','sqlite','ubuntu','unicode'))
AND (q1.favorite_count >= 5)
AND (q1.favorite_count <= 5000)
