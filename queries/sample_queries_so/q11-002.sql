SELECT COUNT(*)
FROM
tag AS t1,
site AS s,
question AS q1,
tag_question AS tq1
WHERE
t1.site_id = s.site_id
AND q1.site_id = s.site_id
AND tq1.site_id = s.site_id
AND tq1.question_id = q1.id
AND tq1.tag_id = t1.id
AND (s.site_name IN ('softwareengineering'))
AND (t1.name IN ('architecture','design','design-patterns'))
AND (q1.score >= 0)
AND (q1.score <= 5)
