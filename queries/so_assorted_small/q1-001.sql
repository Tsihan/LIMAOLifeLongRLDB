SELECT COUNT(*)
FROM tag AS t1, site AS s1, question AS q1, tag_question AS tq1
WHERE
s1.site_name = 'emacs' AND
t1.name = 'line-numbers' AND
t1.site_id = s1.site_id AND
q1.site_id = s1.site_id AND
tq1.site_id = s1.site_id AND
tq1.question_id = q1.id AND
tq1.tag_id = t1.id;
