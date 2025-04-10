select count(*) from tag AS t1, site AS s1, question AS q1, tag_question AS tq1
where
s1.site_name='diy' and
t1.name='rivets' and
t1.site_id = s1.site_id and
q1.site_id = s1.site_id and
tq1.site_id = s1.site_id and
tq1.question_id = q1.id and
tq1.tag_id = t1.id
