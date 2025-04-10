select count(distinct q1.id) from
site AS s1, post_link AS pl, question AS q1, question AS q2, comment AS c1, comment AS c2,
tag AS t1, tag_question AS tq1, tag_question AS tq2
where
s1.site_name = 'german' and
pl.site_id = s1.site_id and
pl.site_id = q1.site_id and
pl.post_id_from = q1.id and
pl.site_id = q2.site_id and
pl.post_id_to = q2.id and
c1.site_id = q1.site_id and
c1.post_id = q1.id and
c2.site_id = q2.site_id and
c2.post_id = q2.id and
c1.date < c2.date and
t1.name in ('sql-server', 'c#', 'asp.net', 'objective-c') and
t1.id = tq1.tag_id and
t1.site_id = tq1.site_id and
t1.id = tq2.tag_id and
t1.site_id = tq1.site_id and
t1.site_id = pl.site_id and
tq1.site_id = q1.site_id and
tq1.question_id = q1.id and
tq2.site_id = q2.site_id and
tq2.question_id = q2.id;
