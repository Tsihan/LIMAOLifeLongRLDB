select COUNT(distinct acc.display_name)
from
tag AS t1, site AS s1, question AS q1, answer AS a1, tag_question AS tq1, so_user AS u1,
tag AS t2, site AS s2, question AS q2, tag_question AS tq2, so_user AS u2,
account AS acc
where
s1.site_name='stackoverflow' and
t1.name  = 'audio' and
t1.site_id = s1.site_id and
q1.site_id = s1.site_id and
tq1.site_id = s1.site_id and
tq1.question_id = q1.id and
tq1.tag_id = t1.id and
a1.site_id = q1.site_id and
a1.question_id = q1.id and
a1.owner_user_id = u1.id and
a1.site_id = u1.site_id and
s2.site_name='superuser' and
t2.name  = 'networking' and
t2.site_id = s2.site_id and
q2.site_id = s2.site_id and
tq2.site_id = s2.site_id and
tq2.question_id = q2.id and
tq2.tag_id = t2.id and
q2.owner_user_id = u2.id and
q2.site_id = u2.site_id and
u1.account_id = u2.account_id and
acc.id = u1.account_id;
