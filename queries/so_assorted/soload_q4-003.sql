select COUNT(distinct acc.display_name)
from
tag AS t1, site AS s1, question AS q1, answer AS a1, tag_question AS tq1, so_user AS u1,
account AS acc
where
s1.site_name='superuser' and
t1.name = 'command-line' and
t1.site_id = s1.site_id and
q1.site_id = s1.site_id and
tq1.site_id = s1.site_id and
tq1.question_id = q1.id and
tq1.tag_id = t1.id and
a1.site_id = q1.site_id and
a1.question_id = q1.id and
a1.owner_user_id = u1.id and
a1.site_id = u1.site_id and
a1.creation_date >= q1.creation_date + '1 year'::interval and
acc.id = u1.account_id;
