select COUNT(distinct acc.display_name)
from
tag AS t1, site AS s1, question AS q1, tag_question AS tq1, so_user AS u1, comment AS c1,
account AS acc
where
s1.site_name='drupal' and
t1.name in ('emails', 'composer', 'drush', 'theming', 'tokens', 'distributions', 'rating', 'navigation', 'path-aliases', 'entities') and
t1.site_id = s1.site_id and
q1.site_id = s1.site_id and
tq1.site_id = s1.site_id and
tq1.question_id = q1.id and
tq1.tag_id = t1.id and
q1.owner_user_id = u1.id and
q1.site_id = u1.site_id and
q1.score > 7 and
q1.view_count < 1940 and
c1.site_id = q1.site_id and
c1.post_id = q1.id and
acc.id = u1.account_id;
