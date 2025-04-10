SELECT COUNT(*)
FROM
site AS s1,
so_user AS u1,
tag AS t1,
tag_question AS tq1,
question AS q1,
badge AS b1,
account AS acc
WHERE
s1.site_id = u1.site_id
AND s1.site_id = b1.site_id
AND s1.site_id = t1.site_id
AND s1.site_id = tq1.site_id
AND s1.site_id = q1.site_id
AND t1.id = tq1.tag_id
AND q1.id = tq1.question_id
AND q1.owner_user_id = u1.id
AND acc.id = u1.account_id
AND b1.user_id = u1.id
AND (q1.score >= 10)
AND (q1.score <= 1000)
AND (s1.site_name = 'stackoverflow')
AND (t1.name in ('apple-push-notifications','argparse','automation','charts','clickonce','deprecated','git-merge','lodash','new-operator','project-management'))
AND (acc.website_url ILIKE ('%de'))
