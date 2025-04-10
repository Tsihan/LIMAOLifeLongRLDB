select count(distinct q1.id) from
site AS s1, post_link AS pl1, post_link AS pl2, question AS q1, question AS q2, question AS q3 where
s1.site_name = 'cseducators' and
q1.site_id = s1.site_id and
q1.site_id = q2.site_id and
q2.site_id = q3.site_id and
pl1.site_id = q1.site_id and
pl1.post_id_from = q1.id and
pl1.post_id_to = q2.id and
pl2.site_id = q1.site_id and
pl2.post_id_from = q2.id and
pl2.post_id_to = q3.id and
exists ( select * from comment AS c1 where c1.site_id = q3.site_id and c1.post_id = q3.id ) and
exists ( select * from comment AS c2 where c2.site_id = q2.site_id and c2.post_id = q2.id ) and
exists ( select * from comment AS c3 where c3.site_id = q1.site_id and c3.post_id = q1.id ) and
q1.score > q3.score;
