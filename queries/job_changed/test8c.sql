SELECT MIN(an1.name) AS writer_pseudo_name,
       MIN(t.title) AS movie_title
FROM aka_name AS an1,
     cast_info AS ci,
     company_name AS cn,
     movie_companies AS mc,
     name AS n1,
     role_type AS rt,
     title AS t
WHERE cn.country_code ='[us]'
  AND rt.role ='writer'
  AND n1.id = ci.person_id
  AND ci.movie_id = t.id
  AND mc.company_id = cn.id
  AND ci.role_id = rt.id
  AND an1.person_id = ci.person_id
  AND ci.movie_id = mc.movie_id
GROUP BY n1.id
ORDER BY actress_pseudonym;
