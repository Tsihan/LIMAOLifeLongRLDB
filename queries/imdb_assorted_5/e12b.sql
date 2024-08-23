-- sorted list of top actors from france in given genres
select n.name
from title as t,
name as n,
cast_info as ci,
movie_info as mi,
info_type as it1,
info_type as it2,
person_info AS pi1
WHERE t.id = ci.movie_id
AND t.id = mi.movie_id
AND ci.person_id = n.id
AND it1.id = 3
AND it1.id = mi.info_type_id
AND (mi.info ILIKE '%adult%')
AND it2.info ILIKE '%birth%'
AND pi1.person_id = n.id
AND pi1.info_type_id = it2.id
AND pi1.info ILIKE '%japan%'
group by n.name
order by count(*) DESC
