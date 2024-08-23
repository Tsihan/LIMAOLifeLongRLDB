--- top rated movies of indian actors
select min(n.name), min(t.title)
from info_type as it1,
info_type as it2,
movie_info_idx AS miidx,
title as t,
cast_info as ci,
name as n,
person_info AS pi1
WHERE it1.info ILIKE 'rating'
AND it1.id = miidx.info_type_id
AND t.id = miidx.movie_id
AND t.id = ci.movie_id
AND ci.person_id = n.id
AND n.id = pi1.person_id
AND pi1.info_type_id = it2.id
AND it2.info ILIKE '%birth%'
AND pi1.info ILIKE '%india%'
AND (miidx.info ILIKE '8%' OR miidx.info ILIKE '9%'
  OR miidx.info ILIKE '10%');


