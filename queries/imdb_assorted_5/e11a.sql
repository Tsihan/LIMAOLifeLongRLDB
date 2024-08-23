--- actors from a bunch of european countries in french movies
select count(*)
from info_type as it1,
info_type as it2,
title as t,
movie_info as mi,
cast_info as ci,
name as n,
person_info AS pi1
WHERE t.id = ci.movie_id
AND ci.person_id = n.id
AND n.id = pi1.person_id
AND it2.info ILIKE '%birth%'
AND pi1.info_type_id = it2.id
AND (pi1.info ILIKE '%uk%'
  OR pi1.info ILIKE '%spain%'
  OR pi1.info ILIKE '%germany%'
  OR pi1.info ILIKE '%italy%'
  OR pi1.info ILIKE '%croatia%'
  OR pi1.info ILIKE '%algeria%'
  OR pi1.info ILIKE '%south%'
  OR pi1.info ILIKE '%austria%'
  OR pi1.info ILIKE '%hungary%'
)
AND it1.info ILIKE '%count%'
AND mi.info_type_id = it1.id
AND t.id = mi.movie_id
AND mi.info ILIKE '%france%';
