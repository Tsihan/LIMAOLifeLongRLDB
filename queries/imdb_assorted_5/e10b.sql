select
  min(n.name),
  min(t.title)
from
  info_type as it1,
  info_type as it2,
  movie_info_idx AS mi_idx,
  title as t,
  cast_info as ci,
  name as n,
  person_info AS pi1
WHERE
  it1.info LIKE 'rating'
  AND it1.id = mi_idx.info_type_id
  AND t.id = mi_idx.movie_id
  AND t.id = ci.movie_id
  AND ci.person_id = n.id
  AND n.id = pi1.person_id
  AND pi1.info_type_id = it2.id
  AND it2.info LIKE '%birth%'
  AND pi1.info LIKE '%USA%'
  AND (
    mi_idx.info LIKE '0%'
    OR mi_idx.info LIKE '1%'
    OR mi_idx.info LIKE '2%'
  );