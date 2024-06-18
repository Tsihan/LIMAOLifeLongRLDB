SELECT COUNT(*)
FROM title as t,
movie_info as mi1,
kind_type as kt,
info_type as it1,
info_type as it3,
info_type as it4,
movie_info_idx as mi_idx1,
movie_info_idx as mi_idx2,
movie_keyword as mk,
keyword as k,
aka_name as an,
name as n,
info_type as it5,
person_info as pi1,
cast_info as ci,
role_type as rt
WHERE
t.id = mi1.movie_id
AND t.id = ci.movie_id
AND t.id = mi_idx1.movie_id
AND t.id = mi_idx2.movie_id
AND t.id = mk.movie_id
AND mk.keyword_id = k.id
AND mi1.info_type_id = it1.id
AND mi_idx1.info_type_id = it3.id
AND mi_idx2.info_type_id = it4.id
AND t.kind_id = kt.id
AND (kt.kind IN ('episode'))
AND (t.production_year <= 1975)
AND (t.production_year >= 1875)
AND (mi1.info IN ('Color','OFM:Live','OFM:Video','PFM:Video'))
AND (it1.id IN ('103','2','7'))
AND it3.id = '100'
AND it4.id = '101'
AND (mi_idx2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mi_idx2.info::float <= 11.0)
AND (mi_idx2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 7.0 <= mi_idx2.info::float)
AND (mi_idx1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0 <= mi_idx1.info::float)
AND (mi_idx1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mi_idx1.info::float <= 10000.0)
AND n.id = ci.person_id
AND ci.person_id = pi1.person_id
AND it5.id = pi1.info_type_id
AND n.id = pi1.person_id
AND n.id = an.person_id
AND rt.id = ci.role_id
AND (n.gender in ('m'))
AND (n.name_pcode_nf in ('C6231','F6362','F6525','J513','R1631','R1632','R1636','R2631','S2153'))
AND (ci.note IS NULL)
AND (rt.role in ('actor'))
AND (it5.id in ('25'))
