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
keyword as k
WHERE
t.id = mi1.movie_id
AND t.id = mi_idx1.movie_id
AND t.id = mi_idx2.movie_id
AND t.id = mk.movie_id
AND mi_idx2.movie_id = mi_idx1.movie_id
AND mi1.movie_id = mi_idx1.movie_id
AND mk.movie_id = mi1.movie_id
AND mk.keyword_id = k.id
AND mi1.info_type_id = it1.id
AND mi_idx1.info_type_id = it3.id
AND mi_idx2.info_type_id = it4.id
AND t.kind_id = kt.id
AND (kt.kind IN ('episode','movie'))
AND (t.production_year <= 2020)
AND (t.production_year >= 1925)
AND (mi1.info IN ('PCS:Spherical','PFM:35 mm','RAT:1.33 : 1','RAT:1.37 : 1'))
AND (it1.id IN ('1','16','7'))
AND it3.id = '100'
AND it4.id = '101'
AND (mi_idx2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mi_idx2.info::float <= 7.0)
AND (mi_idx2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 3.0 <= mi_idx2.info::float)
AND (mi_idx1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0 <= mi_idx1.info::float)
AND (mi_idx1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mi_idx1.info::float <= 1000.0)
