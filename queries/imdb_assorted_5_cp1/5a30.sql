SELECT COUNT(*)
FROM title as t,
movie_info as mi1,
kind_type as kt,
info_type as it1,
info_type as it3,
info_type as it4,
movie_info_idx as miidx,
movie_info_idx as mii2,
movie_keyword as mk,
keyword as k
WHERE
t.id = mi1.movie_id
AND t.id = miidx.movie_id
AND t.id = mii2.movie_id
AND t.id = mk.movie_id
AND mii2.movie_id = miidx.movie_id
AND mi1.movie_id = miidx.movie_id
AND mk.movie_id = mi1.movie_id
AND mk.keyword_id = k.id
AND mi1.info_type_id = it1.id
AND miidx.info_type_id = it3.id
AND mii2.info_type_id = it4.id
AND t.kind_id = kt.id
AND (kt.kind IN ('episode','movie'))
AND (t.production_year <= 2015)
AND (t.production_year >= 1990)
AND (mi1.info IN ('OFM:16 mm','OFM:Video','PCS:Spherical','RAT:16:9 HD','RAT:2.35 : 1'))
AND (it1.id IN ('16','7'))
AND it3.id = '100'
AND it4.id = '101'
AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii2.info::float <= 11.0)
AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 7.0 <= mii2.info::float)
AND (miidx.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 500.0 <= miidx.info::float)
AND (miidx.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND miidx.info::float <= 7200.0)
