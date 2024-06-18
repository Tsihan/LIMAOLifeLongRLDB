SELECT MIN(mi1.info) AS release_date,
       MIN(t.title) AS modern_american_internet_movie
FROM aka_title AS at,
     company_name AS cn,
     company_type AS ct,
     info_type AS it1,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi1,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code = '[us]'
  AND it1.info = 'release dates'
  AND mi1.note LIKE '%internet%'
  AND mi1.info IS NOT NULL
  AND (mi1.info LIKE 'USA:% 199%'
       OR mi1.info LIKE 'USA:% 200%')
  AND t.production_year > 1990
  AND t.id = at.movie_id
  AND t.id = mi1.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi1.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = at.movie_id
  AND mi1.movie_id = mc.movie_id
  AND mi1.movie_id = at.movie_id
  AND mc.movie_id = at.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi1.info_type_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id;

