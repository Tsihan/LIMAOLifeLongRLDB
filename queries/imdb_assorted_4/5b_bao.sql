SELECT MIN(t.title) AS american_vhs_movie
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info AS mi1,
     title AS t
WHERE ct.kind = 'production companies'
  AND mc.note LIKE '%(VHS)%'
  AND mc.note LIKE '%(USA)%'
  AND mc.note LIKE '%(1994)%'
  AND mi1.info IN ('USA',
                  'America')
  AND t.production_year > 2010
  AND t.id = mi1.movie_id
  AND t.id = mc.movie_id
  AND mc.movie_id = mi1.movie_id
  AND ct.id = mc.company_type_id
  AND it.id = mi1.info_type_id;

