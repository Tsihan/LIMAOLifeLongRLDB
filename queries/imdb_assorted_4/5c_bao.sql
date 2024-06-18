SELECT MIN(t.title) AS american_movie
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info AS mi1,
     title AS t
WHERE ct.kind = 'production companies'
  AND mc.note NOT LIKE '%(TV)%'
  AND mc.note LIKE '%(USA)%'
  AND mi1.info IN ('Sweden',
                  'Norway',
                  'Germany',
                  'Denmark',
                  'Swedish',
                  'Denish',
                  'Norwegian',
                  'German',
                  'USA',
                  'American')
  AND t.production_year > 1990
  AND t.id = mi1.movie_id
  AND t.id = mc.movie_id
  AND mc.movie_id = mi1.movie_id
  AND ct.id = mc.company_type_id
  AND it.id = mi1.info_type_id;

