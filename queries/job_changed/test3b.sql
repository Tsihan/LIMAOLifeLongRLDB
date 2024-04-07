SELECT MIN(t.title) AS movie_title, k.keyword, mk.movie_id
FROM keyword AS k,
     movie_info AS mi,
     movie_keyword AS mk,
     title AS t
WHERE k.keyword LIKE '%sequel%'
  AND mi.info = 'Bulgaria'
  AND t.production_year > 2010
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND mk.movie_id = mi.movie_id
  AND k.id = mk.keyword_id
GROUP BY mi.info, k.keyword, mk.movie_id
ORDER BY k.keyword DESC, mk.movie_id DESC;
