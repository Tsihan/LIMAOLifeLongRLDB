SELECT MIN(mi1.info) AS movie_budget,
       MIN(mi_idx.info) AS movie_votes,
       MIN(t.title) AS movie_title
FROM cast_info AS ci,
     info_type AS it1,
     info_type AS it2,
     movie_info AS mi1,
     movie_info_idx AS mi_idx,
     name AS n,
     title AS t
WHERE ci.note IN ('(writer)',
                  '(head writer)',
                  '(written by)',
                  '(story)',
                  '(story editor)')
  AND it1.info = 'genres'
  AND it2.info = 'votes'
  AND mi1.info IN ('Horror',
                  'Action',
                  'Sci-Fi',
                  'Thriller',
                  'Crime',
                  'War')
  AND n.gender = 'm'
  AND t.id = mi1.movie_id
  AND t.id = mi_idx.movie_id
  AND t.id = ci.movie_id
  AND ci.movie_id = mi1.movie_id
  AND ci.movie_id = mi_idx.movie_id
  AND mi1.movie_id = mi_idx.movie_id
  AND n.id = ci.person_id
  AND it1.id = mi1.info_type_id
  AND it2.id = mi_idx.info_type_id;

