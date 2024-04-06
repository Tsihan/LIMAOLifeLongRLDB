SELECT 
    mi.info AS country,
    COUNT(DISTINCT t.id) AS num_movies,
    AVG(t.production_year) AS avg_production_year
FROM 
    movie_info AS mi,
    movie_keyword AS mk,
    title AS t
WHERE 
    
     mi.info IN ('Sweden',
                  'Norway',
                  'Germany',
                  'Denmark',
                  'Swedish',
                  'Denish',
                  'Norwegian',
                  'German',
                  'USA',
                  'American')
    AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND mk.movie_id = mi.movie_id
 
    
GROUP BY 
    mi.info
ORDER BY 
    num_movies DESC, 
    avg_production_year DESC;
