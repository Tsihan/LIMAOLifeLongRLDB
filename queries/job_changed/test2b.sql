SELECT 
    MIN(t.title) AS movie_title,
    COUNT(DISTINCT mc.movie_id) AS num_movies,
    AVG(t.production_year) AS avg_production_year
FROM 
    company_name AS cn,
    keyword AS k,
    movie_companies AS mc,
    movie_keyword AS mk,
    title AS t
WHERE 
    cn.country_code = '[nl]'
    AND k.keyword = 'character-name-in-title'
    AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND t.id = mk.movie_id
  AND mk.keyword_id = k.id
  AND mc.movie_id = mk.movie_id
GROUP BY 
    cn.country_code, 
    k.keyword,
    t.production_year
ORDER BY
    num_movies DESC,
    avg_production_year DESC;
