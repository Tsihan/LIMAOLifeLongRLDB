SELECT
    MIN(chn.name) AS uncredited_voiced_character,
    MIN(t.title)
FROM
    char_name AS chn,
    cast_info AS ci,
    movie_companies AS mc,
    role_type AS rt,
    title AS t
WHERE
    ci.note not like '%(voice)%'
    AND ci.note like '%(uncredited)%'
    AND t.production_year > 2010
    AND t.id = mc.movie_id
    AND t.id = ci.movie_id
    AND ci.movie_id = mc.movie_id
    AND chn.id = ci.person_role_id;