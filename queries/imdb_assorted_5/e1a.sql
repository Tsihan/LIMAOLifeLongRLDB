SELECT
    min(t.title),
    min(t.production_year),
    min(chn.name)
FROM
    cast_info AS ci,
    title AS t,
    char_name AS chn
WHERE
    ci.movie_id = t.id
    AND chn.id = ci.person_role_id
    and t.kind_id = 1
    AND ci.role_id = 2;