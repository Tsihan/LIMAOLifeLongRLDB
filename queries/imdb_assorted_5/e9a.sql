SELECT
    min(t.title),
    min(pi1.info)
FROM
    person_info AS pi1,
    info_type as it1,
    info_type as it2,
    name as n,
    cast_info as ci,
    title as t,
    movie_info as mi
WHERE
    t.id = mi.movie_id
    AND it2.id = 3
    AND mi.info_type_id = it2.id
    AND mi.info LIKE '%sci%'
    AND t.id = ci.movie_id
    AND ci.person_id = n.id
    AND n.id = pi1.person_id
    AND it1.info LIKE 'birth date'
    AND pi1.info_type_id = it1.id
    AND pi1.info LIKE '%199%';