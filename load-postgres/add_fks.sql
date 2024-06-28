ALTER TABLE title ADD CONSTRAINT fk_title_kind_id FOREIGN KEY (kind_id) REFERENCES kind_type(id) ON DELETE CASCADE;

ALTER TABLE aka_name ADD CONSTRAINT fk_aka_name_id FOREIGN KEY (id) REFERENCES name(id) ON DELETE CASCADE;

ALTER TABLE cast_info ADD CONSTRAINT fk_cast_info_movie_id FOREIGN KEY (movie_id) REFERENCES title(id) ON DELETE CASCADE;
ALTER TABLE cast_info ADD CONSTRAINT fk_cast_info_person_role_id FOREIGN KEY (person_role_id) REFERENCES char_name(id) ON DELETE CASCADE;
ALTER TABLE cast_info ADD CONSTRAINT fk_cast_info_role_id FOREIGN KEY (role_id) REFERENCES role_type(id) ON DELETE CASCADE;

ALTER TABLE complete_cast ADD CONSTRAINT fk_complete_cast_movie_id FOREIGN KEY (movie_id) REFERENCES title(id) ON DELETE CASCADE;
ALTER TABLE complete_cast ADD CONSTRAINT fk_complete_cast_subject_id FOREIGN KEY (subject_id) REFERENCES comp_cast_type(id) ON DELETE CASCADE;
ALTER TABLE complete_cast ADD CONSTRAINT fk_complete_cast_status_id FOREIGN KEY (status_id) REFERENCES comp_cast_type(id) ON DELETE CASCADE;

ALTER TABLE movie_companies ADD CONSTRAINT fk_movie_companies_movie_id FOREIGN KEY (movie_id) REFERENCES title(id) ON DELETE CASCADE;

ALTER TABLE movie_info ADD CONSTRAINT fk_movie_info_movie_id FOREIGN KEY (movie_id) REFERENCES title(id) ON DELETE CASCADE;
ALTER TABLE movie_info ADD CONSTRAINT fk_movie_info_info_type_id FOREIGN KEY (info_type_id) REFERENCES info_type(id) ON DELETE CASCADE;

ALTER TABLE movie_info_idx ADD CONSTRAINT fk_movie_info_idx_movie_id FOREIGN KEY (movie_id) REFERENCES title(id) ON DELETE CASCADE;
ALTER TABLE movie_info_idx ADD CONSTRAINT fk_movie_info_idx_info_type_id FOREIGN KEY (info_type_id) REFERENCES info_type(id) ON DELETE CASCADE;

ALTER TABLE movie_keyword ADD CONSTRAINT fk_movie_keyword_movie_id FOREIGN KEY (movie_id) REFERENCES title(id) ON DELETE CASCADE;
ALTER TABLE movie_keyword ADD CONSTRAINT fk_movie_keyword_keyword_id FOREIGN KEY (keyword_id) REFERENCES keyword(id) ON DELETE CASCADE;

ALTER TABLE movie_link ADD CONSTRAINT fk_movie_link_movie_id FOREIGN KEY (movie_id) REFERENCES title(id) ON DELETE CASCADE;
ALTER TABLE movie_link ADD CONSTRAINT fk_movie_link_link_type_id FOREIGN KEY (link_type_id) REFERENCES link_type(id) ON DELETE CASCADE;

ALTER TABLE person_info ADD CONSTRAINT fk_person_info_person_id FOREIGN KEY (person_id) REFERENCES name(id) ON DELETE CASCADE;
ALTER TABLE person_info ADD CONSTRAINT fk_person_info_info_type_id FOREIGN KEY (info_type_id) REFERENCES info_type(id) ON DELETE CASCADE;
