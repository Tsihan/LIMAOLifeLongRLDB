COPY nation FROM '/home/qihan/balsaLifeLongRLDB/TPC-HV3.0.1/dbgen/tbl/nation.tbl' WITH DELIMITER AS '|' NULL '';
COPY region FROM '/home/qihan/balsaLifeLongRLDB/TPC-HV3.0.1/dbgen/tbl/region.tbl' WITH DELIMITER AS '|' NULL '';
COPY part FROM '/home/qihan/balsaLifeLongRLDB/TPC-HV3.0.1/dbgen/tbl/part.tbl' WITH DELIMITER AS '|' NULL '';
COPY supplier FROM '/home/qihan/balsaLifeLongRLDB/TPC-HV3.0.1/dbgen/tbl/supplier.tbl' WITH DELIMITER AS '|' NULL '';
COPY partsupp FROM '/home/qihan/balsaLifeLongRLDB/TPC-HV3.0.1/dbgen/tbl/partsupp.tbl' WITH DELIMITER AS '|' NULL '';
COPY customer FROM '/home/qihan/balsaLifeLongRLDB/TPC-HV3.0.1/dbgen/tbl/customer.tbl' WITH DELIMITER AS '|' NULL '';
COPY orders FROM '/home/qihan/balsaLifeLongRLDB/TPC-HV3.0.1/dbgen/tbl/orders.tbl' WITH DELIMITER AS '|' NULL '';
COPY lineitem FROM '/home/qihan/balsaLifeLongRLDB/TPC-HV3.0.1/dbgen/tbl/lineitem.tbl' WITH DELIMITER AS '|' NULL '';

