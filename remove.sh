rm /mydata/LIMAOLifeLongRLDB/bao_server/bao.db
rm -rf /mydata/LIMAOLifeLongRLDB/bao_server/bao_default_model
rm -rf /mydata/LIMAOLifeLongRLDB/bao_server/bao_previous_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/current_progress.cfg
rm /mydata/logfile
cd /mydata
pg_ctl -D /mydata/databases restart -l logfile