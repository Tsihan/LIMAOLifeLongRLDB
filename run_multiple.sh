#!/bin/bash
source /mydata/anaconda3/bin/activate balsa

############################################
cd /mydata/LIMAOLifeLongRLDB
rm ./logfile
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_default_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_previous_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao.db
pg_ctl -D /mydata/databases restart -l logfile
cd /mydata/LIMAOLifeLongRLDB/bao_server
python ./main.py > /mydata/arm_selection_1.log 2>&1 &
MAIN_PID=$!
sleep 10 
python /mydata/LIMAOLifeLongRLDB/run_queries_assorted.py --seed 10 > /mydata/bao_assorted_1.log 2>&1
kill $MAIN_PID
wait $MAIN_PID 2>/dev/null
############################################
cd /mydata/LIMAOLifeLongRLDB
rm ./logfile
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_default_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_previous_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao.db
pg_ctl -D /mydata/databases restart -l logfile
cd /mydata/LIMAOLifeLongRLDB/bao_server
python ./main.py > /mydata/arm_selection_2.log 2>&1 &
MAIN_PID=$!
sleep 10
python /mydata/LIMAOLifeLongRLDB/run_queries_assorted.py --seed 20 > /mydata/bao_assorted_2.log 2>&1
kill $MAIN_PID
wait $MAIN_PID 2>/dev/null
############################################
cd /mydata/LIMAOLifeLongRLDB
rm ./logfile
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_default_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_previous_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao.db
pg_ctl -D /mydata/databases restart -l logfile
cd /mydata/LIMAOLifeLongRLDB/bao_server
python ./main.py > /mydata/arm_selection_3.log 2>&1 &
MAIN_PID=$!
sleep 10  
python /mydata/LIMAOLifeLongRLDB/run_queries_assorted.py --seed 30 > /mydata/bao_assorted_3.log 2>&1
kill $MAIN_PID
wait $MAIN_PID 2>/dev/null
############################################
cd /mydata/LIMAOLifeLongRLDB
rm ./logfile
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_default_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_previous_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao.db
pg_ctl -D /mydata/databases restart -l logfile
cd /mydata/LIMAOLifeLongRLDB/bao_server
python ./main.py > /mydata/arm_selection_4.log 2>&1 &
MAIN_PID=$!
sleep 10
python /mydata/LIMAOLifeLongRLDB/run_queries_assorted.py --seed 40 > /mydata/bao_assorted_4.log 2>&1
kill $MAIN_PID
wait $MAIN_PID 2>/dev/null
############################################
cd /mydata/LIMAOLifeLongRLDB
rm ./logfile
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_default_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_previous_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao.db
pg_ctl -D /mydata/databases restart -l logfile
cd /mydata/LIMAOLifeLongRLDB/bao_server
python ./main.py > /mydata/arm_selection_5.log 2>&1 &
MAIN_PID=$!
sleep 10  
python /mydata/LIMAOLifeLongRLDB/run_queries_assorted.py --seed 50 > /mydata/bao_assorted_5.log 2>&1
kill $MAIN_PID
wait $MAIN_PID 2>/dev/null
cd /mydata/LIMAOLifeLongRLDB
rm ./logfile
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_default_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao_previous_model
rm /mydata/LIMAOLifeLongRLDB/bao_server/bao.db
pg_ctl -D /mydata/databases restart -l logfile
