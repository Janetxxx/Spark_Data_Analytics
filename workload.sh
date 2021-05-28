spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 3 \
    ass_task1.py \
    --output $1 