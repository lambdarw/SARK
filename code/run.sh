source activate env

python /home/zhangshengxiang/project/KARM_zrw/train.py --pretrained_model_path=BAAI/bge-reranker-base \
--train_data_path=/data/train_informal_all_final.jsonl \
--valid_data_pathdata=/data/valid_informal_all_final.jsonl \
--test_data_path=/data/test_informal_all.jsonl \
--outdir=/output \
--tensorboard_log_dir=/output/log \
--cls_loss \
--rank_loss \
--learning_rate 1e-5 \
--batch_size 8 \
--epoch_num 100 \
--ttype=informal \
