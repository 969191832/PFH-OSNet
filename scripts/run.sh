# export CUDA_VISIBLE_DEVICES=0 
tmp=scratch.yaml
python ../main.py \
	--config-file configs/$tmp \
	--root /data2/zsl/dataset
