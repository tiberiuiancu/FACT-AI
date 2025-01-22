echo '***********************DBLP_poisoned***********************'
echo '==========GCN-spmm=========='
python fairvgnn.py --dataset='dblp_poisoned' --encoder='GCN' --clip_e=0.1 --d_epochs=5 --g_epochs=5 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=0.5 --epochs=400 --prop='spmm'
