.PHONY: setup combine_csv all nifa defense hyperparam-alpha hyperparam-beta hyperparam-perturbation hyperparam-k hyperparam-d hyperparameters

DEVICE ?= 0

OUT_DIR ?= output

setup:
	@mkdir -p $(OUT_DIR)

all: nifa defense hyperparameters

combine_csv:
	@awk 'NR == 1 {print; next} FNR > 1' $(ARGS) > $(OUTPUT_CSV)
	@if [ "$(REMOVE_CSV)" = "1" ] || [ "$(REMOVE_CSV)" = "true" ]; then \
		rm -f $(ARGS); \
	fi

# Runs the main experiment of the paper (results in Table 2)
# The results are saved in the output folder at nifa.csv
# todo: re-add FairGNN, FairVGNN, FairSIN
nifa: setup
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --output_path $(OUT_DIR)/nifa_pokec_z.csv --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC' 'GAT' #'FairGNN' 'FairVGNN' 'FairSIN'

	@python code/main.py --seed 42 --n_times 5 --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --before --device $(DEVICE) --output_path $(OUT_DIR)/nifa_pokec_n.csv --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC' 'GAT'  #'FairGNN' 'FairVGNN' 'FairSIN'

	@python code/main.py --seed 42 --n_times 5 --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --epochs 500 --before --device $(DEVICE) --output_path $(OUT_DIR)/nifa_dblp.csv --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC' 'GAT'  #'FairGNN' 'FairVGNN' 'FairSIN'

	$(MAKE) combine_csv ARGS='$(OUT_DIR)/nifa_pokec_z.csv $(OUT_DIR)/nifa_pokec_n.csv $(OUT_DIR)/nifa_dblp.csv' OUTPUT_CSV=$(OUT_DIR)/nifa.csv REMOVE_CSV=1


defense: setup
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.1 --output_path $(OUT_DIR)/defense_1.csv

	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.2 --output_path $(OUT_DIR)/defense_2.csv

	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.3 --output_path $(OUT_DIR)/defense_3.csv

	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.4 --output_path $(OUT_DIR)/defense_4.csv

	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.5 --output_path $(OUT_DIR)/defense_5.csv

	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.6 --output_path $(OUT_DIR)/defense_6.csv

	$(MAKE) combine_csv ARGS='$(OUT_DIR)/defense_1.csv $(OUT_DIR)/defense_2.csv $(OUT_DIR)/defense_3.csv $(OUT_DIR)/defense_4.csv $(OUT_DIR)/defense_5.csv $(OUT_DIR)/defense_6.csv' OUTPUT_CSV=$(OUT_DIR)/defense.csv REMOVE_CSV=1


# analysis on alpha: 0.005, 0.01, 0.02, 0.05, 0.1, 0.2
# produces Figure A2
hyperparam-alpha: setup
	# pokec_z
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.005 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_1.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_2.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.02 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_3.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.05 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_4.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.1 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_5.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.2 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_6.csv
	# pokec_n
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.005 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_7.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_8.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.02 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_9.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.05 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_10.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.1 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_11.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.2 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_12.csv
	# dblp
	@python code/main.py --seed 42 --dataset dblp --alpha 0.005 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_13.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_14.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.02 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_15.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.05 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_16.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_17.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.2 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_alpha_18.csv
	# combine
	$(MAKE) combine_csv ARGS='$(OUT_DIR)/hyperparameters_alpha_1.csv $(OUT_DIR)/hyperparameters_alpha_2.csv $(OUT_DIR)/hyperparameters_alpha_3.csv $(OUT_DIR)/hyperparameters_alpha_4.csv $(OUT_DIR)/hyperparameters_alpha_5.csv $(OUT_DIR)/hyperparameters_alpha_6.csv $(OUT_DIR)/hyperparameters_alpha_7.csv $(OUT_DIR)/hyperparameters_alpha_8.csv $(OUT_DIR)/hyperparameters_alpha_9.csv $(OUT_DIR)/hyperparameters_alpha_10.csv $(OUT_DIR)/hyperparameters_alpha_11.csv $(OUT_DIR)/hyperparameters_alpha_12.csv $(OUT_DIR)/hyperparameters_alpha_13.csv $(OUT_DIR)/hyperparameters_alpha_14.csv $(OUT_DIR)/hyperparameters_alpha_15.csv $(OUT_DIR)/hyperparameters_alpha_16.csv $(OUT_DIR)/hyperparameters_alpha_17.csv $(OUT_DIR)/hyperparameters_alpha_18.csv' OUTPUT_CSV=$(OUT_DIR)/hyperparameters_alpha.csv REMOVE_CSV=1


# analysis on beta: 2, 4, 8, 16
# produces Figure A3
hyperparam-beta: setup
	# pokec_z
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 2 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_1.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_2.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 8 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_3.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 16 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_4.csv
	# pokec_n
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 2 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_5.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_6.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 8 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_7.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 16 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_8.csv
	# dblp
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 2 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_9.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 4 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_10.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_11.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 16 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_beta_12.csv
	# combine
	$(MAKE) combine_csv ARGS='$(OUT_DIR)/hyperparameters_beta_1.csv $(OUT_DIR)/hyperparameters_beta_2.csv $(OUT_DIR)/hyperparameters_beta_3.csv $(OUT_DIR)/hyperparameters_beta_4.csv $(OUT_DIR)/hyperparameters_beta_5.csv $(OUT_DIR)/hyperparameters_beta_6.csv $(OUT_DIR)/hyperparameters_beta_7.csv $(OUT_DIR)/hyperparameters_beta_8.csv $(OUT_DIR)/hyperparameters_beta_9.csv $(OUT_DIR)/hyperparameters_beta_10.csv $(OUT_DIR)/hyperparameters_beta_11.csv $(OUT_DIR)/hyperparameters_beta_12.csv' OUTPUT_CSV=$(OUT_DIR)/hyperparameters_beta.csv REMOVE_CSV=1


# analysis on perturbation rate: 0.005, 0.01, 0.02, 0.03
# produces Figure A4
hyperparam-perturbation: setup
	# pokec_z
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 51 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_1.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_2.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 204 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_3.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --node 306 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_4.csv
	# pokec_n
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 4 --node 43 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_5.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_6.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 4 --node 174 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_7.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 4 --node 261 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_8.csv
	# dblp
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 4 --node 16 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_9.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 4 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_10.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 4 --node 64 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_11.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 4 --node 96 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_perturbation_12.csv
	# combine
	$(MAKE) combine_csv ARGS='$(OUT_DIR)/hyperparameters_perturbation_1.csv $(OUT_DIR)/hyperparameters_perturbation_2.csv $(OUT_DIR)/hyperparameters_perturbation_3.csv $(OUT_DIR)/hyperparameters_perturbation_4.csv $(OUT_DIR)/hyperparameters_perturbation_5.csv $(OUT_DIR)/hyperparameters_perturbation_6.csv $(OUT_DIR)/hyperparameters_perturbation_7.csv $(OUT_DIR)/hyperparameters_perturbation_8.csv $(OUT_DIR)/hyperparameters_perturbation_9.csv $(OUT_DIR)/hyperparameters_perturbation_10.csv $(OUT_DIR)/hyperparameters_perturbation_11.csv $(OUT_DIR)/hyperparameters_perturbation_12.csv' OUTPUT_CSV=$(OUT_DIR)/hyperparameters_perturbation.csv REMOVE_CSV=1


# analysis on k (ratio): 0.1, 0.25, 0.5, 0.75
# produces Figure A5
hyperparam-k:
	#pokec_z
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --ratio 0.1 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_1.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --ratio 0.25 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_2.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --ratio 0.5 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_3.csv
	@python code/main.py --seed 42 --dataset pokec_z --alpha 0.01 --beta 4 --ratio 0.75 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_4.csv
	#pokec_n
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 4 --ratio 0.1 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_5.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 4 --ratio 0.25 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_6.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 4 --ratio 0.5 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_7.csv
	@python code/main.py --seed 42 --dataset pokec_n --alpha 0.01 --beta 4 --ratio 0.75 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_8.csv
	#dblp
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 4 --ratio 0.1 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_9.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 4 --ratio 0.25 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_10.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 4 --ratio 0.5 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_11.csv
	@python code/main.py --seed 42 --dataset dblp --alpha 0.1 --beta 4 --ratio 0.75 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_k_12.csv
	# combine
	$(MAKE) combine_csv ARGS='$(OUT_DIR)/hyperparameters_k_1.csv $(OUT_DIR)/hyperparameters_k_2.csv $(OUT_DIR)/hyperparameters_k_3.csv $(OUT_DIR)/hyperparameters_k_4.csv $(OUT_DIR)/hyperparameters_k_5.csv $(OUT_DIR)/hyperparameters_k_6.csv $(OUT_DIR)/hyperparameters_k_7.csv $(OUT_DIR)/hyperparameters_k_8.csv $(OUT_DIR)/hyperparameters_k_9.csv $(OUT_DIR)/hyperparameters_k_10.csv $(OUT_DIR)/hyperparameters_k_11.csv $(OUT_DIR)/hyperparameters_k_12.csv' OUTPUT_CSV=$(OUT_DIR)/hyperparameters_k.csv REMOVE_CSV=1


hyperparam-d: setup
	#pokec_z
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 10 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_1.csv
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 25 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_2.csv
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_3.csv
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 100 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_4.csv
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 150 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_5.csv
	#pokec_n
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 10 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_6.csv
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 25 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_7.csv
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_8.csv
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 100 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_9.csv
	@python code/main.py --seed 42 --n_times 5 --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 150 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_10.csv
	#dblp
	@python code/main.py --seed 42 --n_times 5 --dataset dblp --alpha 0.1 --beta 4 --node 32 --edge 6 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_11.csv
	@python code/main.py --seed 42 --n_times 5 --dataset dblp --alpha 0.1 --beta 4 --node 32 --edge 12 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_12.csv
	@python code/main.py --seed 42 --n_times 5 --dataset dblp --alpha 0.1 --beta 4 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_13.csv
	@python code/main.py --seed 42 --n_times 5 --dataset dblp --alpha 0.1 --beta 4 --node 32 --edge 48 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_14.csv
	@python code/main.py --seed 42 --n_times 5 --dataset dblp --alpha 0.1 --beta 4 --node 32 --edge 72 --device $(DEVICE) --models 'GCN' --output_path $(OUT_DIR)/hyperparam_d_15.csv
	#combine
	$(MAKE) combine_csv ARGS='$(OUT_DIR)/hyperparam_d_1.csv $(OUT_DIR)/hyperparam_d_2.csv $(OUT_DIR)/hyperparam_d_3.csv $(OUT_DIR)/hyperparam_d_4.csv $(OUT_DIR)/hyperparam_d_5.csv $(OUT_DIR)/hyperparam_d_6.csv $(OUT_DIR)/hyperparam_d_7.csv $(OUT_DIR)/hyperparam_d_8.csv $(OUT_DIR)/hyperparam_d_9.csv $(OUT_DIR)/hyperparam_d_10.csv $(OUT_DIR)/hyperparam_d_11.csv $(OUT_DIR)/hyperparam_d_12.csv $(OUT_DIR)/hyperparam_d_13.csv $(OUT_DIR)/hyperparam_d_14.csv $(OUT_DIR)/hyperparam_d_15.csv' OUTPUT_CSV=$(OUT_DIR)/hyperparameters_d.csv REMOVE_CSV=1


hyperparameters: hyperparam-alpha hyperparam-beta hyperparam-perturbation hyperparam-k hyperparam-d
	$(MAKE) combine_csv ARGS='$(OUT_DIR)/hyperparameters_alpha.csv $(OUT_DIR)/hyperparameters_beta.csv $(OUT_DIR)/hyperparameters_perturbation.csv $(OUT_DIR)/hyperparameters_k.csv' OUTPUT_CSV=$(OUT_DIR)/hyperparameters.csv REMOVE_CSV=0

parameter-scaling: setup
	@python code/main.py --hid_dim 16 --seed 42 --n_times 3 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' 'GAT' --output_path $(OUT_DIR)/parameter_scaling_1.csv
	@python code/main.py --hid_dim 32 --seed 42 --n_times 3 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' 'GAT' --output_path $(OUT_DIR)/parameter_scaling_2.csv
	@python code/main.py --hid_dim 64 --seed 42 --n_times 3 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' 'GAT' --output_path $(OUT_DIR)/parameter_scaling_3.csv
	@python code/main.py --hid_dim 128 --seed 42 --n_times 3 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' 'GAT' --output_path $(OUT_DIR)/parameter_scaling_4.csv
	@python code/main.py --hid_dim 256 --seed 42 --n_times 3 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' 'GAT' --output_path $(OUT_DIR)/parameter_scaling_5.csv
	@python code/main.py --hid_dim 512 --seed 42 --n_times 3 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' 'GAT' --output_path $(OUT_DIR)/parameter_scaling_6.csv
	@python code/main.py --hid_dim 1024 --seed 42 --n_times 3 --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' 'GAT' --output_path $(OUT_DIR)/parameter_scaling_7.csv
	$(MAKE) combine_csv ARGS='$(OUT_DIR)/parameter_scaling_1.csv $(OUT_DIR)/parameter_scaling_2.csv $(OUT_DIR)/parameter_scaling_3.csv $(OUT_DIR)/parameter_scaling_4.csv $(OUT_DIR)/parameter_scaling_5.csv $(OUT_DIR)/parameter_scaling_6.csv $(OUT_DIR)/parameter_scaling_7.csv' OUTPUT_CSV=$(OUT_DIR)/parameter_scaling.csv REMOVE_CSV=1
