.PHONY: setup combine_csv all nifa defense hyperparam-alpha hyperparam-beta hyperparam-perturbation hyperparam-k hyperparameters

DEVICE ?= 0

setup:
	mkdir -p output

all: nifa defense hyperparameters

combine_csv:
	@awk 'NR == 1 {print; next} FNR > 1' $(ARGS) > $(OUTPUT_CSV)
	@if [ "$(REMOVE_CSV)" = "1" ] || [ "$(REMOVE_CSV)" = "true" ]; then \
		rm -f $(ARGS); \
	fi

# Runs the main experiment of the paper (results in Table 2)
# The results are saved in the output folder at nifa.csv
nifa: setup
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --mode 'degree' --output_path output/nifa_pokec_z.csv --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC' 'FairGNN' 'FairVGNN' 'FairSIN'

	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --before --device $(DEVICE) --mode 'degree' --output_path output/nifa_pokec_n.csv --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC' 'FairGNN' 'FairVGNN' 'FairSIN'

	@python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --epochs 500 --before --device $(DEVICE) --mode 'degree' --output_path output/nifa_dblp.csv --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC' 'FairGNN' 'FairVGNN' 'FairSIN'

	$(MAKE) combine_csv ARGS='output/nifa_pokec_z.csv output/nifa_pokec_n.csv output/nifa_dblp.csv' OUTPUT_CSV=output/nifa.csv REMOVE_CSV=1


defense: setup
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.1 --output_path output/defense_1.csv

	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.2 --output_path output/defense_2.csv

	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.3 --output_path output/defense_3.csv

	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.4 --output_path output/defense_4.csv

	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.5 --output_path output/defense_5.csv

	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --models 'APPNP' --defense 0.6 --output_path output/defense_6.csv

	$(MAKE) combine_csv ARGS='output/defense_1.csv output/defense_2.csv output/defense_3.csv output/defense_4.csv output/defense_5.csv output/defense_6.csv' OUTPUT_CSV=output/defense.csv REMOVE_CSV=1


# analysis on alpha: 0.005, 0.01, 0.02, 0.05, 0.1, 0.2
# produces Figure A2
hyperparam-alpha: setup
	# pokec_z
	@python main.py --dataset pokec_z --alpha 0.005 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_1.csv
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_2.csv
	@python main.py --dataset pokec_z --alpha 0.02 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_3.csv
	@python main.py --dataset pokec_z --alpha 0.05 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_4.csv
	@python main.py --dataset pokec_z --alpha 0.1 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_5.csv
	@python main.py --dataset pokec_z --alpha 0.2 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_6.csv
	# pokec_n
	@python main.py --dataset pokec_n --alpha 0.005 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_7.csv
	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_8.csv
	@python main.py --dataset pokec_n --alpha 0.02 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_9.csv
	@python main.py --dataset pokec_n --alpha 0.05 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_10.csv
	@python main.py --dataset pokec_n --alpha 0.1 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_11.csv
	@python main.py --dataset pokec_n --alpha 0.2 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_12.csv
	# dblp
	@python main.py --dataset dblp --alpha 0.005 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_13.csv
	@python main.py --dataset dblp --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_14.csv
	@python main.py --dataset dblp --alpha 0.02 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_15.csv
	@python main.py --dataset dblp --alpha 0.05 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_16.csv
	@python main.py --dataset dblp --alpha 0.1 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_17.csv
	@python main.py --dataset dblp --alpha 0.2 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_alpha_18.csv
	# combine
	$(MAKE) combine_csv ARGS='output/hyperparameters_alpha_1.csv output/hyperparameters_alpha_2.csv output/hyperparameters_alpha_3.csv output/hyperparameters_alpha_4.csv output/hyperparameters_alpha_5.csv output/hyperparameters_alpha_6.csv output/hyperparameters_alpha_7.csv output/hyperparameters_alpha_8.csv output/hyperparameters_alpha_9.csv output/hyperparameters_alpha_10.csv output/hyperparameters_alpha_11.csv output/hyperparameters_alpha_12.csv output/hyperparameters_alpha_13.csv output/hyperparameters_alpha_14.csv output/hyperparameters_alpha_15.csv output/hyperparameters_alpha_16.csv output/hyperparameters_alpha_17.csv output/hyperparameters_alpha_18.csv' OUTPUT_CSV=output/hyperparameters_alpha.csv REMOVE_CSV=1


# analysis on beta: 2, 4, 8, 16
# produces Figure A3
hyperparam-beta: setup
	# pokec_z
	@python main.py --dataset pokec_z --alpha 0.01 --beta 2 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_1.csv
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_2.csv
	@python main.py --dataset pokec_z --alpha 0.01 --beta 8 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_3.csv
	@python main.py --dataset pokec_z --alpha 0.01 --beta 16 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_4.csv
	# pokec_n
	@python main.py --dataset pokec_n --alpha 0.01 --beta 2 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_5.csv
	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_6.csv
	@python main.py --dataset pokec_n --alpha 0.01 --beta 8 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_7.csv
	@python main.py --dataset pokec_n --alpha 0.01 --beta 16 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_8.csv
	# dblp
	@python main.py --dataset dblp --alpha 0.1 --beta 2 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_9.csv
	@python main.py --dataset dblp --alpha 0.1 --beta 4 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_10.csv
	@python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_11.csv
	@python main.py --dataset dblp --alpha 0.1 --beta 16 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_beta_12.csv
	# combine
	$(MAKE) combine_csv ARGS='output/hyperparameters_beta_1.csv output/hyperparameters_beta_2.csv output/hyperparameters_beta_3.csv output/hyperparameters_beta_4.csv output/hyperparameters_beta_5.csv output/hyperparameters_beta_6.csv output/hyperparameters_beta_7.csv output/hyperparameters_beta_8.csv output/hyperparameters_beta_9.csv output/hyperparameters_beta_10.csv output/hyperparameters_beta_11.csv output/hyperparameters_beta_12.csv' OUTPUT_CSV=output/hyperparameters_beta.csv REMOVE_CSV=1


# analysis on perturbation rate: 0.005, 0.01, 0.02, 0.03
# produces Figure A4
hyperparam-perturbation: setup
	# pokec_z
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 51 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_1.csv
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_2.csv
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 204 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_3.csv
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 306 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_4.csv
	# pokec_n
	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 43 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_5.csv
	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_6.csv
	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 174 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_7.csv
	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 261 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_8.csv
	# dblp
	@python main.py --dataset dblp --alpha 0.1 --beta 4 --node 16 --edge 24 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_9.csv
	@python main.py --dataset dblp --alpha 0.1 --beta 4 --node 32 --edge 24 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_10.csv
	@python main.py --dataset dblp --alpha 0.1 --beta 4 --node 64 --edge 24 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_11.csv
	@python main.py --dataset dblp --alpha 0.1 --beta 4 --node 96 --edge 24 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_perturbation_12.csv
	# combine
	$(MAKE) combine_csv ARGS='output/hyperparameters_perturbation_1.csv output/hyperparameters_perturbation_2.csv output/hyperparameters_perturbation_3.csv output/hyperparameters_perturbation_4.csv output/hyperparameters_perturbation_5.csv output/hyperparameters_perturbation_6.csv output/hyperparameters_perturbation_7.csv output/hyperparameters_perturbation_8.csv output/hyperparameters_perturbation_9.csv output/hyperparameters_perturbation_10.csv output/hyperparameters_perturbation_11.csv output/hyperparameters_perturbation_12.csv' OUTPUT_CSV=output/hyperparameters_perturbation.csv REMOVE_CSV=1


# analysis on k (ratio): 0.1, 0.25, 0.5, 0.75
# produces Figure A5
hyperparam-k:
	#pokec_z
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --ratio 0.1 --node 102 --edge 25 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_1.csv
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --ratio 0.25 --node 102 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_2.csv
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --ratio 0.5 --node 102 --edge 100 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_3.csv
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --ratio 0.75 --node 102 --edge 150 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_4.csv
	#pokec_n
	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --ratio 0.1 --node 87 --edge 21 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_5.csv
	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --ratio 0.25 --node 87 --edge 50 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_6.csv
	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --ratio 0.5 --node 87 --edge 100 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_7.csv
	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --ratio 0.75 --node 87 --edge 150 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_8.csv
	#dblp
	@python main.py --dataset dblp --alpha 0.1 --beta 4 --ratio 0.1 --node 32 --edge 6 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_9.csv
	@python main.py --dataset dblp --alpha 0.1 --beta 4 --ratio 0.25 --node 32 --edge 15 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_10.csv
	@python main.py --dataset dblp --alpha 0.1 --beta 4 --ratio 0.5 --node 32 --edge 30 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_11.csv
	@python main.py --dataset dblp --alpha 0.1 --beta 4 --ratio 0.75 --node 32 --edge 45 --device $(DEVICE) --models 'GCN' --output_path output/hyperparam_k_12.csv
	# combine
	$(MAKE) combine_csv ARGS='output/hyperparameters_k_1.csv output/hyperparameters_k_2.csv output/hyperparameters_k_3.csv output/hyperparameters_k_4.csv output/hyperparameters_k_5.csv output/hyperparameters_k_6.csv output/hyperparameters_k_7.csv output/hyperparameters_k_8.csv output/hyperparameters_k_9.csv output/hyperparameters_k_10.csv output/hyperparameters_k_11.csv output/hyperparameters_k_12.csv' OUTPUT_CSV=output/hyperparameters_k.csv REMOVE_CSV=1


hyperparameters: hyperparam-alpha hyperparam-beta hyperparam-perturbation hyperparam-k
	$(MAKE) combine_csv ARGS='output/hyperparameters_alpha.csv output/hyperparameters_beta.csv output/hyperparameters_perturbation.csv output/hyperparameters_k.csv' OUTPUT_CSV=output/hyperparameters.csv REMOVE_CSV=0
