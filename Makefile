.PHONY: setup combine_csv nifa all

DEVICE ?= 0

setup:
	mkdir -p output

all: nifa

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
