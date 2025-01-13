.PHONY: setup combine_csv nifa

DEVICE ?= 0

setup:
	mkdir -p output

combine_csv:
	@awk 'NR == 1 {print; next} FNR > 1' $(ARGS) > $(OUTPUT_CSV)

# Runs the main experiment of the paper (results in Table 2)
# The results are saved in the output folder at nifa.csv
nifa: setup
	@python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device $(DEVICE) --mode 'degree' --output_path output/nifa_pokec_z.csv --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC' 'FairGNN' 'FairVGNN' 'FairSIN'

	@python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --before --device $(DEVICE) --mode 'degree' --output_path output/nifa_pokec_n.csv --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC' 'FairGNN' 'FairVGNN' 'FairSIN'

	@python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --epochs 500 --before --device $(DEVICE) --mode 'degree' --output_path output/nifa_dblp.csv --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC' 'FairGNN' 'FairVGNN' 'FairSIN'

	$(MAKE) combine_csv ARGS='output/nifa_pokec_z.csv output/nifa_pokec_n.csv output/nifa_dblp.csv' OUTPUT_CSV=output/nifa.csv
