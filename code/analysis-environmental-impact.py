import pandas as pd


df = pd.read_json('output/nifa_emissions.jsonl', lines=True)

PUE = 1.21
intensity = 270 # gCO2/kWh

print(df.energy_consumed.sum() * PUE * intensity / 1000)
